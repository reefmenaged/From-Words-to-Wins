"""Unified trainer for DeBERTa, ModernBERT and the features-only ablation."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import (
    DEFAULT_DROPOUT,
    DEFAULT_ENCODER_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FREQUENT_PLAYER_THRESHOLD,
    DEFAULT_HEAD_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    INTERVIEW_COLUMN,
    LABEL_COLUMN,
    MODEL_SPECS,
)
from .data import FeaturePreprocessor, grouped_proportional_split, load_dataset, prepare_interviews
from .runtime import resolve_device
from .model import (
    BertATMHead,
    CachedEmbeddingDataset,
    MTOnlyClassifier,
    MTOnlyDataset,
    collate_cached_batch,
)

SUPPORTED_MODES = ("deberta", "modernbert", "features")
RESULTS_FILENAME = "test_results.csv"
EMBEDDINGS_FILENAME = "test_interview_embeddings_after_attention.pt"


def set_seed(seed: int) -> None:
    """Set all random seeds used by the original implementation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_data(
    csv_path: str | Path,
    test_size: float,
    frequent_player_threshold: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared data path for all three modes, preventing split/preprocessing drift."""
    df = load_dataset(csv_path)
    train_df, test_df = grouped_proportional_split(
        df,
        test_size=test_size,
        frequent_player_threshold=frequent_player_threshold,
        seed=seed,
    )
    preprocessor = FeaturePreprocessor.fit(train_df)
    x_train = preprocessor.transform_frame(train_df)
    x_test = preprocessor.transform_frame(test_df)
    y_train = train_df[LABEL_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_df[LABEL_COLUMN].to_numpy(dtype=np.float32)
    return train_df, test_df, x_train, x_test, y_train, y_test


def _metric_values(labels: List[int], probabilities: List[float]) -> Dict[str, Any]:
    predictions = [1 if p >= 0.5 else 0 for p in probabilities]
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def _evaluate_bert(
    model: BertATMHead,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Evaluate the BERT-A-TM head and collect post-attention test embeddings."""
    model.eval()
    probabilities: List[float] = []
    labels: List[int] = []
    interview_vectors: List[torch.Tensor] = []
    with torch.inference_mode():
        for qa, mask, features, y in loader:
            qa = qa.to(device)
            mask = mask.to(device)
            features = features.to(device)
            logits, _, interview_vector = model.forward_with_interview_vector(
                qa, mask, features
            )
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(y.int().tolist())
            interview_vectors.append(interview_vector.detach().cpu())
    return _metric_values(labels, probabilities), torch.cat(interview_vectors, dim=0)


def _evaluate_features(
    model: MTOnlyClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate the features-only classifier."""
    model.eval()
    probabilities: List[float] = []
    labels: List[int] = []
    with torch.inference_mode():
        for features, y in loader:
            logits = model(features.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(y.int().tolist())
    return _metric_values(labels, probabilities)


def _save_results_table(
    path: Path,
    *,
    mode: str,
    hf_model: Optional[str],
    device: torch.device,
    train_rows: int,
    test_rows: int,
    metrics: Dict[str, Any],
) -> None:
    """Write the single requested result table with decimal and percentage metrics."""
    row: Dict[str, Any] = {
        "training_mode": mode,
        "hf_encoder": hf_model or "none",
        "device": str(device),
        "train_rows": train_rows,
        "test_rows": test_rows,
    }
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        value = metrics[name]
        row[name] = value
        row[f"{name}_percent"] = None if value is None else value * 100.0
    for name in ("tn", "fp", "fn", "tp"):
        row[name] = metrics[name]
    pd.DataFrame([row]).to_csv(path, index=False)


def _train_bert_mode(
    *,
    mode: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    device: str,
    max_length: int,
    encoder_batch_size: int,
    head_batch_size: int,
    epochs: int,
    learning_rate: float,
    dropout: float,
    seed: int,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.device]:
    """Train either text mode; only MODEL_SPECS[mode] changes between them."""
    from .encoders import FrozenQAEncoder

    model_name = MODEL_SPECS[mode]
    train_interviews = prepare_interviews(train_df[INTERVIEW_COLUMN].tolist())
    test_interviews = prepare_interviews(test_df[INTERVIEW_COLUMN].tolist())

    encoder = FrozenQAEncoder(model_name=model_name, device=device, max_length=max_length)
    print(f"Encoder device: {encoder.device} | frozen encoder: {model_name}")
    print(
        f"Train rows: {len(train_df)} | Test rows: {len(test_df)} | "
        f"MT features: {x_train.shape[1]}"
    )

    train_embeddings = encoder.encode_interviews(
        train_interviews, batch_size=encoder_batch_size
    )
    test_embeddings = encoder.encode_interviews(
        test_interviews, batch_size=encoder_batch_size
    )
    hidden_size = encoder.hidden_size

    # The frozen encoder is not part of head training, matching the original code.
    del encoder.model
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_dataset = CachedEmbeddingDataset(train_embeddings, x_train, y_train)
    test_dataset = CachedEmbeddingDataset(test_embeddings, x_test, y_test)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=head_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_cached_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=head_batch_size,
        shuffle=False,
        collate_fn=collate_cached_batch,
    )

    head_device = resolve_device(device)
    model = BertATMHead(
        hidden_size=hidden_size,
        feature_dim=x_train.shape[1],
        dropout=dropout,
    ).to(head_device)
    optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for qa, mask, features, labels in train_loader:
            qa = qa.to(head_device)
            mask = mask.to(head_device)
            features = features.to(head_device)
            labels = labels.to(head_device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(qa, mask, features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_n = labels.shape[0]
            total_loss += float(loss.item()) * batch_n
            seen += batch_n

        mean_loss = total_loss / max(seen, 1)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d}/{epochs} - train_loss={mean_loss:.6f}")

    metrics, interview_vectors = _evaluate_bert(model, test_loader, head_device)
    return metrics, interview_vectors, head_device


def _train_features_mode(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    device: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    dropout: float,
    seed: int,
) -> Tuple[Dict[str, Any], torch.device]:
    """Train the original MT/features-only ablation without interview text."""
    train_dataset = MTOnlyDataset(x_train, y_train)
    test_dataset = MTOnlyDataset(x_test, y_test)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_device = resolve_device(device)
    model = MTOnlyClassifier(feature_dim=x_train.shape[1], dropout=dropout).to(model_device)
    optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"Features-only device: {model_device}")
    print(
        f"Train rows: {len(y_train)} | Test rows: {len(y_test)} | "
        f"MT features: {x_train.shape[1]} | Interview used: False"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for features, labels in train_loader:
            features = features.to(model_device)
            labels = labels.to(model_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_n = labels.shape[0]
            total_loss += float(loss.item()) * batch_n
            seen += batch_n

        mean_loss = total_loss / max(seen, 1)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d}/{epochs} - train_loss={mean_loss:.6f}")

    return _evaluate_features(model, test_loader, model_device), model_device


def train(
    *,
    mode: str,
    csv_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    test_size: float = DEFAULT_TEST_SIZE,
    frequent_player_threshold: int = DEFAULT_FREQUENT_PLAYER_THRESHOLD,
    seed: int = DEFAULT_SEED,
    max_length: int = DEFAULT_MAX_LENGTH,
    encoder_batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    head_batch_size: int = DEFAULT_HEAD_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    dropout: float = DEFAULT_DROPOUT,
) -> Dict[str, Any]:
    """Run one of the three experiments and save only the requested final outputs."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of {SUPPORTED_MODES}; got {mode!r}")

    set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Prevent stale files from a previous run from looking like current outputs.
    for filename in (RESULTS_FILENAME, EMBEDDINGS_FILENAME):
        path = out / filename
        if path.exists():
            path.unlink()

    train_df, test_df, x_train, x_test, y_train, y_test = _prepare_data(
        csv_path,
        test_size=test_size,
        frequent_player_threshold=frequent_player_threshold,
        seed=seed,
    )

    if mode == "features":
        metrics, model_device = _train_features_mode(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            device=device,
            batch_size=head_batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dropout=dropout,
            seed=seed,
        )
        hf_model = None
    else:
        metrics, interview_vectors, model_device = _train_bert_mode(
            mode=mode,
            train_df=train_df,
            test_df=test_df,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            device=device,
            max_length=max_length,
            encoder_batch_size=encoder_batch_size,
            head_batch_size=head_batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dropout=dropout,
            seed=seed,
        )
        hf_model = MODEL_SPECS[mode]
        torch.save(
            {
                "training_mode": mode,
                "hf_encoder": hf_model,
                "source_rows": torch.as_tensor(
                    test_df["_source_row"].to_numpy(), dtype=torch.long
                ),
                "embeddings": interview_vectors,
            },
            out / EMBEDDINGS_FILENAME,
        )

    _save_results_table(
        out / RESULTS_FILENAME,
        mode=mode,
        hf_model=hf_model,
        device=model_device,
        train_rows=len(train_df),
        test_rows=len(test_df),
        metrics=metrics,
    )

    print("\nTest metrics:")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"\nSaved: {out / RESULTS_FILENAME}")
    if mode != "features":
        print(f"Saved: {out / EMBEDDINGS_FILENAME}")
    else:
        print("No .pt interview embedding is produced in features mode (there is no interview attention).")

    return {
        "mode": mode,
        "output_dir": str(out),
        "results_csv": str(out / RESULTS_FILENAME),
        "embeddings_pt": None if mode == "features" else str(out / EMBEDDINGS_FILENAME),
        "accuracy": metrics["accuracy"],
        "accuracy_percent": metrics["accuracy"] * 100.0,
    }
