"""Trainable attention/classification heads used by the three training modes."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class QAAttention(nn.Module):
    """Additive attention over the sequence of frozen Q-A vectors."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=True)
        self.context_vector = nn.Parameter(torch.empty(hidden_size))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.normal_(self.context_vector, mean=0.0, std=0.02)

    def forward(
        self,
        qa_vectors: torch.Tensor,
        qa_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = torch.tanh(self.projection(qa_vectors))
        scores = torch.einsum("bnh,h->bn", projected, self.context_vector)
        scores = scores.masked_fill(~qa_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        interview_vector = torch.sum(qa_vectors * weights.unsqueeze(-1), dim=1)
        return interview_vector, weights


class BertATMHead(nn.Module):
    """Q-A attention -> concatenate MT features -> dropout -> binary linear classifier."""

    def __init__(self, hidden_size: int, feature_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.attention = QAAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + feature_dim, 1)

    def forward(
        self,
        qa_vectors: torch.Tensor,
        qa_mask: torch.Tensor,
        mt_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, weights, _ = self.forward_with_interview_vector(
            qa_vectors, qa_mask, mt_features
        )
        return logits, weights

    def forward_with_interview_vector(
        self,
        qa_vectors: torch.Tensor,
        qa_mask: torch.Tensor,
        mt_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        interview_vector, weights = self.attention(qa_vectors, qa_mask)
        combined = torch.cat([interview_vector, mt_features], dim=-1)
        logits = self.classifier(self.dropout(combined)).squeeze(-1)
        return logits, weights, interview_vector


class MTOnlyClassifier(nn.Module):
    """Features-only ablation: normalized MT features -> dropout -> linear logit."""

    def __init__(self, feature_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(feature_dim), 1)

    def forward(self, mt_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(mt_features)).squeeze(-1)


class CachedEmbeddingDataset(Dataset):
    """Dataset for already-computed frozen Q-A vectors."""

    def __init__(self, interview_embeddings: Sequence[torch.Tensor], mt_features, labels) -> None:
        if len(interview_embeddings) != len(mt_features) or len(mt_features) != len(labels):
            raise ValueError("Embeddings, MT features, and labels must have the same length.")
        self.interview_embeddings = list(interview_embeddings)
        self.mt_features = torch.as_tensor(mt_features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.interview_embeddings[idx], self.mt_features[idx], self.labels[idx]


class MTOnlyDataset(Dataset):
    """Dataset for the features-only ablation."""

    def __init__(self, mt_features, labels) -> None:
        if len(mt_features) != len(labels):
            raise ValueError("MT features and labels must have the same length.")
        self.mt_features = torch.as_tensor(mt_features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.mt_features[idx], self.labels[idx]


def collate_cached_batch(batch):
    """Pad variable numbers of Q-A pairs and return a boolean attention mask."""
    embeddings, features, labels = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in embeddings], dtype=torch.long)
    padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    positions = torch.arange(padded.shape[1]).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    return padded, mask, torch.stack(features), torch.stack(labels)
