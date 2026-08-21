"""Dataset loading, deterministic splitting, interview parsing and MT preprocessing."""
from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .config import (
    AGE_COLUMNS,
    DROP_COLUMNS,
    INTERVIEW_COLUMN,
    LABEL_COLUMN,
    LEVEL_COLUMNS,
    PLAYER_COLUMN,
    RANK_COLUMNS,
)

QAPair = Tuple[str, str]


def validate_dataframe(df: pd.DataFrame) -> None:
    """Fail early if the training CSV is not the expected processed table."""
    required = {
        PLAYER_COLUMN,
        LABEL_COLUMN,
        INTERVIEW_COLUMN,
        *DROP_COLUMNS,
        *AGE_COLUMNS,
        *RANK_COLUMNS,
        *LEVEL_COLUMNS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[LABEL_COLUMN].isna().any():
        raise ValueError(f"Label column {LABEL_COLUMN!r} contains missing values.")
    labels = set(df[LABEL_COLUMN].astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{LABEL_COLUMN!r} must be binary 0/1; found {sorted(labels)}")


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the processed 830-row interview dataset used by all three modes.

    The default path is repository-relative:
    data/processed/player_tournament_interview_dataset.csv
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    df = pd.read_csv(path).copy()
    validate_dataframe(df)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df["_source_row"] = np.arange(len(df), dtype=np.int64)
    return df

def add_split_groups(
    df: pd.DataFrame,
    frequent_player_threshold: int = 10,
) -> pd.DataFrame:
    """Frequent players get their own group; all others share one group."""
    out = df.copy()
    counts = out[PLAYER_COLUMN].value_counts()
    frequent = set(counts[counts > frequent_player_threshold].index.tolist())
    out["_split_group"] = np.where(
        out[PLAYER_COLUMN].isin(frequent),
        out[PLAYER_COLUMN],
        "__OTHER_PLAYERS__",
    )
    return out


def _split_one_group(
    group_df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split one player/group while stratifying the binary label when possible."""
    n = len(group_df)
    if n < 2:
        raise ValueError(f"Cannot split a group with only {n} row(s).")

    # Keep the original round-to-nearest allocation instead of sklearn's ceil
    # behavior for float test_size.
    n_train = int(round((1.0 - test_size) * n))
    n_train = min(max(n_train, 1), n - 1)
    n_test = n - n_train

    y = group_df[LABEL_COLUMN].to_numpy(dtype=int)
    values, counts = np.unique(y, return_counts=True)
    can_stratify = (
        len(values) >= 2
        and counts.min() >= 2
        and n_train >= len(values)
        and n_test >= len(values)
    )

    if can_stratify:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=n_train,
            test_size=n_test,
            random_state=seed,
        )
        train_pos, test_pos = next(splitter.split(np.zeros(n), y))
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        train_pos, test_pos = perm[:n_train], perm[n_train:]

    indices = group_df.index.to_numpy()
    return indices[train_pos], indices[test_pos]


def grouped_proportional_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    frequent_player_threshold: int = 10,
    seed: int = 212,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the original grouped ~75/25 train/test split exactly."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    work = add_split_groups(df, frequent_player_threshold)
    train_idx: List[int] = []
    test_idx: List[int] = []

    for group_name, group_df in work.groupby("_split_group", sort=True):
        group_seed = seed + (zlib.crc32(str(group_name).encode("utf-8")) % 1_000_000)
        tr, te = _split_one_group(group_df, test_size=test_size, seed=group_seed)
        train_idx.extend(tr.tolist())
        test_idx.extend(te.tolist())

    train_df = work.loc[sorted(train_idx)].copy()
    test_df = work.loc[sorted(test_idx)].copy()

    if set(train_df.index).intersection(test_df.index):
        raise AssertionError("Train/test overlap detected.")
    if len(train_df) + len(test_df) != len(work):
        raise AssertionError("Split did not account for all rows.")
    return train_df, test_df


def parse_interview(raw: Any) -> List[QAPair]:
    """Convert the QA JSON cell into an ordered list of (question, answer) pairs."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("Interview is empty.")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Interview string must be the QA JSON from first_pre_match_interview_qa_json."
            ) from exc

    pairs: List[QAPair] = []
    if isinstance(raw, Mapping):
        question_numbers = []
        for key in raw.keys():
            if isinstance(key, str) and key.startswith("question_"):
                suffix = key[len("question_") :]
                if suffix.isdigit():
                    question_numbers.append(int(suffix))
        for number in sorted(set(question_numbers)):
            q = raw.get(f"question_{number}")
            a = raw.get(f"answer_{number}")
            if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
                pairs.append((q.strip(), a.strip()))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, Mapping):
                q = item.get("question") or item.get("q")
                a = item.get("answer") or item.get("a")
            elif (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes, bytearray))
                and len(item) == 2
            ):
                q, a = item[0], item[1]
            else:
                raise ValueError(
                    "List interviews must contain {'question','answer'} objects "
                    "or [question, answer] pairs."
                )
            if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
                pairs.append((q.strip(), a.strip()))
    else:
        raise TypeError("Unsupported interview format.")

    if not pairs:
        raise ValueError("No non-empty question-answer pairs were found in the interview.")
    return pairs


def prepare_interviews(series: Iterable[Any]) -> List[List[QAPair]]:
    """Parse all interview cells while preserving row and Q-A order."""
    return [parse_interview(raw) for raw in series]


@dataclass
class FeaturePreprocessor:
    """Train-only normalization used by both BERT-A-TM and features-only runs."""

    feature_columns: List[str]
    age_stats: Dict[str, Dict[str, float]]
    rank_stats: Dict[str, Dict[str, float]]
    level_columns: List[str]
    missing_rank_sentinel: float = -1.0
    level_min: float = 0.0
    level_max: float = 4.0

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "FeaturePreprocessor":
        excluded = set(
            DROP_COLUMNS
            + [LABEL_COLUMN, INTERVIEW_COLUMN, "_source_row", "_split_group"]
        )
        feature_columns = [c for c in train_df.columns if c not in excluded]

        non_numeric = [
            c for c in feature_columns if not pd.api.types.is_numeric_dtype(train_df[c])
        ]
        if non_numeric:
            raise ValueError(
                f"All MT features must be numeric after dropping columns; non-numeric: {non_numeric}"
            )
        if train_df[feature_columns].isna().any().any():
            bad = train_df[feature_columns].columns[
                train_df[feature_columns].isna().any()
            ].tolist()
            raise ValueError(f"MT features contain NaN values: {bad}")

        age_stats: Dict[str, Dict[str, float]] = {}
        for col in AGE_COLUMNS:
            vals = train_df[col].astype(float).to_numpy()
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            if not math.isfinite(std) or std < 1e-12:
                std = 1.0
            age_stats[col] = {"mean": mean, "std": std}

        rank_stats: Dict[str, Dict[str, float]] = {}
        for col in RANK_COLUMNS:
            vals = train_df[col].astype(float).to_numpy()
            valid = vals[vals != -1.0]
            if len(valid) == 0:
                raise ValueError(f"No non--1 training values available for rank column {col!r}.")
            mean = float(valid.mean())
            std = float(valid.std(ddof=0))
            if not math.isfinite(std) or std < 1e-12:
                std = 1.0
            rank_stats[col] = {"mean": mean, "std": std}

        return cls(
            feature_columns=feature_columns,
            age_stats=age_stats,
            rank_stats=rank_stats,
            level_columns=list(LEVEL_COLUMNS),
        )

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply statistics learned only from the training partition."""
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        work = frame[self.feature_columns].copy().astype(float)
        if work.isna().any().any():
            bad = work.columns[work.isna().any()].tolist()
            raise ValueError(f"Feature input contains NaN values: {bad}")

        for col, stats in self.age_stats.items():
            work[col] = (work[col] - stats["mean"]) / stats["std"]

        for col, stats in self.rank_stats.items():
            vals = work[col].to_numpy(dtype=float, copy=True)
            mask = vals != self.missing_rank_sentinel
            vals[mask] = (vals[mask] - stats["mean"]) / stats["std"]
            work[col] = vals  # missing ranks remain exactly -1

        scale = self.level_max - self.level_min
        for col in self.level_columns:
            vals = work[col].to_numpy(dtype=float, copy=True)
            if np.any(vals < self.level_min) or np.any(vals > self.level_max):
                raise ValueError(
                    f"{col!r} must stay in [{self.level_min}, {self.level_max}] before scaling; "
                    f"got min={float(vals.min())}, max={float(vals.max())}."
                )
            work[col] = (vals - self.level_min) / scale

        return work.to_numpy(dtype=np.float32)
