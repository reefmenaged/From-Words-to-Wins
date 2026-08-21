"""Single command-line entry point for all three training experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from tennis_atm.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_DROPOUT,
    DEFAULT_ENCODER_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FREQUENT_PLAYER_THRESHOLD,
    DEFAULT_HEAD_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
)
from tennis_atm.training import SUPPORTED_MODES, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one of three From-Words-to-Wins experiments with one command: "
            "DeBERTa BERT-A-TM, ModernBERT BERT-A-TM, or features-only."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODES,
        help="Training mode: deberta, modernbert, or features.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "Optional CSV override. By default training starts from "
            "data/processed/player_tournament_interview_dataset.csv in the Git repository."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/<selected-model>.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--encoder-batch-size", type=int, default=DEFAULT_ENCODER_BATCH_SIZE)
    parser.add_argument("--head-batch-size", type=int, default=DEFAULT_HEAD_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument(
        "--frequent-player-threshold",
        type=int,
        default=DEFAULT_FREQUENT_PLAYER_THRESHOLD,
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (Path("outputs") / args.model)
    train(
        mode=args.model,
        csv_path=args.csv,
        output_dir=output_dir,
        device=args.device,
        epochs=args.epochs,
        encoder_batch_size=args.encoder_batch_size,
        head_batch_size=args.head_batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        seed=args.seed,
        test_size=args.test_size,
        frequent_player_threshold=args.frequent_player_threshold,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
