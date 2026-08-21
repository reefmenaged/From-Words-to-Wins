"""Project-wide constants for the unified interview-model training pipeline."""
from __future__ import annotations

from pathlib import Path

# This folder is intended to be copied as one directory into the Git repository.
# The dataset remains in the repository-level data/ directory.
TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "player_tournament_interview_dataset.csv"
)

MODEL_SPECS = {
    "deberta": "microsoft/deberta-v3-large",
    "modernbert": "answerdotai/ModernBERT-large",
}

LABEL_COLUMN = "current_finish_at_least_recent_average"
PLAYER_COLUMN = "player_name"
INTERVIEW_COLUMN = "first_pre_match_interview_qa_json"

# Columns excluded from the numerical MT branch in the original implementation.
DROP_COLUMNS = [
    "tourney_year",
    "tourney_name",
    "tourney_date",
    "player_name",
    "player_ioc",
    "tournament_finish_score",
    "first_pre_match_interview_date",
    "first_pre_match_interview_url",
    "opponent_difficulty_score",
    "opponent_difficulty_confidence",
    "num_questions",
    "avg_answer_length_words",
    "total_answer_words",
]

AGE_COLUMNS = ["player_age"]
RANK_COLUMNS = [
    "player_rank",
    "prev1_player_rank",
    "prev2_player_rank",
    "prev3_player_rank",
]
LEVEL_COLUMNS = [
    "tournament_level_score",
    "prev1_tournament_level_score",
    "prev2_tournament_level_score",
    "prev3_tournament_level_score",
]

DEFAULT_TEST_SIZE = 0.25
DEFAULT_FREQUENT_PLAYER_THRESHOLD = 10
DEFAULT_SEED = 212
DEFAULT_MAX_LENGTH = 512
DEFAULT_ENCODER_BATCH_SIZE = 4
DEFAULT_HEAD_BATCH_SIZE = 8
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_DROPOUT = 0.2
