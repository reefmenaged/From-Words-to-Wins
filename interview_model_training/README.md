# Unified interview-model training

This directory contains only the code for the three training modes. Copy the whole
`interview_model_training/` directory into the root of the Git repository. The data
is **not** duplicated inside this directory.

## Dataset

By default, all modes load exactly:

```text
data/processed/player_tournament_interview_dataset.csv
```

The path is resolved from the Git repository root, so the expected layout is:

```text
<repo>/
├── data/
│   └── processed/
│       └── player_tournament_interview_dataset.csv
└── interview_model_training/
    ├── train.py
    └── tennis_atm/
```

## Training

Run from the repository root:

```bash
python interview_model_training/train.py --model deberta --device auto
python interview_model_training/train.py --model modernbert --device auto
python interview_model_training/train.py --model features --device auto
```

`--device auto` uses CUDA when available and otherwise uses CPU. You can force a
device with `--device cpu` or `--device cuda`.

The default training hyperparameters and split behavior are preserved from the
existing implementations. Optional overrides are available through `python
interview_model_training/train.py --help`.

## Outputs

Default outputs are written under the current working directory:

```text
outputs/deberta/
outputs/modernbert/
outputs/features/
```

For `deberta` and `modernbert`, the final deliverables are:

- `test_interview_embeddings_after_attention.pt` — test interview embeddings after attention.
- `test_results.csv` — test metrics, including accuracy.

For `features`, there is no interview-attention branch, so the final deliverable is:

- `test_results.csv` — test metrics, including accuracy.

There is intentionally no `requirements.txt` in this directory.
