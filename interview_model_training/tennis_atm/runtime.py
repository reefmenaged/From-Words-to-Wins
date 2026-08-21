"""Small runtime helpers shared by training and encoder code."""
from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda and fail clearly for unavailable requested CUDA."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device)
