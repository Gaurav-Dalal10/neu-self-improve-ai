"""Small utility helpers shared across modules."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach a tensor and move to CPU numpy."""
    return t.detach().cpu().numpy()


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance between predictions and targets."""
    var = np.var(y_true)
    if var < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / (var + 1e-8))


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
