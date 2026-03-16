"""Seed management and deterministic settings for reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic(enabled: bool = True) -> None:
    """Enable or disable deterministic mode for PyTorch.

    Note: deterministic mode may reduce performance.

    Args:
        enabled: Whether to enable deterministic operations.
    """
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled
    torch.use_deterministic_algorithms(enabled, warn_only=True)
