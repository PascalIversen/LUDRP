"""Reproducibility helper: seed the Python, NumPy and Torch RNGs used by the experiments."""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0) -> int:
    """Seed all RNGs that drive the benchmark and active-learning runs.

    Covers the ``numpy`` global state used for the leave-cell-line-out early-stopping split, the
    ensemble train/validation resampling, the synthetic-OOD perturbation and the active-learning
    random baseline, plus torch weight initialisation and dropout. Pass a fixed seed for
    reproducible runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
