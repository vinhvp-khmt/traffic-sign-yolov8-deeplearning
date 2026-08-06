"""Single place to seed all RNGs so training/eval runs are reproducible.

Importing torch/numpy lazily keeps this safe to import in CPU-only environments where the
DL stack isn't installed (the EDA/data modules can still use `seed_everything` for the
stdlib + numpy RNGs).
"""
from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and (if available) PyTorch RNGs. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
