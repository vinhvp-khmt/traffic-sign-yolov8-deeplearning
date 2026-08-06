"""Shared plotting helpers so every EDA/result figure has a consistent look and save path.

Uses a non-interactive matplotlib backend so the scripts run headless (Colab, CI) without a
display. Notebooks can still call `plt.show()` after these helpers return the figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; set before pyplot import side effects matter
import matplotlib.pyplot as plt  # noqa: E402

from .paths import ensure_dir  # noqa: E402

# Project-wide style. Kept minimal so it layers cleanly over seaborn if a script imports it.
plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "savefig.bbox": "tight",
    }
)


def save_fig(fig, path: Path, *, close: bool = True) -> Path:
    """Save `fig` to `path` (creating parent dirs) and optionally close it. Returns the path."""
    ensure_dir(Path(path).parent)
    fig.savefig(path)
    if close:
        plt.close(fig)
    return Path(path)
