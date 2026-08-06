"""Shared pytest fixtures.

These build a tiny *synthetic* dataset on disk in the Roboflow/YOLO layout so the data,
EDA and conversion logic can be exercised end-to-end without the real ~300 MB download and
without a GPU. A second fixture exposes the real raw dataset when it happens to be present,
so the suite also smoke-tests against actual labels in dev.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 always available per requirements
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CAR = REPO_ROOT / "dataset" / "raw" / "car"


def _write_label(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(str(v) for v in r) for r in rows) + ("\n" if rows else ""))


def _write_image(path: Path, w: int = 32, h: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (np.random.rand(h, w, 3) * 255).astype("uint8")
    cv2.imwrite(str(path), img)


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """A minimal, fully-valid YOLO dataset with a known box layout.

    Layout (3 classes, ids 0..2):
        train: img0 (1 box cls0), img1 (2 boxes cls1, cls2)
        valid: img2 (1 box cls0)
        test : img3 (1 box cls2)

    Total boxes per class: cls0=2, cls1=1, cls2=2.
    """
    root = tmp_path / "cardetection"

    layout = {
        "train": {
            "img0": [(0, 0.5, 0.5, 0.2, 0.2)],
            "img1": [(1, 0.25, 0.25, 0.1, 0.1), (2, 0.75, 0.75, 0.3, 0.4)],
        },
        "valid": {"img2": [(0, 0.5, 0.5, 0.5, 0.5)]},
        "test": {"img3": [(2, 0.5, 0.5, 0.1, 0.1)]},
    }
    for split, items in layout.items():
        for stem, boxes in items.items():
            _write_image(root / split / "images" / f"{stem}.jpg")
            _write_label(root / split / "labels" / f"{stem}.txt", boxes)

    # a data.yaml so format detection + config sync have something to read
    (root / "data.yaml").write_text(
        "nc: 3\nnames: ['Green Light', 'Red Light', 'Stop']\n"
    )
    return root


@pytest.fixture
def classes_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "classes.yaml"
    p.write_text("nc: 3\nnames:\n  0: Green Light\n  1: Red Light\n  2: Stop\n")
    return p


@pytest.fixture
def raw_car_root() -> Path:
    """The real raw dataset root, or skip the test if it isn't downloaded locally."""
    if not (RAW_CAR / "data.yaml").exists():
        pytest.skip("real raw dataset not present (dataset/raw/car)")
    return RAW_CAR
