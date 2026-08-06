"""Smoke tests against the *real* raw dataset, skipped automatically when it's absent.

These don't assert exact counts (the dataset can be re-versioned) but they guard the
contract the rest of the pipeline relies on: 15 classes, YOLO format, parseable labels with
in-range coordinates.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.data import inspect_dataset as ins


def test_raw_has_data_yaml_with_15_classes(raw_car_root: Path):
    cfg = yaml.safe_load((raw_car_root / "data.yaml").read_text())
    assert cfg["nc"] == 15
    assert len(cfg["names"]) == 15


def test_raw_detected_as_yolo(raw_car_root: Path):
    assert "YOLO" in ins.detect_format(raw_car_root)


def test_raw_labels_parse_and_are_in_range(raw_car_root: Path):
    """Sample up to 50 label files across the export and confirm coordinates are normalized."""
    label_files = list(raw_car_root.rglob("labels/*.txt"))[:50]
    if not label_files:
        import pytest
        pytest.skip("no label files found under raw dataset")

    checked = 0
    for lf in label_files:
        for cls, xc, yc, w, h in ins.parse_label_file(lf):
            assert 0 <= cls < 15
            for v in (xc, yc, w, h):
                assert 0.0 <= v <= 1.0
            checked += 1
    assert checked > 0
