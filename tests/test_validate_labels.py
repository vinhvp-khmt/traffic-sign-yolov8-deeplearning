"""Tests for the data-quality validator, including the bug-fix regressions."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data import validate_labels as vl


def _types(df):
    return set(df["issue_type"])


def test_clean_dataset_has_no_issues(synthetic_root: Path):
    df = vl.validate(synthetic_root, num_classes=3)
    assert df.empty
    assert vl.summarize(df) == "Total issues: 0"


def test_missing_label_detected(synthetic_root: Path):
    # drop one label file
    (synthetic_root / "train" / "labels" / "img0.txt").unlink()
    df = vl.validate(synthetic_root, num_classes=3)
    assert "missing_label" in _types(df)


def test_missing_image_detected(synthetic_root: Path):
    # add an orphan label with no image
    (synthetic_root / "train" / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    df = vl.validate(synthetic_root, num_classes=3)
    missing = df[df["issue_type"] == "missing_image"]
    assert len(missing) == 1
    assert "orphan" in missing.iloc[0]["image"]


def test_empty_label_detected(synthetic_root: Path):
    (synthetic_root / "train" / "labels" / "img0.txt").write_text("")
    df = vl.validate(synthetic_root, num_classes=3)
    assert "empty_label" in _types(df)


def test_invalid_class_id_high(synthetic_root: Path):
    (synthetic_root / "train" / "labels" / "img0.txt").write_text("99 0.5 0.5 0.2 0.2\n")
    df = vl.validate(synthetic_root, num_classes=3)
    assert "invalid_class_id" in _types(df)


def test_invalid_class_id_negative(synthetic_root: Path):
    """Regression: negative class ids must be flagged (previously slipped through)."""
    (synthetic_root / "train" / "labels" / "img0.txt").write_text("-1 0.5 0.5 0.2 0.2\n")
    df = vl.validate(synthetic_root, num_classes=3)
    assert "invalid_class_id" in _types(df)


def test_box_out_of_range_detected(synthetic_root: Path):
    (synthetic_root / "train" / "labels" / "img0.txt").write_text("0 1.5 0.5 0.2 0.2\n")
    df = vl.validate(synthetic_root, num_classes=3)
    assert "box_out_of_range" in _types(df)


def test_zero_size_box_detected(synthetic_root: Path):
    (synthetic_root / "train" / "labels" / "img0.txt").write_text("0 0.5 0.5 0.0 0.2\n")
    df = vl.validate(synthetic_root, num_classes=3)
    assert "zero_size_box" in _types(df)


def test_corrupt_image_detected(synthetic_root: Path):
    # overwrite an image with garbage bytes so cv2.imread returns None
    bad = synthetic_root / "train" / "images" / "img0.jpg"
    bad.write_bytes(b"not a real jpeg")
    df = vl.validate(synthetic_root, num_classes=3)
    assert "corrupt_image" in _types(df)


def test_report_columns_stable(synthetic_root: Path):
    (synthetic_root / "train" / "labels" / "img0.txt").unlink()
    df = vl.validate(synthetic_root, num_classes=3)
    assert list(df.columns) == ["split", "image", "issue_type", "detail"]
