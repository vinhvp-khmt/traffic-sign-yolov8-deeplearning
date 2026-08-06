"""Tests for path helpers, the annotation drawer, and the comparison assembler."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils import paths as P
from src.data import visualize_annotations as viz
from src.evaluation import compare_models as cmp


# ── paths ─────────────────────────────────────────────────────────────────────
def test_split_dir_val_alias(tmp_path: Path):
    assert P.split_dir("val", tmp_path).name == "valid"
    assert P.split_dir("valid", tmp_path).name == "valid"
    assert P.split_dir("train", tmp_path).name == "train"


def test_load_classes_dict_form(classes_yaml: Path):
    assert P.load_classes(classes_yaml) == {0: "Green Light", 1: "Red Light", 2: "Stop"}


def test_load_classes_list_form(tmp_path: Path):
    p = tmp_path / "list.yaml"
    p.write_text("names: [a, b, c]\n")
    assert P.load_classes(p) == {0: "a", 1: "b", 2: "c"}


def test_ensure_dir_creates(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()
    P.ensure_dir(target)
    assert target.is_dir()


# ── visualize_annotations.draw_boxes ───────────────────────────────────────────
def test_draw_boxes_marks_pixels_and_preserves_shape():
    img = np.zeros((100, 100, 3), dtype="uint8")
    before = img.copy()
    out = viz.draw_boxes(img, [(0, 0.5, 0.5, 0.4, 0.4)], {0: "Stop"})
    assert out.shape == before.shape
    # something was drawn (non-zero pixels now exist)
    assert out.sum() > 0


def test_draw_boxes_unknown_class_falls_back_to_id():
    img = np.zeros((50, 50, 3), dtype="uint8")
    # should not raise even when the class id is absent from the name map
    viz.draw_boxes(img, [(7, 0.5, 0.5, 0.2, 0.2)], {})


# ── compare_models ──────────────────────────────────────────────────────────────
def test_compare_reads_both_metrics(tmp_path: Path):
    (tmp_path / "yolo_baseline.json").write_text(json.dumps({
        "model": "YOLO", "map50": 0.8, "map50_95": 0.5,
        "precision": 0.9, "recall": 0.7, "fps": 120, "latency_ms": 8.3,
        "model_size_mb": 6.0, "device": "cuda",
    }))
    (tmp_path / "detr_baseline.json").write_text(json.dumps({
        "model": "DETR", "map50": 0.7, "map50_95": 0.45,
        "fps": 20, "latency_ms": 50.0, "model_size_mb": 160.0, "device": "cuda",
    }))
    df = cmp.compare(tmp_path)
    assert len(df) == 2
    assert set(df["model"]) == {"YOLO", "DETR"}
    assert "mAP@0.5" in df.columns


def test_compare_missing_files_returns_empty(tmp_path: Path):
    df = cmp.compare(tmp_path)
    assert df.empty
