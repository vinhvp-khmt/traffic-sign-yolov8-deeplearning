"""Tests for the dataset inspection + shared parsing helpers."""
from __future__ import annotations

from pathlib import Path

from src.data import inspect_dataset as ins


def test_list_images_sorted_and_filtered(synthetic_root: Path):
    imgs = ins.list_images("train", synthetic_root)
    assert [p.name for p in imgs] == ["img0.jpg", "img1.jpg"]  # sorted
    assert all(p.suffix == ".jpg" for p in imgs)


def test_list_images_missing_split_returns_empty(synthetic_root: Path):
    assert ins.list_images("nonexistent", synthetic_root) == []


def test_label_path_for_maps_to_labels_dir(synthetic_root: Path):
    img = synthetic_root / "train" / "images" / "img0.jpg"
    lp = ins.label_path_for(img, synthetic_root)
    assert lp == synthetic_root / "train" / "labels" / "img0.txt"
    assert lp.exists()


def test_parse_label_file_valid(synthetic_root: Path):
    lp = synthetic_root / "train" / "labels" / "img1.txt"
    rows = ins.parse_label_file(lp)
    assert rows == [(1, 0.25, 0.25, 0.1, 0.1), (2, 0.75, 0.75, 0.3, 0.4)]
    # class id parsed as int, coords as float
    assert isinstance(rows[0][0], int)
    assert all(isinstance(v, float) for v in rows[0][1:])


def test_parse_label_file_missing_returns_empty(tmp_path: Path):
    assert ins.parse_label_file(tmp_path / "nope.txt") == []


def test_parse_label_file_skips_malformed_lines(tmp_path: Path):
    p = tmp_path / "bad.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\nthis is junk\n1 0.1\n2 0.5 0.5 0.5 0.5\n")
    rows = ins.parse_label_file(p)
    # only the two complete 5-field rows survive
    assert rows == [(0, 0.5, 0.5, 0.2, 0.2), (2, 0.5, 0.5, 0.5, 0.5)]


def test_parse_label_handles_float_class_id(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("3.0 0.5 0.5 0.2 0.2\n")
    assert ins.parse_label_file(p) == [(3, 0.5, 0.5, 0.2, 0.2)]


def test_iter_pairs_yields_every_image(synthetic_root: Path):
    pairs = list(ins.iter_pairs("train", synthetic_root))
    assert len(pairs) == 2
    for img, lbl in pairs:
        assert img.exists() and lbl.exists()


def test_detect_format_yolo(synthetic_root: Path):
    assert "YOLO" in ins.detect_format(synthetic_root)


def test_find_data_yaml(synthetic_root: Path):
    assert ins.find_data_yaml(synthetic_root) == synthetic_root / "data.yaml"


def test_inspect_counts(synthetic_root: Path):
    df = ins.inspect(synthetic_root)
    by_split = df.set_index("split")
    assert by_split.loc["train", "images"] == 2
    assert by_split.loc["train", "boxes"] == 3
    assert by_split.loc["valid", "boxes"] == 1
    assert by_split.loc["test", "boxes"] == 1
    assert by_split.loc["train", "num_classes"] == 3
