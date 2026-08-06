"""Tests for YOLO-TXT → COCO JSON conversion, focusing on the coordinate math."""
from __future__ import annotations

import json
from pathlib import Path

from src.data import convert_to_coco as c2c


def test_convert_writes_all_splits(synthetic_root: Path, tmp_path: Path, monkeypatch):
    # point load_classes at the synthetic 3-class set
    monkeypatch.setattr(c2c, "load_classes", lambda: {0: "Green Light", 1: "Red Light", 2: "Stop"})
    out = tmp_path / "coco"
    c2c.convert(root=synthetic_root, out_dir=out)
    for split in ("train", "valid", "test"):
        assert (out / f"instances_{split}.json").exists()


def test_coco_bbox_math_is_correct(synthetic_root: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(c2c, "load_classes", lambda: {0: "Green Light", 1: "Red Light", 2: "Stop"})
    out = tmp_path / "coco"
    c2c.convert(root=synthetic_root, out_dir=out)

    data = json.loads((out / "instances_train.json").read_text())
    # images are 32x32 in the fixture
    img_w = data["images"][0]["width"]
    img_h = data["images"][0]["height"]
    assert (img_w, img_h) == (32, 32)

    # img0 has one box: cls0 at (0.5,0.5) size (0.2,0.2)
    # → top-left = ((0.5-0.1)*32, (0.5-0.1)*32) = (12.8, 12.8); w=h=6.4
    img0 = next(i for i in data["images"] if i["file_name"] == "img0.jpg")
    ann = next(a for a in data["annotations"] if a["image_id"] == img0["id"])
    assert ann["bbox"] == [12.8, 12.8, 6.4, 6.4]
    assert ann["area"] == round(6.4 * 6.4, 2)
    assert ann["category_id"] == 0
    assert ann["iscrowd"] == 0


def test_coco_annotation_ids_unique_and_sequential(synthetic_root: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(c2c, "load_classes", lambda: {0: "Green Light", 1: "Red Light", 2: "Stop"})
    out = tmp_path / "coco"
    c2c.convert(root=synthetic_root, out_dir=out)
    data = json.loads((out / "instances_train.json").read_text())
    ids = [a["id"] for a in data["annotations"]]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_coco_categories_match_classes(synthetic_root: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(c2c, "load_classes", lambda: {0: "Green Light", 1: "Red Light", 2: "Stop"})
    out = tmp_path / "coco"
    c2c.convert(root=synthetic_root, out_dir=out)
    data = json.loads((out / "instances_train.json").read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    assert cats == {0: "Green Light", 1: "Red Light", 2: "Stop"}
