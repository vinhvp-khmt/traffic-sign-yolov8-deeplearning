"""Tests for the EDA collectors and the class-imbalance logic (regression guard)."""
from __future__ import annotations

from pathlib import Path

from src.eda import bbox_statistics as bbox
from src.eda import class_distribution as cdist


def test_collect_boxes_shape_and_derived_columns(synthetic_root: Path):
    df = bbox.collect_boxes(synthetic_root)
    assert len(df) == 5  # 3 (train) + 1 (valid) + 1 (test)
    assert set(df.columns) >= {"split", "image", "cls", "xc", "yc", "w", "h", "area", "aspect"}

    row = df[(df["split"] == "train") & (df["cls"] == 2)].iloc[0]
    assert row["area"] == 0.3 * 0.4
    assert abs(row["aspect"] - (0.3 / 0.4)) < 1e-9


def test_collect_boxes_empty(tmp_path: Path):
    df = bbox.collect_boxes(tmp_path)
    assert df.empty


def test_imbalance_ratio_includes_zero_count_classes(monkeypatch, tmp_path, synthetic_root):
    """Regression: a class present in classes.yaml but absent from the data must NOT be
    silently dropped — otherwise the imbalance ratio is understated.

    The synthetic data has classes {0,1,2} but we declare a 4th class (id 3) with zero
    boxes. The correct imbalance ratio is therefore infinite, and the table must list 4 rows.
    """
    import pandas as pd

    # 4-class universe; class 3 never appears in the data
    monkeypatch.setattr(cdist, "load_classes", lambda *a, **k: {0: "A", 1: "B", 2: "C", 3: "D"})
    monkeypatch.setattr(cdist, "collect_boxes", lambda *a, **k: bbox.collect_boxes(synthetic_root))

    captured = {}
    real_to_csv = pd.DataFrame.to_csv

    def spy_to_csv(self, *args, **kwargs):
        captured["df"] = self.copy()
        return real_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", spy_to_csv)
    monkeypatch.setattr(cdist, "save_fig", lambda *a, **k: None)

    cdist.run()

    df = captured["df"]
    assert len(df) == 4, "zero-count class must appear in the distribution table"
    zero_row = df[df["class_id"] == 3].iloc[0]
    assert zero_row["count"] == 0
