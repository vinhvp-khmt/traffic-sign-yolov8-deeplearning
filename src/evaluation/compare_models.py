"""Initial YOLO vs DETR comparison (plan §8.7) — OWNER: Vinh.

Reads both metrics JSONs and assembles the comparison table.
Output: results/tables/comparison.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.paths import RESULTS_METRICS, RESULTS_TABLES, ensure_dir


def compare(metrics_dir: Path = RESULTS_METRICS) -> pd.DataFrame:
    rows = []
    for fname in ("yolo_baseline.json", "detr_baseline.json"):
        fpath = metrics_dir / fname
        if not fpath.exists():
            print(f"[compare_models] {fname} not found — skipping")
            continue
        m = json.loads(fpath.read_text())
        rows.append({
            "model":        m.get("model", fname.replace("_baseline.json", "").upper()),
            "mAP@0.5":      m.get("map50"),
            "mAP@0.5:0.95": m.get("map50_95"),
            "precision":    m.get("precision"),
            "recall":       m.get("recall"),
            "fps":          m.get("fps"),
            "latency_ms":   m.get("latency_ms"),
            "model_size_mb":m.get("model_size_mb"),
            "device":       m.get("device"),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def main() -> None:
    df = compare()
    if df.empty:
        print("[compare_models] no metrics found — run evaluate_yolo and evaluate_detr first")
        return
    out = ensure_dir(RESULTS_TABLES) / "comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
