"""Class distribution and imbalance (plan §8.4) — OWNER: Vinh."""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.eda.bbox_statistics import collect_boxes
from src.utils.paths import CLASSES_YAML, RESULTS_EDA, RESULTS_TABLES, load_classes
from src.utils.plotting import save_fig

def run() -> None:
    df = collect_boxes()
    if df.empty:
        print("[class_distribution] No boxes found.")
        return

    # Load class names
    class_names = load_classes(CLASSES_YAML) if CLASSES_YAML.exists() else {}

    # Count boxes per class, then reindex against the full class set so that classes
    # with ZERO boxes are not silently dropped (otherwise the imbalance ratio is
    # understated and the "least frequent" class is wrong).
    raw_counts = df["cls"].value_counts()
    if class_names:
        all_ids = sorted(class_names.keys())
    else:
        all_ids = sorted(raw_counts.index.tolist())
    counts = (
        raw_counts.reindex(all_ids, fill_value=0)
        .rename_axis("class_id")
        .reset_index(name="count")
    )

    # Add class name
    counts["class_name"] = counts["class_id"].map(lambda x: class_names.get(x, str(x)))

    # Calculate ratios (guard against divide-by-zero for absent classes)
    max_count = int(counts["count"].max())
    min_count = int(counts["count"].min())
    imbalance_ratio = (max_count / min_count) if min_count > 0 else float("inf")

    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total * 100).round(2) if total else 0.0
    counts["ratio_to_max"] = counts["count"].map(
        lambda c: round(max_count / c, 2) if c > 0 else float("inf")
    )

    # Sort for plotting
    counts = counts.sort_values(by="count", ascending=True) # Ascending for horizontal bar chart

    ratio_label = f"{imbalance_ratio:.2f}x" if min_count > 0 else "∞ (a class has 0 boxes)"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(counts["class_name"], counts["count"], color="cornflowerblue", edgecolor="black")
    ax.set_title(f"Class Distribution (Imbalance Ratio: {ratio_label})")
    ax.set_xlabel("Number of Bounding Boxes")
    ax.set_ylabel("Class")
    save_fig(fig, RESULTS_EDA / "class_distribution.png")

    # Sort descending for CSV
    counts = counts.sort_values(by="count", ascending=False)
    
    out_csv = RESULTS_TABLES / "class_distribution.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_csv, index=False)

    print(f"[class_distribution] Most frequent: {counts.iloc[0]['class_name']} ({counts.iloc[0]['count']})")
    print(f"[class_distribution] Least frequent: {counts.iloc[-1]['class_name']} ({counts.iloc[-1]['count']})")
    print(f"[class_distribution] Imbalance ratio: {ratio_label}")
    print(f"[class_distribution] Generated plot in {RESULTS_EDA}")
    print(f"[class_distribution] Saved table to {out_csv}")

if __name__ == "__main__":
    run()
