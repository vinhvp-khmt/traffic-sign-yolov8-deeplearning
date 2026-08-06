"""Bounding-box statistics (plan §8.4) — OWNER: Vinh."""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.data.inspect_dataset import iter_pairs, parse_label_file
from src.utils.paths import DATA_PROCESSED, RESULTS_EDA, RESULTS_TABLES, SPLITS
from src.utils.plotting import save_fig

def collect_boxes(root=DATA_PROCESSED) -> pd.DataFrame:
    """Collect all bounding boxes from the dataset.
    
    Returns a DataFrame with columns:
      split | image | cls | xc | yc | w | h | area | aspect
    """
    rows = []
    for split in SPLITS:
        for img_path, label_path in iter_pairs(split, root):
            boxes = parse_label_file(label_path)
            for cls, xc, yc, w, h in boxes:
                area = w * h
                aspect = w / h if h > 0 else 0
                rows.append({
                    "split": split,
                    "image": img_path.name,
                    "cls": cls,
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h,
                    "area": area,
                    "aspect": aspect
                })
    return pd.DataFrame(rows)

def run() -> None:
    df = collect_boxes()
    if df.empty:
        print("[bbox_statistics] No boxes found.")
        return

    # Plot Area
    fig, ax = plt.subplots()
    ax.hist(df["area"], bins=50, color="skyblue", edgecolor="black")
    ax.set_title("Bounding Box Area Distribution (Normalized)")
    ax.set_xlabel("Area (w * h)")
    ax.set_ylabel("Count")
    save_fig(fig, RESULTS_EDA / "bbox_area.png")

    # Plot W vs H
    fig, ax = plt.subplots()
    ax.scatter(df["w"], df["h"], alpha=0.3, s=2)
    ax.set_title("Bounding Box Width vs Height")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    save_fig(fig, RESULTS_EDA / "bbox_wh.png")

    # Plot Aspect Ratio
    fig, ax = plt.subplots()
    ax.hist(df["aspect"], bins=50, range=(0, 5), color="lightgreen", edgecolor="black")
    ax.set_title("Bounding Box Aspect Ratio Distribution (w/h)")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Count")
    save_fig(fig, RESULTS_EDA / "bbox_aspect_ratio.png")

    # Categorize Sizes
    def categorize_size(area):
        if area < 0.01:
            return "Small"
        elif area < 0.05:
            return "Medium"
        else:
            return "Large"
            
    df["size_cat"] = df["area"].apply(categorize_size)
    size_counts = df["size_cat"].value_counts().reindex(["Small", "Medium", "Large"], fill_value=0)
    
    # Plot Size Categories
    fig, ax = plt.subplots()
    size_counts.plot(kind="bar", color=["tomato", "gold", "limegreen"], edgecolor="black", ax=ax)
    ax.set_title("Bounding Box Size Categories")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    save_fig(fig, RESULTS_EDA / "bbox_size_categories.png")

    # Save CSV
    size_df = size_counts.reset_index()
    size_df.columns = ["size_cat", "count"]
    size_df["pct"] = (size_df["count"] / size_df["count"].sum() * 100).round(2)
    
    out_csv = RESULTS_TABLES / "bbox_size_categories.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    size_df.to_csv(out_csv, index=False)
    
    print(f"[bbox_statistics] Generated plots in {RESULTS_EDA}")
    print(f"[bbox_statistics] Saved size categories to {out_csv}")

if __name__ == "__main__":
    run()
