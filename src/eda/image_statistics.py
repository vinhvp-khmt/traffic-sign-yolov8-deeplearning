"""Image-level statistics (plan §8.4) — OWNER: Vinh."""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.data.inspect_dataset import list_images, label_path_for, parse_label_file
from src.utils.paths import DATA_PROCESSED, RESULTS_EDA, RESULTS_TABLES, SPLITS
from src.utils.plotting import save_fig

def run() -> None:
    rows = []
    
    for split in SPLITS:
        images = list_images(split)
        for img_path in images:
            # Read header only
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                continue # Skip if unreadable, data quality step will catch this
                
            label_path = label_path_for(img_path)
            boxes = parse_label_file(label_path)
            
            rows.append({
                "split": split,
                "image": img_path.name,
                "width": width,
                "height": height,
                "n_objects": len(boxes)
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[image_statistics] No images found.")
        return

    # Plot Images per split
    split_counts = df["split"].value_counts().reindex(list(SPLITS), fill_value=0)
    fig, ax = plt.subplots()
    split_counts.plot(kind="bar", color=["mediumseagreen", "dodgerblue", "orchid"], edgecolor="black", ax=ax)
    ax.set_title("Number of Images per Split")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    save_fig(fig, RESULTS_EDA / "images_per_split.png")

    # Plot Image Resolution
    fig, ax = plt.subplots()
    ax.scatter(df["width"], df["height"], alpha=0.3, s=5, color="purple")
    ax.set_title("Image Resolution Distribution")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    save_fig(fig, RESULTS_EDA / "image_resolution.png")

    # Plot Objects per Image
    fig, ax = plt.subplots()
    ax.hist(df["n_objects"], bins=range(0, df["n_objects"].max() + 2), color="orange", edgecolor="black", align="left")
    ax.set_title("Objects per Image")
    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Frequency")
    save_fig(fig, RESULTS_EDA / "objects_per_image.png")

    # Save CSV
    out_csv = RESULTS_TABLES / "image_statistics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[image_statistics] Total images: {len(df)}")
    print(f"[image_statistics] Average objects per image: {df['n_objects'].mean():.2f}")
    print(f"[image_statistics] Generated plots in {RESULTS_EDA}")
    print(f"[image_statistics] Saved table to {out_csv}")

if __name__ == "__main__":
    run()
