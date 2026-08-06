"""Object-center location heatmap (plan §8.4) — OWNER: Vinh."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.eda.bbox_statistics import collect_boxes
from src.utils.paths import RESULTS_EDA
from src.utils.plotting import save_fig

def run() -> None:
    df = collect_boxes()
    if df.empty:
        print("[heatmap_analysis] No boxes found.")
        return

    # Extract centers
    xc = df["xc"]
    # Invert yc so that origin (0,0) is top-left when plotted, mimicking image coordinates
    yc = 1.0 - df["yc"] 

    # Create 2D histogram
    bins = 64
    H, xedges, yedges = np.histogram2d(xc, yc, bins=bins, range=[[0, 1], [0, 1]])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # H.T because histogram2d returns (x_bins, y_bins) but imshow expects (rows, cols)
    im = ax.imshow(H.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', interpolation='nearest')
    
    # Add thin grid lines to help with reading the heatmap
    ax.grid(color='white', linestyle='-', linewidth=0.2, alpha=0.5)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Frequency')
    
    # Adjust axes to match image coordinate system intuitively
    # We inverted yc above, so y=1 is the top of the image (0 in image coords)
    ax.set_title("Object Center Location Heatmap")
    ax.set_xlabel("x-center (normalized)")
    ax.set_ylabel("y-center (normalized) [0=top, 1=bottom]")
    
    # To make y-axis labels match the image coordinate system (0 at top, 1 at bottom)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['1.0', '0.75', '0.5', '0.25', '0.0'])

    save_fig(fig, RESULTS_EDA / "object_center_heatmap.png")

    print(f"[heatmap_analysis] Generated heatmap in {RESULTS_EDA}")

if __name__ == "__main__":
    run()
