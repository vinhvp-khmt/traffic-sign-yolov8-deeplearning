"""Draw ground-truth boxes on random sample images (plan §8.3).

Required sanity check before training. Picks N random images from a split, draws each YOLO
box (denormalized to pixels) with its class name, and saves the annotated copies.

CLI:
    python -m src.data.visualize_annotations --split train --n 12

Output:
    results/samples/annotated_<split>_samples/
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from src.data.inspect_dataset import label_path_for, list_images, parse_label_file
from src.utils.paths import (
    CLASSES_YAML,
    DATA_PROCESSED,
    RESULTS_SAMPLES,
    ensure_dir,
    load_classes,
)

# Distinct-ish BGR palette cycled by class id.
_PALETTE = [
    (66, 135, 245), (66, 245, 132), (245, 66, 66), (245, 197, 66), (197, 66, 245),
    (66, 245, 245), (245, 66, 167), (140, 245, 66), (66, 110, 245), (245, 140, 66),
]


def draw_boxes(img, boxes, class_names):
    """Draw normalized YOLO boxes onto a BGR image (in place) and return it."""
    h, w = img.shape[:2]
    for cls, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = _PALETTE[cls % len(_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_names.get(cls, str(cls))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def visualize(split: str, n: int, root: Path = DATA_PROCESSED, seed: int = 0) -> Path:
    """Annotate `n` random images from `split`; return the output directory."""
    class_names = load_classes(CLASSES_YAML) if CLASSES_YAML.exists() else {}
    images = list_images(split, root)
    if not images:
        raise FileNotFoundError(f"no images found for split '{split}' under {root}")
    random.Random(seed).shuffle(images)
    out_dir = ensure_dir(RESULTS_SAMPLES / f"annotated_{split}_samples")
    saved = 0
    for i, img_path in enumerate(images[:n], 1):
        print(f"[visualize] {i}/{min(n, len(images))} — {img_path.name}", flush=True)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[visualize]   WARNING: could not read {img_path.name}", flush=True)
            continue
        boxes = parse_label_file(label_path_for(img_path, root))
        draw_boxes(img, boxes, class_names)
        cv2.imwrite(str(out_dir / img_path.name), img)
        saved += 1
    print(f"[visualize] done — wrote {saved} annotated images to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize annotations (plan §8.3).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--root", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    visualize(args.split, args.n, args.root, args.seed)


if __name__ == "__main__":
    main()
