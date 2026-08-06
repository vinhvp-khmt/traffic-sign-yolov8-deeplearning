"""Inspect the raw dataset and report what it actually is (plan §8.1).

This module deliberately *detects* the dataset layout and annotation format rather than
assuming it. It also exposes small iteration helpers (list_images, parse_label_file,
iter_pairs) that the other data/EDA modules reuse, so YOLO-label parsing lives in one place.

CLI:
    python -m src.data.inspect_dataset                 # inspect data/processed/cardetection
    python -m src.data.inspect_dataset --root data/raw/cardetection
    python -m src.data.inspect_dataset --write-configs # also sync configs/*.yaml from data.yaml

Output:
    results/tables/dataset_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.utils.paths import (
    CLASSES_YAML,
    DATA_PROCESSED,
    DATA_YAML,
    IMAGE_EXTS,
    RESULTS_TABLES,
    SPLITS,
    ensure_dir,
    images_dir,
    labels_dir,
    split_dir,
)


# ── Iteration helpers (reused by validate_labels, visualize_annotations, eda/*) ──────────
def list_images(split: str, root: Path = DATA_PROCESSED) -> list[Path]:
    """Sorted list of image files for a split."""
    d = images_dir(split, root)
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def label_path_for(image_path: Path, root: Path = DATA_PROCESSED) -> Path:
    """Map an image path to its YOLO .txt label path (…/labels/<stem>.txt)."""
    split = image_path.parent.parent.name  # …/<split>/images/<file>
    return labels_dir(split, root) / f"{image_path.stem}.txt"


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file → list of (class_id, xc, yc, w, h). Missing file → []."""
    if not path.exists():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(v) for v in parts[1:5])
        rows.append((cls, xc, yc, w, h))
    return rows


def iter_pairs(split: str, root: Path = DATA_PROCESSED):
    """Yield (image_path, label_path) pairs for a split."""
    for img in list_images(split, root):
        yield img, label_path_for(img, root)


def find_data_yaml(root: Path) -> Path | None:
    """Locate a Roboflow/Ultralytics data.yaml at/under `root`."""
    direct = root / "data.yaml"
    if direct.exists():
        return direct
    hits = list(root.rglob("data.yaml"))
    return hits[0] if hits else None


def detect_format(root: Path) -> str:
    """Best-effort annotation-format detection. Reports, never assumes."""
    if any(root.rglob("*.txt")) and (find_data_yaml(root) or any(root.rglob("labels"))):
        return "YOLO TXT (Roboflow/Ultralytics)"
    if any(root.rglob("*.json")):
        return "COCO JSON (suspected)"
    if any(root.rglob("*.xml")):
        return "PASCAL VOC XML (suspected)"
    if any(root.rglob("*.csv")):
        return "CSV (suspected)"
    return "UNKNOWN — inspect manually"


def inspect(root: Path = DATA_PROCESSED) -> pd.DataFrame:
    """Build the per-split summary table for `root`."""
    fmt = detect_format(root)
    dyaml = find_data_yaml(root)
    class_names: dict[int, str] | None = None
    if dyaml:
        cfg = yaml.safe_load(dyaml.read_text())
        names = cfg.get("names")
        if isinstance(names, dict):
            class_names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            class_names = {i: str(n) for i, n in enumerate(names)}

    rows = []
    for split in SPLITS:
        imgs = list_images(split, root)
        print(f"[inspect] {split}: {len(imgs)} images — counting labels...", flush=True)
        n_labels, n_boxes = 0, 0
        for i, (_, lp) in enumerate(iter_pairs(split, root), 1):
            if lp.exists():
                n_labels += 1
                n_boxes += len(parse_label_file(lp))
            if i % 500 == 0:
                print(f"[inspect]   {split}: {i}/{len(imgs)} done", flush=True)
        print(f"[inspect] {split}: {n_labels} label files, {n_boxes} boxes", flush=True)
        rows.append(
            {
                "split": split,
                "exists": split_dir(split, root).exists(),
                "images": len(imgs),
                "label_files": n_labels,
                "boxes": n_boxes,
                "format": fmt,
                "num_classes": len(class_names) if class_names else None,
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["class_names"] = class_names
    df.attrs["data_yaml"] = str(dyaml) if dyaml else None
    return df


def write_configs(root: Path) -> None:
    """Regenerate configs/classes.yaml + sync data.yaml names from the dataset's data.yaml."""
    dyaml = find_data_yaml(root)
    if not dyaml:
        print("[write-configs] no data.yaml found; skipping config sync")
        return
    cfg = yaml.safe_load(dyaml.read_text())
    names = cfg.get("names")
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    payload = {"nc": len(names), "names": {int(k): str(v) for k, v in names.items()}}
    CLASSES_YAML.write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=True))
    # keep data.yaml names in sync
    data_cfg = yaml.safe_load(DATA_YAML.read_text())
    data_cfg["nc"] = payload["nc"]
    data_cfg["names"] = payload["names"]
    DATA_YAML.write_text(yaml.safe_dump(data_cfg, sort_keys=False, allow_unicode=True))
    print(f"[write-configs] synced {CLASSES_YAML} and {DATA_YAML} ({payload['nc']} classes)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the traffic-sign dataset (plan §8.1).")
    ap.add_argument("--root", type=Path, default=DATA_PROCESSED,
                    help="dataset root (default: data/processed/cardetection)")
    ap.add_argument("--write-configs", action="store_true",
                    help="regenerate configs/classes.yaml + data.yaml names from data.yaml")
    args = ap.parse_args()

    df = inspect(args.root)
    out = ensure_dir(RESULTS_TABLES) / "dataset_summary.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nformat: {df['format'].iloc[0]}")
    print(f"data.yaml: {df.attrs.get('data_yaml')}")
    print(f"class_names: {df.attrs.get('class_names')}")
    print(f"\nwrote {out}")

    if args.write_configs:
        write_configs(args.root)


if __name__ == "__main__":
    main()
