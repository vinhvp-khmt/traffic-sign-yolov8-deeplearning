"""Central path registry for the project.

Every other module imports paths from here so that locations are defined in exactly one
place and the code works identically on a laptop and on Colab. Nothing here touches the
filesystem on import except resolving the repo root, so it is safe to import anywhere
(including with no dataset present).

Layout (relative to the repo root):

    dataset/raw/               raw Kaggle download (Roboflow export, gitignored)
    dataset/processed/cardetection/{train,valid,test}/{images,labels}   normalized YOLO layout
    dataset/coco/              COCO JSON annotations for DETR
    weights/{yolo,detr}/       trained checkpoints (gitignored)
    results/{eda,samples,tables,metrics,plots}/   generated outputs
    configs/{data.yaml,classes.yaml}
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# ── Repo root ────────────────────────────────────────────────────────────────
# This file lives at <repo>/src/utils/paths.py → parents[2] is <repo>.
# Allow an override so the same code runs from a Drive-mounted Colab path.
REPO_ROOT = Path(os.environ.get("TSD_REPO_ROOT", Path(__file__).resolve().parents[2]))

# ── Configs ──────────────────────────────────────────────────────────────────
CONFIGS = REPO_ROOT / "configs"
DATA_YAML = CONFIGS / "data.yaml"
CLASSES_YAML = CONFIGS / "classes.yaml"

# ── Data ─────────────────────────────────────────────────────────────────────
# TSD_DATA_DIR lets Colab point data at Drive while the repo lives in /content.
DATA = Path(os.environ["TSD_DATA_DIR"]) if os.environ.get("TSD_DATA_DIR") else REPO_ROOT / "dataset"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed" / "cardetection"
DATA_COCO = DATA / "coco"

# ── Weights ──────────────────────────────────────────────────────────────────
WEIGHTS = REPO_ROOT / "weights"
WEIGHTS_YOLO = WEIGHTS / "yolo"
WEIGHTS_DETR = WEIGHTS / "detr"

# ── Results ──────────────────────────────────────────────────────────────────
RESULTS = REPO_ROOT / "results"
RESULTS_EDA = RESULTS / "eda"
RESULTS_SAMPLES = RESULTS / "samples"
RESULTS_TABLES = RESULTS / "tables"
RESULTS_METRICS = RESULTS / "metrics"
RESULTS_PLOTS = RESULTS / "plots"

# Standard split names for the Roboflow export.
SPLITS = ("train", "valid", "test")

# Common image extensions, lowercased.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if missing; return it. Use before writing outputs."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_classes(classes_yaml: Path = CLASSES_YAML) -> dict[int, str]:
    """Return the {id: name} class map from a classes.yaml (or data.yaml) file.

    Handles `names` written either as a {id: name} mapping or as a plain list.
    """
    with open(classes_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def resolved_data_yaml(data_yaml: Path = DATA_YAML, root: Path = DATA_PROCESSED) -> Path:
    """Return a data.yaml whose `path` is ABSOLUTE so Ultralytics always finds the dataset.

    The tracked configs/data.yaml uses a relative `path` (nice for git), but Ultralytics
    resolves a relative `path` against its own datasets_dir — which breaks on Colab/Drive
    where the data lives outside the repo. This writes a sibling `data.local.yaml` with the
    `path` pinned to the absolute, environment-aware dataset root and returns it.
    """
    cfg = yaml.safe_load(data_yaml.read_text())
    cfg["path"] = str(root.resolve())
    out = data_yaml.parent / "data.local.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return out


def split_dir(split: str, root: Path = DATA_PROCESSED) -> Path:
    """Return the directory for a split, accepting 'val' as an alias for 'valid'."""
    return root / ("valid" if split == "val" else split)


def images_dir(split: str, root: Path = DATA_PROCESSED) -> Path:
    return split_dir(split, root) / "images"


def labels_dir(split: str, root: Path = DATA_PROCESSED) -> Path:
    return split_dir(split, root) / "labels"
