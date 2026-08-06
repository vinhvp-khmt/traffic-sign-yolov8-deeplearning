# Traffic Sign Detection for Self-Driving Cars: YOLO vs DETR

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huynhphtloi/traffic-sign-detection-yolo-detr/blob/main/notebooks/traffic_sign_detection_pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible study comparing **YOLO** (one-stage CNN) and **DETR** (transformer) for
**traffic-sign object detection** in self-driving-car scenarios. The goal is a detector that is
both **robust** and **data-efficient** — and a clean, runnable pipeline anyone can reproduce.

> **Research question:** How can traffic-sign detection for self-driving cars be made more
> **robust** and **data-efficient** using YOLO and DETR?

## Dataset

[pkdarabi/cardetection](https://www.kaggle.com/datasets/pkdarabi/cardetection/data) — *Traffic
Signs Detection*. Despite the URL name, the task is **traffic-sign** detection, not car detection.
**15 native classes** (Green/Red Light, Speed Limit 10–120, Stop), used as-is with no remapping.

## Quickstart

The whole pipeline lives in **one notebook** so it reproduces top-to-bottom with a single click.

**On Colab (recommended — GPU for training):**
open [`notebooks/traffic_sign_detection_pipeline.ipynb`](notebooks/traffic_sign_detection_pipeline.ipynb)
([Colab badge above](https://colab.research.google.com/github/huynhphtloi/traffic-sign-detection-yolo-detr/blob/main/notebooks/traffic_sign_detection_pipeline.ipynb))
and run the cells in order. Section 0 mounts Drive, installs deps, and downloads the dataset (you
provide a `kaggle.json`); later sections inspect the data, run EDA, and train/evaluate the baselines.

**Locally (CPU — inspection/EDA only, no training):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the Roboflow export at dataset/processed/cardetection/{train,valid,test}/{images,labels}
python -m src.data.inspect_dataset --write-configs
```

Every step is also a standalone module — `python -m src.<pkg>.<module>` — so you can run any
part of the pipeline outside the notebook.

## Tests

A `pytest` suite covers the pure-Python logic (parsing, COCO conversion math, EDA collectors,
data-quality checks, path helpers, RNG seeding) so it runs in seconds on CPU with no GPU or
model downloads. It uses a tiny synthetic dataset built on the fly, plus smoke tests that run
against the real raw dataset when it's present and skip cleanly when it isn't.

```bash
pip install -r requirements.txt
pytest                 # ~40 tests, < 1s
```

Reproducibility: all RNGs (Python, NumPy, PyTorch, and the Ultralytics `seed`) are seeded via
`src.utils.seeding.seed_everything` (default seed 42), and `requirements.txt` is version-pinned.

## What's inside

```
notebooks/   traffic_sign_detection_pipeline.ipynb   — the full, single-file pipeline
src/data/    inspect_dataset · validate_labels · visualize_annotations · convert_to_coco
src/eda/     class_distribution · bbox_statistics · image_statistics · heatmap_analysis
src/training/   train_yolo · train_detr
src/evaluation/ evaluate_yolo · evaluate_detr · benchmark_fps · compare_models
src/utils/   paths (central path registry) · plotting · seeding (reproducible RNGs)
configs/     data.yaml (Ultralytics) + classes.yaml (15 classes, auto-synced from the dataset)
tests/       pytest suite (synthetic fixtures + real-dataset smoke tests)
results/     eda/ samples/ tables/ metrics/ plots/   (generated; gitignored)
```

The notebook is **thin orchestration** over `src/` — all logic lives in the modules.

## Status & roadmap

- **Phase 1 — baselines (in progress):** dataset inspection, EDA, data-quality checks, YOLO and
  DETR baselines, and an initial comparison.
- **Phase 2 — planned:** model improvement, robustness + data-efficiency experiments, error
  analysis, a **Gradio** demo app, and **Hugging Face Hub + Spaces** deployment.

In the notebook, sections marked **⏳ WIP** call modules that are still being implemented; they run
as soon as those modules land.

## Data & weights policy

`dataset/`, `weights/`, and heavy `results/` outputs are **gitignored** — only code, the notebook,
configs, and small generated tables are tracked. Bring your own dataset via Kaggle (Section 0).

## Contributors

Loi · Vinh · Tu.

## License

Released under the [MIT License](LICENSE).
