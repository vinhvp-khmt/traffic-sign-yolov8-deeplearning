"""Run the full train→eval→compare pipeline locally (Apple Silicon / MPS).

Usage:
    python -m scripts.run_local_pipeline --stage yolo     # train+eval YOLO
    python -m scripts.run_local_pipeline --stage detr     # convert+train+eval DETR
    python -m scripts.run_local_pipeline --stage compare  # assemble comparison.csv

Writes an absolute-path copy of configs/data.yaml so Ultralytics resolves the dataset
regardless of its datasets_dir setting. Defaults are tuned for a quick-but-real baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.utils.paths import (
    DATA_YAML, DATA_PROCESSED, RESULTS_METRICS, WEIGHTS_YOLO, WEIGHTS_DETR,
    DATA_COCO, ensure_dir,
)


def _abs_data_yaml() -> Path:
    """Write a sibling data.yaml with an absolute `path` so Ultralytics always finds it."""
    cfg = yaml.safe_load(DATA_YAML.read_text())
    cfg["path"] = str(DATA_PROCESSED.resolve())
    out = DATA_YAML.parent / "data.local.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def run_yolo(epochs: int, batch: int, imgsz: int, device: str, resume_from: str | None = None):
    from src.evaluation.evaluate_yolo import evaluate

    data = _abs_data_yaml()

    if resume_from:
        from ultralytics import YOLO
        from pathlib import Path as _P
        import shutil
        print(f"[pipeline] RESUMING YOLO from {resume_from}")
        net = YOLO(resume_from)
        results = net.train(resume=True)
        run_dir = _P(results.save_dir) if hasattr(results, "save_dir") else net.trainer.save_dir
        src_best = _P(run_dir) / "weights" / "best.pt"
        dst_dir = ensure_dir(WEIGHTS_YOLO / "yolo_baseline")
        best = dst_dir / "best.pt"
        if src_best.exists():
            shutil.copy2(src_best, best)
            curves = _P(run_dir) / "results.png"
            if curves.exists():
                shutil.copy2(curves, dst_dir / "training_curves.png")
    else:
        from src.training.train_yolo import train
        print(f"[pipeline] YOLO train: epochs={epochs} batch={batch} imgsz={imgsz} device={device}")
        results, best = train(model="yolov8n.pt", data=data, epochs=epochs, imgsz=imgsz,
                              batch=batch, name="yolo_baseline", device=device)
    print(f"[pipeline] best checkpoint: {best}")

    print("[pipeline] YOLO evaluate on test split")
    metrics = evaluate(best, data=data, split="test", imgsz=imgsz, device=device)
    ensure_dir(RESULTS_METRICS)
    (RESULTS_METRICS / "yolo_baseline.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def run_detr(epochs: int, batch: int, device: str, limit_eval: int | None):
    from src.data.convert_to_coco import convert
    from src.training.train_detr import train
    from src.evaluation.evaluate_detr import evaluate

    print("[pipeline] convert YOLO → COCO")
    convert(root=DATA_PROCESSED, out_dir=DATA_COCO)

    print(f"[pipeline] DETR train: epochs={epochs} batch={batch} device={device}")
    model_dir = train(epochs=epochs, batch=batch, name="detr_baseline")

    print("[pipeline] DETR evaluate on test split")
    metrics = evaluate(model_dir, split="test", device=device)
    ensure_dir(RESULTS_METRICS)
    (RESULTS_METRICS / "detr_baseline.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def run_compare():
    from src.evaluation.compare_models import compare
    from src.utils.paths import RESULTS_TABLES
    df = compare()
    if not df.empty:
        out = ensure_dir(RESULTS_TABLES) / "comparison.csv"
        df.to_csv(out, index=False)
        print(f"[pipeline] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["yolo", "detr", "compare"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit-eval", type=int, default=None)
    ap.add_argument("--resume-from", default=None, help="path to last.pt to resume YOLO")
    args = ap.parse_args()

    if args.stage == "yolo":
        run_yolo(args.epochs or 30, args.batch or 16, args.imgsz, args.device, args.resume_from)
    elif args.stage == "detr":
        run_detr(args.epochs or 10, args.batch or 4, args.device, args.limit_eval)
    else:
        run_compare()


if __name__ == "__main__":
    main()
