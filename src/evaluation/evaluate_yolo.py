"""Evaluate a trained YOLO checkpoint (plan §8.5).

Runs Ultralytics validation on a split and writes the standard metrics (mAP@0.5,
mAP@0.5:0.95, precision, recall) plus model size to a JSON. FPS/latency come from
benchmark_fps and are merged in here so yolo_baseline.json is the single YOLO metrics file.

CLI:
    python -m src.evaluation.evaluate_yolo --weights weights/yolo/yolo_baseline/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.benchmark_fps import benchmark_yolo
from src.utils.paths import DATA_YAML, RESULTS_METRICS, WEIGHTS_YOLO, ensure_dir


def evaluate(weights: Path, data: Path | None = None, split: str = "test",
             imgsz: int = 640, device: str | None = None, with_fps: bool = True) -> dict:
    from ultralytics import YOLO

    from src.utils.paths import resolved_data_yaml
    data = Path(data) if data is not None else resolved_data_yaml()

    print(f"[evaluate_yolo] ▶ validating {weights} on split='{split}' (device={device or 'auto'})",
          flush=True)
    model = YOLO(str(weights))
    metrics = model.val(data=str(data), split=split, imgsz=imgsz, device=device)
    print("[evaluate_yolo]   accuracy done; benchmarking inference speed...", flush=True)

    result = {
        "model": "YOLO",
        "weights": str(weights),
        "split": split,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "model_size_mb": round(Path(weights).stat().st_size / 1e6, 2),
    }
    if with_fps:
        result.update(benchmark_yolo(weights, imgsz=imgsz, device=device))
    print("[evaluate_yolo] ✔ done", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate YOLO (plan §8.5).")
    ap.add_argument("--weights", type=Path, default=WEIGHTS_YOLO / "yolo_baseline" / "best.pt")
    ap.add_argument("--data", type=Path, default=None,
                    help="data.yaml (default: auto-resolved absolute-path config)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=RESULTS_METRICS / "yolo_baseline.json")
    args = ap.parse_args()

    result = evaluate(args.weights, args.data, args.split, args.imgsz, args.device)
    ensure_dir(args.out.parent)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
