"""Inference speed benchmarking (plan §8.5/§8.6): latency + FPS.

Times forward passes on dummy input after a warm-up. Used by evaluate_yolo/evaluate_detr and
runnable standalone. Speed is hardware-dependent — always report the device.

CLI:
    python -m src.evaluation.benchmark_fps --yolo weights/yolo/yolo_baseline/best.pt
    python -m src.evaluation.benchmark_fps --detr weights/detr/detr_baseline
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def _timed_runs(fn, runs: int) -> float:
    """Return mean seconds/run over `runs` calls (caller does warm-up)."""
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) / runs


def benchmark_yolo(weights: Path, imgsz: int = 640, device: str | None = None,
                   warmup: int = 5, runs: int = 30) -> dict:
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(weights))
    dummy = (np.random.rand(imgsz, imgsz, 3) * 255).astype("uint8")
    call = lambda: model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    for _ in range(warmup):
        call()
    spr = _timed_runs(call, runs)
    return {"device": str(device or "auto"), "latency_ms": round(spr * 1000, 2),
            "fps": round(1.0 / spr, 1)}


def benchmark_detr(model_dir: Path, device: str | None = None,
                   warmup: int = 3, runs: int = 20) -> dict:
    import torch
    from transformers import DetrForObjectDetection, DetrImageProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = DetrImageProcessor.from_pretrained(model_dir)
    model = DetrForObjectDetection.from_pretrained(model_dir).to(device).eval()
    size = processor.size.get("shortest_edge", 800) if isinstance(processor.size, dict) else 800
    dummy = torch.rand(1, 3, size, size).to(device)

    def call():
        with torch.no_grad():
            model(pixel_values=dummy)

    for _ in range(warmup):
        call()
    if device == "cuda":
        torch.cuda.synchronize()
    spr = _timed_runs(call, runs)
    return {"device": device, "latency_ms": round(spr * 1000, 2), "fps": round(1.0 / spr, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark inference speed (plan §8.5/§8.6).")
    ap.add_argument("--yolo", type=Path, help="YOLO .pt checkpoint")
    ap.add_argument("--detr", type=Path, help="DETR model dir")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.yolo:
        print("YOLO:", benchmark_yolo(args.yolo, device=args.device))
    if args.detr:
        print("DETR:", benchmark_detr(args.detr, device=args.device))
    if not args.yolo and not args.detr:
        ap.error("provide --yolo and/or --detr")


if __name__ == "__main__":
    main()
