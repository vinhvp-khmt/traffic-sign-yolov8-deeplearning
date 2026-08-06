"""YOLO baseline training (plan §8.5).

Trains an Ultralytics YOLOv8 model on the traffic-sign dataset. Defaults match the midterm
baseline: YOLOv8n, pretrained, imgsz 640, 30 epochs. Horizontal flip is OFF — many signs are
directional (left/right turn, arrows) and mirroring would corrupt their meaning.

Needs a GPU; run on Colab. Copies the best checkpoint to weights/yolo/<name>/best.pt.

CLI:
    python -m src.training.train_yolo --model yolov8n.pt --epochs 30 --name yolo_baseline
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils.paths import DATA_YAML, WEIGHTS_YOLO, ensure_dir


def train(
    model: str = "yolov8n.pt",
    data: Path | None = None,
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    name: str = "yolo_baseline",
    device: str | None = None,
):
    """Train YOLO and return (results, best_checkpoint_path)."""
    from ultralytics import YOLO  # imported lazily so the module loads without ultralytics

    from src.utils.paths import resolved_data_yaml
    from src.utils.seeding import DEFAULT_SEED, seed_everything
    seed_everything(DEFAULT_SEED)

    # Resolve to an absolute-path data config so Ultralytics finds the dataset on any host.
    data = Path(data) if data is not None else resolved_data_yaml()

    print(f"[train_yolo] ▶ starting: model={model} epochs={epochs} imgsz={imgsz} "
          f"batch={batch} device={device or 'auto'}", flush=True)
    print(f"[train_yolo]   data config: {data}", flush=True)
    print("[train_yolo]   this is the long step — watch per-epoch mAP in the table below / "
          "runs/detect/<name>/results.csv", flush=True)

    net = YOLO(model)  # pretrained weights
    results = net.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        device=device,
        seed=DEFAULT_SEED,  # reproducible augmentation/initialisation
        fliplr=0.0,   # directional signs — never mirror horizontally
        flipud=0.0,
        verbose=True,
    )
    print("[train_yolo] ✔ training finished — copying best checkpoint", flush=True)

    # Ultralytics writes to runs/detect/<name>/weights/best.pt; copy into our weights/ tree.
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else net.trainer.save_dir
    src_best = Path(run_dir) / "weights" / "best.pt"
    dst_dir = ensure_dir(WEIGHTS_YOLO / name)
    dst_best = dst_dir / "best.pt"
    if src_best.exists():
        shutil.copy2(src_best, dst_best)
        # training curves PNG produced by ultralytics
        curves = Path(run_dir) / "results.png"
        if curves.exists():
            shutil.copy2(curves, dst_dir / "training_curves.png")
        print(f"saved checkpoint -> {dst_best}")
    return results, dst_best


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLO baseline (plan §8.5).")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--data", type=Path, default=None,
                    help="data.yaml (default: auto-resolved absolute-path config)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="yolo_baseline")
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU")
    args = ap.parse_args()
    train(args.model, args.data, args.epochs, args.imgsz, args.batch, args.name, args.device)


if __name__ == "__main__":
    main()
