"""Evaluate a trained DETR checkpoint (plan §8.6) — OWNER: Tu.

Computes mAP@0.5 and mAP@0.5:0.95, merges FPS from benchmark_fps,
and writes results/metrics/detr_baseline.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.evaluation.benchmark_fps import benchmark_detr
from src.utils.paths import DATA_COCO, RESULTS_METRICS, WEIGHTS_DETR, ensure_dir


def evaluate(
    model_dir: Path = WEIGHTS_DETR / "detr_baseline",
    split: str = "test",
    coco_dir: Path = DATA_COCO,
    device: str | None = None,
) -> dict:
    from PIL import Image

    device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[evaluate_detr] loading model from {model_dir} on {device}")

    processor = DetrImageProcessor.from_pretrained(model_dir)
    model = DetrForObjectDetection.from_pretrained(model_dir).to(device).eval()

    coco_json = coco_dir / f"instances_{split}.json"
    with open(coco_json) as f:
        data = json.load(f)

    images_meta = {img["id"]: img for img in data["images"]}
    img_ids = [img["id"] for img in data["images"]]
    anns_by_img: dict[int, list] = {i: [] for i in img_ids}
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # Detect images folder from split name
    from src.utils.paths import DATA_PROCESSED
    img_dir = DATA_PROCESSED / split / "images"

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    print(f"[evaluate_detr] running inference on {len(img_ids)} images...")
    for idx, img_id in enumerate(img_ids):
        if idx % 100 == 0:
            print(f"[evaluate_detr] evaluated {idx}/{len(img_ids)} images...")
        meta = images_meta[img_id]
        img_path = img_dir / meta["file_name"]
        image = Image.open(img_path).convert("RGB")
        w_img, h_img = image.size

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process: convert logits → boxes (xyxy, absolute pixels).
        # threshold=0.0 keeps all 100 DETR queries: mAP is a ranking metric and needs the
        # full score-ordered prediction set, so we deliberately do NOT pre-filter here.
        target_sizes = torch.tensor([[h_img, w_img]], device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes
        )[0]

        preds = [{
            "boxes":  results["boxes"].cpu(),
            "scores": results["scores"].cpu(),
            "labels": results["labels"].cpu(),
        }]

        # Ground truth: COCO bbox [x,y,w,h] → xyxy
        gt_boxes, gt_labels = [], []
        for ann in anns_by_img[img_id]:
            x, y, bw, bh = ann["bbox"]
            gt_boxes.append([x, y, x + bw, y + bh])
            gt_labels.append(ann["category_id"])

        targets = [{
            "boxes":  torch.tensor(gt_boxes,  dtype=torch.float32) if gt_boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(gt_labels, dtype=torch.long),
        }]
        metric.update(preds, targets)

    computed = metric.compute()
    map50    = float(computed["map_50"])
    map50_95 = float(computed["map"])
    print(f"[evaluate_detr] mAP@0.5={map50:.4f}  mAP@0.5:0.95={map50_95:.4f}")

    fps_info = benchmark_detr(model_dir, device=device)

    result = {
        "model":        "DETR",
        "weights":      str(model_dir),
        "split":        split,
        "map50":        round(map50, 4),
        "map50_95":     round(map50_95, 4),
        "model_size_mb": round(sum(p.numel() * p.element_size()
                                   for p in model.parameters()) / 1e6, 2),
        **fps_info,
    }
    return result


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate DETR (plan §8.6).")
    ap.add_argument("--model-dir", type=Path, default=WEIGHTS_DETR / "detr_baseline")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=RESULTS_METRICS / "detr_baseline.json")
    args = ap.parse_args()

    result = evaluate(args.model_dir, args.split, device=args.device)
    ensure_dir(args.out.parent)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
