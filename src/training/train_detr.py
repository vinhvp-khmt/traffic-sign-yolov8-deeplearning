"""Lightweight DETR baseline training (plan §8.6) — OWNER: Tu.

Fine-tunes facebook/detr-resnet-50 on the COCO export.
Needs a GPU (Colab).
Output: weights/detr/detr_baseline/
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.utils.paths import DATA_COCO, DATA_PROCESSED, WEIGHTS_DETR, ensure_dir


class CocoDetrDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json: Path, processor: DetrImageProcessor):
        self.processor = processor

        with open(coco_json) as f:
            data = json.load(f)

        self.images = {img["id"]: img for img in data["images"]}
        self.img_ids = [img["id"] for img in data["images"]]

        self.anns: dict[int, list] = {img_id: [] for img_id in self.img_ids}
        for ann in data["annotations"]:
            self.anns[ann["image_id"]].append(ann)

        split = coco_json.stem.replace("instances_", "")
        self.img_dir = DATA_PROCESSED / split / "images"

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        from PIL import Image
        img_id = self.img_ids[idx]
        meta = self.images[img_id]
        img_path = self.img_dir / meta["file_name"]
        image = Image.open(img_path).convert("RGB")

        anns = self.anns[img_id]
        target = {
            "image_id": img_id,
            "annotations": self.anns[img_id]
        }
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "pixel_mask":   encoding["pixel_mask"].squeeze(0),
            "labels":       encoding["labels"][0],  # list of 1 dict → unwrap
        }


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask   = torch.stack([b["pixel_mask"]   for b in batch])
    labels = [b["labels"] for b in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}


def train(
    epochs: int = 10,
    batch: int = 2,
    lr: float = 1e-5,
    lr_backbone: float = 1e-6,
    name: str = "detr_baseline",
    coco_dir: Path = DATA_COCO,
    out_root: Path = WEIGHTS_DETR,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[train_detr] device={device}, epochs={epochs}, batch={batch}")

    from src.utils.seeding import seed_everything
    seed_everything()  # reproducible shuffling / initialisation

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=15,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_ds = CocoDetrDataset(coco_dir / "instances_train.json", processor)
    val_ds   = CocoDetrDataset(coco_dir / "instances_valid.json", processor)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # Lower LR on the pretrained ResNet-50 backbone than on the transformer head.
    # Fine-tuning the whole DETR at a single 1e-4 LR makes the loss diverge to NaN
    # (the official HF DETR recipe uses lr=1e-4 head / 1e-5 backbone; we go an order
    # lower for extra stability on this small, imbalanced dataset).
    backbone_params, head_params = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if "backbone" in pname else head_params).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr},
            {"params": backbone_params, "lr": lr_backbone},
        ],
        weight_decay=1e-4,
    )

    import time
    n_train, n_val = len(train_loader), len(val_loader)
    print(f"[train_detr] ▶ {n_train} train batches, {n_val} val batches/epoch — "
          f"lr(head)={lr} lr(backbone)={lr_backbone}; watch per-epoch val_loss below", flush=True)

    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        # Train
        model.train()
        total_loss = 0.0
        for step, batch_data in enumerate(train_loader):
            pixel_values = batch_data["pixel_values"].to(device)
            pixel_mask   = batch_data["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in lbl.items()}
                      for lbl in batch_data["labels"]]
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss

            # Guard against divergence: if a step produces NaN/Inf, skip it instead of
            # letting it poison every subsequent weight (which is what produced the
            # `nan` boxes seen earlier). Persistent NaNs mean the LR is still too high.
            if not torch.isfinite(loss):
                print(f"[train_detr]   ⚠ non-finite loss at epoch {epoch} step {step}; "
                      f"skipping batch", flush=True)
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            total_loss += loss.item()

            if step % 50 == 0:
                elapsed = time.perf_counter() - epoch_start
                rate = (step + 1) / elapsed if elapsed > 0 else 0
                eta = (n_train - step - 1) / rate if rate > 0 else 0
                print(f"[train_detr] epoch {epoch}/{epochs} | batch {step}/{n_train} "
                      f"| loss {loss.item():.4f} | {rate:.1f} it/s | ~{eta/60:.1f} min left in epoch",
                      flush=True)

        avg_train = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val_ok = 0
        with torch.no_grad():
            for batch_data in val_loader:
                pixel_values = batch_data["pixel_values"].to(device)
                pixel_mask   = batch_data["pixel_mask"].to(device)
                labels = [{k: v.to(device) for k, v in lbl.items()}
                          for lbl in batch_data["labels"]]
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                if torch.isfinite(outputs.loss):
                    val_loss += outputs.loss.item()
                    n_val_ok += 1

        avg_val = (val_loss / n_val_ok) if n_val_ok else float("inf")
        epoch_min = (time.perf_counter() - epoch_start) / 60
        print(f"[train_detr] ✔ epoch {epoch}/{epochs}  "
              f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  ({epoch_min:.1f} min)",
              flush=True)

        # Save the BEST checkpoint (lowest val loss), not just the last epoch — so the
        # reported best epoch and the weights on disk actually match.
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_dir = ensure_dir(out_root / name)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"[train_detr]   ↳ new best (val_loss={avg_val:.4f}) → saved to {best_dir}")

    # Fall back to saving the final model if no epoch ever improved (e.g. epochs<1 guard).
    out_dir = ensure_dir(out_root / name)
    if best_epoch == 0:
        model.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)
    print(f"[train_detr] done → {out_dir}  (best val_loss={best_val_loss:.4f} at epoch {best_epoch})")
    return out_dir


if __name__ == "__main__":
    train()
