import os
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.training.train_detr import CocoDetrDataset, collate_fn
from src.utils.paths import DATA_COCO

device = "mps"
print(f"Loading processor and model on {device}...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=15,
    ignore_mismatched_sizes=True,
).to(device)

print("Loading dataset...")
train_ds = CocoDetrDataset(DATA_COCO / "instances_train.json", processor)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)

print("Starting loop...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for idx, batch_data in enumerate(train_loader):
    print(f"Batch {idx} loaded")
    pixel_values = batch_data["pixel_values"].to(device)
    pixel_mask   = batch_data["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in lbl.items()}
              for lbl in batch_data["labels"]]
              
    print("Running forward...")
    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
    loss = outputs.loss
    
    print("Running backward...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Batch {idx} complete. Loss: {loss.item()}")
    if idx >= 4:
        break

print("All done!")
