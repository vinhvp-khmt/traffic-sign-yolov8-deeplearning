import time
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image

device = "mps"
print(f"Testing on device: {device}")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=15,
    ignore_mismatched_sizes=True,
).to(device)

img = Image.new('RGB', (640, 640), color='red')
annotations = {
    "image_id": 1,
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 3,
            "bbox": [100, 100, 50, 50],
            "area": 2500,
            "iscrowd": 0
        }
    ]
}

inputs = processor(images=img, annotations=annotations, return_tensors="pt").to(device)
# unpack target labels
labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in inputs['labels']]

# Warmup
t0 = time.time()
outputs = model(pixel_values=inputs['pixel_values'], pixel_mask=inputs['pixel_mask'], labels=labels)
loss = outputs.loss
loss.backward()
print(f"First step took: {time.time() - t0:.2f} seconds")

# Benchmark 5 steps
times = []
for i in range(5):
    t0 = time.time()
    outputs = model(pixel_values=inputs['pixel_values'], pixel_mask=inputs['pixel_mask'], labels=labels)
    loss = outputs.loss
    loss.backward()
    times.append(time.time() - t0)

print(f"Average step time over 5 steps: {sum(times)/len(times):.2f} seconds")
