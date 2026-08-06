import os
from transformers import DetrImageProcessor
from PIL import Image

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
print("Processor loaded successfully!")

# Let's mock a simple image and annotations
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

try:
    outputs = processor(images=img, annotations=annotations, return_tensors="pt")
    print("Preprocess outputs keys:", outputs.keys())
    print("Labels structure:", outputs['labels'])
except Exception as e:
    import traceback
    traceback.print_exc()
