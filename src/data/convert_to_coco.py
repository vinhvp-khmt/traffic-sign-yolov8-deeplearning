"""Convert the YOLO-TXT dataset to COCO JSON for DETR (plan §8.6) — OWNER: Tu.

Output: dataset/coco/instances_{train,valid,test}.json
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.data.inspect_dataset import label_path_for, list_images, parse_label_file
from src.utils.paths import DATA_COCO, DATA_PROCESSED, SPLITS, ensure_dir, load_classes


def convert(root: Path = DATA_PROCESSED, out_dir: Path = DATA_COCO) -> None:
    ensure_dir(out_dir)
    classes = load_classes()
    categories = [
        {"id": cid, "name": name, "supercategory": "traffic_sign"}
        for cid, name in sorted(classes.items())
    ]

    for split in SPLITS:
        images_list: list[dict] = []
        annotations: list[dict] = []
        ann_id = 1

        all_imgs = list_images(split, root)
        n_imgs = len(all_imgs)
        print(f"[convert_to_coco] ▶ {split}: converting {n_imgs} images...", flush=True)
        for img_id, img_path in enumerate(all_imgs, start=1):
            if n_imgs and img_id % 500 == 0:
                print(f"[convert_to_coco]   {split}: {img_id}/{n_imgs} images done", flush=True)
            with Image.open(img_path) as im:
                w_img, h_img = im.size

            images_list.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": w_img,
                "height": h_img,
            })

            for cls_id, xc, yc, bw, bh in parse_label_file(label_path_for(img_path, root)):
                # YOLO (normalized xc,yc,w,h) → COCO (absolute x_top_left, y_top_left, w, h)
                x = (xc - bw / 2) * w_img
                y = (yc - bh / 2) * h_img
                abs_w = bw * w_img
                abs_h = bh * h_img
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(x, 2), round(y, 2), round(abs_w, 2), round(abs_h, 2)],
                    "area": round(abs_w * abs_h, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

        coco = {
            "info": {"description": "Traffic Sign Detection — COCO export", "version": "1.0"},
            "licenses": [],
            "categories": categories,
            "images": images_list,
            "annotations": annotations,
        }
        out_file = out_dir / f"instances_{split}.json"
        out_file.write_text(json.dumps(coco, indent=2))
        print(f"[convert_to_coco] {split}: {len(images_list)} images, "
              f"{len(annotations)} annotations → {out_file}")


if __name__ == "__main__":
    convert()
