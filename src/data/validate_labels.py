"""Annotation / data-quality validation (plan §8.2) — OWNER: Vinh."""
from __future__ import annotations

import pandas as pd
import cv2
from pathlib import Path

from src.data.inspect_dataset import list_images, label_path_for, parse_label_file, labels_dir
from src.utils.paths import CLASSES_YAML, DATA_PROCESSED, RESULTS_TABLES, SPLITS, load_classes

def validate(root=DATA_PROCESSED, num_classes=None) -> pd.DataFrame:
    if num_classes is None:
        class_names = load_classes(CLASSES_YAML) if CLASSES_YAML.exists() else {}
        num_classes = len(class_names) if class_names else float('inf')
        
    issues = []
    seen_stems = set()
    
    for split in SPLITS:
        images = list_images(split, root)
        image_stems = {img_path.stem for img_path in images}  # O(1) membership, no O(n²) scan

        # Check missing images by looking at label files directly
        ldir = labels_dir(split, root)
        if ldir.exists():
            for label_file in sorted(ldir.glob("*.txt")):
                if label_file.stem not in image_stems:
                    issues.append({
                        "split": split,
                        "image": f"{label_file.stem}.???",  # exact image extension unknown
                        "issue_type": "missing_image",
                        "detail": f"Label file {label_file.name} exists but no corresponding image"
                    })

        for img_path in images:
            # Duplicate check
            if img_path.stem in seen_stems:
                issues.append({
                    "split": split,
                    "image": img_path.name,
                    "issue_type": "duplicate_image_name",
                    "detail": "Image with same name exists in another split"
                })
            seen_stems.add(img_path.stem)
            
            # Corrupt image check
            img = cv2.imread(str(img_path))
            if img is None:
                issues.append({
                    "split": split,
                    "image": img_path.name,
                    "issue_type": "corrupt_image",
                    "detail": "cv2.imread() returned None"
                })
            
            label_path = label_path_for(img_path, root)
            
            # Missing label
            if not label_path.exists():
                issues.append({
                    "split": split,
                    "image": img_path.name,
                    "issue_type": "missing_label",
                    "detail": f"No label file {label_path.name}"
                })
                continue
                
            boxes = parse_label_file(label_path)
            
            # Empty label
            if not boxes:
                issues.append({
                    "split": split,
                    "image": img_path.name,
                    "issue_type": "empty_label",
                    "detail": "Label file exists but contains no valid boxes"
                })
                continue
                
            for i, (cls, xc, yc, w, h) in enumerate(boxes):
                if cls < 0 or cls >= num_classes:
                    issues.append({
                        "split": split,
                        "image": img_path.name,
                        "issue_type": "invalid_class_id",
                        "detail": f"Box {i}: class {cls} outside valid range [0, {num_classes})"
                    })
                
                if xc < 0 or xc > 1 or yc < 0 or yc > 1 or w < 0 or w > 1 or h < 0 or h > 1:
                    issues.append({
                        "split": split,
                        "image": img_path.name,
                        "issue_type": "box_out_of_range",
                        "detail": f"Box {i}: (xc, yc, w, h) = ({xc}, {yc}, {w}, {h}) out of [0, 1]"
                    })
                    
                if w == 0 or h == 0:
                    issues.append({
                        "split": split,
                        "image": img_path.name,
                        "issue_type": "zero_size_box",
                        "detail": f"Box {i}: w or h is 0"
                    })

    df = pd.DataFrame(issues, columns=["split", "image", "issue_type", "detail"])
    
    out_csv = RESULTS_TABLES / "data_quality_report.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    
    return df

def summarize(report_df: pd.DataFrame) -> str:
    if report_df.empty:
        return "Total issues: 0"
        
    counts = report_df["issue_type"].value_counts()
    lines = [f"Total issues: {len(report_df)}"]
    for issue_type, count in counts.items():
        lines.append(f"  {issue_type}: {count}")
    return "\n".join(lines)

if __name__ == "__main__":
    rep = validate()
    print(summarize(rep))
    print(f"[validate_labels] Saved report to {RESULTS_TABLES / 'data_quality_report.csv'}")
