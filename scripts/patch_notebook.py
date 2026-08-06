"""Patch the pipeline notebook: add progress/status logging to long-running cells.

Idempotent — replaces the source of specific cells matched by a stable substring.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "traffic_sign_detection_pipeline.ipynb"

# (match_substring, new_source_lines)
PATCHES = [
    (
        "from src.training.train_yolo import train\nresults, best = train(",
        """import time
from src.training.train_yolo import train

print('⏳ YOLO training started — this is the longest step (~1.5–2.5 h on Colab T4 for 30 epochs).')
print('   Watch the per-epoch table below; live metrics also stream to runs/detect/yolo_baseline/results.csv')
_t0 = time.time()
results, best = train(model='yolov8n.pt', epochs=30, name='yolo_baseline')
print(f'✅ YOLO training done in {(time.time()-_t0)/60:.1f} min')
print('best checkpoint:', best)
""",
    ),
    (
        "from src.evaluation.evaluate_yolo import evaluate\nfrom src.utils.paths import RESULTS_METRICS, ensure_dir\nimport json\nmetrics = evaluate(",
        """import time, json
from src.evaluation.evaluate_yolo import evaluate
from src.utils.paths import RESULTS_METRICS, ensure_dir

print('⏳ Evaluating YOLO on the test split + benchmarking FPS...')
_t0 = time.time()
metrics = evaluate(best, split='test')
ensure_dir(RESULTS_METRICS)
(RESULTS_METRICS / 'yolo_baseline.json').write_text(json.dumps(metrics, indent=2))
print(f'✅ YOLO eval done in {(time.time()-_t0):.0f}s → results/metrics/yolo_baseline.json')
metrics
""",
    ),
    (
        "from src.data.convert_to_coco import convert; convert()",
        """import time, json
from src.data.convert_to_coco import convert
from src.training.train_detr import train
from src.evaluation.evaluate_detr import evaluate
from src.utils.paths import RESULTS_METRICS, ensure_dir

print('⏳ [1/3] Converting YOLO labels → COCO JSON...')
_t0 = time.time(); convert(); print(f'   ✅ conversion done in {(time.time()-_t0):.0f}s')

print('⏳ [2/3] Fine-tuning DETR (facebook/detr-resnet-50) — long step; per-epoch val_loss prints below.')
_t0 = time.time()
model_dir = train(epochs=10, batch=2, name='detr_baseline')
print(f'   ✅ DETR training done in {(time.time()-_t0)/60:.1f} min')

print('⏳ [3/3] Evaluating DETR on the test split (mAP + FPS)...')
_t0 = time.time()
m = evaluate(model_dir, split='test')
ensure_dir(RESULTS_METRICS)
(RESULTS_METRICS / 'detr_baseline.json').write_text(json.dumps(m, indent=2))
print(f'   ✅ DETR eval done in {(time.time()-_t0)/60:.1f} min → results/metrics/detr_baseline.json')
m
""",
    ),
]


def main() -> None:
    nb = json.loads(NB.read_text())
    patched = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for match, new_src in PATCHES:
            if match in src and "⏳" not in src:  # don't double-patch
                cell["source"] = new_src.splitlines(keepends=True)
                cell["outputs"] = []
                cell["execution_count"] = None
                patched += 1
                break
    NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"patched {patched} cells")


if __name__ == "__main__":
    main()
