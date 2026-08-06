"""Structured training logger / monitor.

Ultralytics and our DETR loop both emit noisy console output (progress bars redraw with
carriage returns, which makes a tee'd file unreadable). This monitor instead snapshots the
*clean* per-epoch signal into a tracked `logs/` directory:

  - tails the most recent runs/detect/*/results.csv (YOLO) into logs/yolo_training.log
  - writes a compact, human-readable per-epoch summary (epoch, losses, P/R, mAP)

Run it in the background alongside training:
    python -m scripts.log_training --watch yolo
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs"
RUNS = REPO / "runs" / "detect"


def _latest_results_csv() -> Path | None:
    csvs = sorted(RUNS.glob("*/results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def _fmt_epoch(row: dict) -> str:
    def g(*keys, default="?"):
        for k in keys:
            for col in row:
                if col.strip() == k:
                    return row[col].strip()
        return default
    return (
        f"epoch={g('epoch'):>3} "
        f"box={g('train/box_loss'):>7} cls={g('train/cls_loss'):>7} "
        f"P={g('metrics/precision(B)'):>7} R={g('metrics/recall(B)'):>7} "
        f"mAP50={g('metrics/mAP50(B)'):>7} mAP50-95={g('metrics/mAP50-95(B)'):>7}"
    )


def watch_yolo(poll: float = 10.0) -> None:
    LOGS.mkdir(exist_ok=True)
    out = LOGS / "yolo_training.log"
    seen: set[str] = set()
    with out.open("a") as f:
        f.write(f"\n=== YOLO training watch started {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
        f.flush()
        while True:
            rc = _latest_results_csv()
            if rc and rc.exists():
                try:
                    rows = list(csv.DictReader(rc.read_text().splitlines()))
                except Exception:
                    rows = []
                for row in rows:
                    ep = (row.get("epoch") or "").strip()
                    if ep and ep not in seen:
                        seen.add(ep)
                        line = f"[{datetime.now():%H:%M:%S}] {_fmt_epoch(row)} (src={rc.parent.name})"
                        print(line, flush=True)
                        f.write(line + "\n")
                        f.flush()
            time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", choices=["yolo"], default="yolo")
    ap.add_argument("--poll", type=float, default=10.0)
    args = ap.parse_args()
    if args.watch == "yolo":
        watch_yolo(args.poll)


if __name__ == "__main__":
    main()
