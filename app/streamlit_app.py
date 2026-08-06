"""Streamlit report dashboard and demo for traffic-sign detection."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
except ImportError:
    av = None
    RTCConfiguration = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "app" / "outputs"
DETECTION_COLUMNS = ["class", "confidence", "x1", "y1", "x2", "y2"]
REMOTE_WEIGHTS_URL = os.environ.get(
    "TSD_REMOTE_WEIGHTS_URL",
    "https://huggingface.co/datasets/vancevo/best_yolo_sign/resolve/main/best.pt",
)
CACHE_WEIGHTS = Path(tempfile.gettempdir()) / "traffic-sign-yolo" / "best.pt"
VIDEO_STABILIZER_IOU = 0.25
VIDEO_STABILIZER_SMOOTHING = 0.65
VIDEO_STABILIZER_MAX_AGE = 4
VIDEO_STABILIZER_MIN_HITS = 2

# ── Small-object defaults ────────────────────────────────────────────────────
# The training data is dominated by close-up signs (~41% of it is GTSRB-style crops
# where the sign fills ~67% of the frame), so the model is weak on the regime that
# actually matters in dashcam footage: a 30-50px sign in a 1920x1080 frame. At
# imgsz=640 such a frame is letterboxed by 0.33x and a 40px sign becomes 13px —
# barely above the stride-8 P3 head. These defaults push inference toward that
# regime instead; see DEFAULT_* below and the "Video thực tế" sidebar section.
DEFAULT_CONF = 0.20
DEFAULT_IOU = 0.70
DEFAULT_IMGSZ = 1280
DEFAULT_DROP_BOTTOM_PCT = 30
DEFAULT_TILE_SIZE = 640
DEFAULT_TILE_OVERLAP = 0.20
TILE_BATCH = 8


def latest_output_dir() -> Path | None:
    output_root = REPO_ROOT / "outputs"
    if not output_root.exists():
        return None

    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "weights" / "yolo" / "yolo_baseline" / "best.pt").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def latest_trained_weights() -> Path | None:
    output_root = REPO_ROOT / "outputs"
    if not output_root.exists():
        return None

    patterns = [
        "*/weights/best.pt",
        "*/weights/yolo/yolov8n_cbam_attention/best.pt",
        "*/weights/yolo/yolo_baseline/best.pt",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in output_root.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def resolve_default_weights() -> Path:
    override = os.environ.get("TSD_YOLO_WEIGHTS")
    if override:
        return Path(override).expanduser().resolve()

    latest_weights = latest_trained_weights()
    if latest_weights is not None:
        return latest_weights

    fallback = REPO_ROOT / "weights" / "yolo" / "yolo_baseline" / "best.pt"
    if fallback.exists():
        return fallback
    return CACHE_WEIGHTS


def resolve_default_metrics() -> Path:
    latest = latest_output_dir()
    if latest is not None:
        return latest / "results" / "metrics" / "yolo_baseline.json"
    return REPO_ROOT / "results" / "metrics" / "yolo_baseline.json"


def report_asset(*parts: str) -> Path:
    latest = latest_output_dir()
    if latest is not None:
        candidate = latest.joinpath(*parts)
        if candidate.exists():
            return candidate
    return REPO_ROOT.joinpath(*parts)


DEFAULT_WEIGHTS = resolve_default_weights()
YOLO_METRICS = resolve_default_metrics()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo(weights_path: str):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tsd-matplotlib"))
    weights = ensure_weights_available(Path(weights_path))

    try:
        from src.models.yolo_attention import register_yolo_attention_modules

        register_yolo_attention_modules()
    except Exception:
        pass

    from ultralytics import YOLO

    return YOLO(str(weights))


@st.cache_data(show_spinner="Downloading YOLO weights...")
def download_remote_weights(url: str, destination: str) -> str:
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".download")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    return str(dst)


def ensure_weights_available(path: Path) -> Path:
    if path.exists():
        return path
    if path == CACHE_WEIGHTS:
        return Path(download_remote_weights(REMOTE_WEIGHTS_URL, str(CACHE_WEIGHTS)))
    return path


@st.cache_data(show_spinner=False)
def load_metrics(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def load_checkpoint_summary(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import torch

        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"error": str(exc)}

    args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
    metrics = ckpt.get("train_metrics", {}) if isinstance(ckpt, dict) else {}
    return {
        "model": args.get("model"),
        "epochs": args.get("epochs"),
        "imgsz": args.get("imgsz"),
        "device": args.get("device") or "auto",
        "seed": args.get("seed"),
        "date": ckpt.get("date") if isinstance(ckpt, dict) else None,
        "map50_val": metrics.get("metrics/mAP50(B)"),
        "map50_95_val": metrics.get("metrics/mAP50-95(B)"),
        "precision_val": metrics.get("metrics/precision(B)"),
        "recall_val": metrics.get("metrics/recall(B)"),
    }


def check_model_ready() -> None:
    global DEFAULT_WEIGHTS
    DEFAULT_WEIGHTS = ensure_weights_available(DEFAULT_WEIGHTS)
    if not DEFAULT_WEIGHTS.exists():
        st.error(
            f"Missing YOLO weights: {DEFAULT_WEIGHTS}. "
            f"Set TSD_REMOTE_WEIGHTS_URL or TSD_YOLO_WEIGHTS if the model URL changes."
        )
        st.stop()


def dataset_split_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Tập": "Huấn luyện", "Số ảnh": 3530, "Tỉ lệ": "71.0%", "Số vật thể": 4298, "Vai trò": "Cập nhật trọng số"},
            {"Tập": "Kiểm định", "Số ảnh": 801, "Tỉ lệ": "16.1%", "Số vật thể": 944, "Vai trò": "Chọn checkpoint"},
            {"Tập": "Kiểm tra", "Số ảnh": 638, "Tỉ lệ": "12.8%", "Số vật thể": 770, "Vai trò": "Báo cáo kết quả"},
        ]
    )


def experiment_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Mã": "E0",
                "Cấu hình": "YOLOv8n",
                "Cách làm / thay đổi": "Mô hình nền, train 30 epoch",
                "mAP@0.5": 0.9633,
                "mAP@.5:.95": 0.8068,
                "FPS": 65.3,
                "Kết luận": "Mốc tốt nhất về độ chính xác",
            },
            {
                "Mã": "E0",
                "Cấu hình": "DETR-ResNet50",
                "Cách làm / thay đổi": "Transformer baseline, 10 epoch",
                "mAP@0.5": 0.1220,
                "mAP@.5:.95": 0.0988,
                "FPS": 15.7,
                "Kết luận": "Chưa hội tụ, không dùng demo",
            },
            {
                "Mã": "E1",
                "Cấu hình": "YOLO11n",
                "Cách làm / thay đổi": "YOLO thế hệ mới, nhẹ hơn YOLOv8n",
                "mAP@0.5": 0.9252,
                "mAP@.5:.95": 0.7833,
                "FPS": 59.0,
                "Kết luận": "Tốt nhất trong 3 mô hình bổ sung, nhưng kém YOLOv8n",
            },
            {
                "Mã": "E1",
                "Cấu hình": "SSDLite-MNv3",
                "Cách làm / thay đổi": "Mobile CNN, 320px, GFLOPs rất thấp",
                "mAP@0.5": 0.7432,
                "mAP@.5:.95": 0.6260,
                "FPS": 57.9,
                "Kết luận": "Định vị kém, gần như mù với vật thể nhỏ",
            },
            {
                "Mã": "E1",
                "Cấu hình": "D-FINE-Nano",
                "Cách làm / thay đổi": "Transformer thế hệ mới, nhẹ hơn DETR",
                "mAP@0.5": 0.7836,
                "mAP@.5:.95": 0.6589,
                "FPS": 25.8,
                "Kết luận": "Cứu được DETR, nhưng phân loại còn yếu",
            },
            {
                "Mã": "E2",
                "Cấu hình": "PyTorch FP16",
                "Cách làm / thay đổi": "Giữ YOLOv8n, đổi số học sang FP16",
                "mAP@0.5": 0.9633,
                "mAP@.5:.95": 0.8068,
                "FPS": 65.3,
                "Kết luận": "Không mất độ chính xác, giảm khoảng 49% dung lượng",
            },
            {
                "Mã": "E2",
                "Cấu hình": "ONNX INT8",
                "Cách làm / thay đổi": "Post-training quantization bằng ONNX",
                "mAP@0.5": 0.9552,
                "mAP@.5:.95": 0.7973,
                "FPS": 5.7,
                "Kết luận": "Nén tốt nhất, nhưng runtime CPU chậm trong thử nghiệm",
            },
            {
                "Mã": "E2",
                "Cấu hình": "OpenVINO INT8",
                "Cách làm / thay đổi": "INT8 qua OpenVINO trên CPU",
                "mAP@0.5": 0.9265,
                "mAP@.5:.95": 0.7846,
                "FPS": 13.9,
                "Kết luận": "Nhanh hơn ONNX INT8, nhưng mất AP nhỏ nhiều",
            },
            {
                "Mã": "E3",
                "Cấu hình": "YOLO26n đối chứng",
                "Cách làm / thay đổi": "Student YOLO26n train thường 50 epoch",
                "mAP@0.5": 0.9348,
                "mAP@.5:.95": 0.7773,
                "FPS": 69.7,
                "Kết luận": "Đối chứng mạnh, nhưng vẫn dưới YOLOv8n",
            },
            {
                "Mã": "E3",
                "Cấu hình": "YOLO26n chưng cất",
                "Cách làm / thay đổi": "Student học từ YOLO26s, dis=6.0",
                "mAP@0.5": 0.9014,
                "mAP@.5:.95": 0.7450,
                "FPS": 74.8,
                "Kết luận": "Kết quả âm: chưng cất kém đối chứng",
            },
        ]
    )


def notebook_source_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Notebook": "traffic_sign_3models_COLAB.ipynb",
                "Phần trong báo cáo": "E1 - Ba kiến trúc nhẹ",
                "Kết quả khớp": "YOLO11n 0.9252, SSDLite 0.7432, D-FINE 0.7836",
                "Nhận xét": "Khớp với report; chứng minh YOLOv8n vẫn là mốc tốt nhất.",
            },
            {
                "Notebook": "1_traffic_sign_quantization_COLAB.ipynb",
                "Phần trong báo cáo": "E2 - Lượng tử hoá",
                "Kết quả khớp": "FP16/ONNX FP32 0.9633, ONNX INT8 0.9552, OpenVINO INT8 0.9265",
                "Nhận xét": "Khớp với report; INT8 nén mạnh nhưng phụ thuộc runtime.",
            },
            {
                "Notebook": "2_BO_SUNG_OPENVINO_COLAB.ipynb",
                "Phần trong báo cáo": "E2 - Bổ sung OpenVINO",
                "Kết quả khớp": "OpenVINO INT8 0.9265, nhanh hơn ONNX INT8 trên CPU",
                "Nhận xét": "Khớp với report; cần đọc AP vật thể nhỏ khi chọn OpenVINO.",
            },
            {
                "Notebook": "giaidoan2.ipynb",
                "Phần trong báo cáo": "E3 - Chưng cất tri thức",
                "Kết quả khớp": "Thầy 0.9520, trò đối chứng 0.9348, trò chưng cất 0.9014",
                "Nhận xét": "Khớp với report; chưng cất là kết quả âm trong điều kiện dis=6.0.",
            },
        ]
    )


def selected_model_table(metrics: dict, ckpt: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Thuộc tính": "Model dùng demo", "Giá trị": "YOLOv8n PyTorch best.pt"},
            {"Thuộc tính": "Checkpoint", "Giá trị": display_path(DEFAULT_WEIGHTS)},
            {"Thuộc tính": "Nguồn", "Giá trị": "outputs mới nhất / YOLO baseline 30 epoch"},
            {"Thuộc tính": "Epoch train", "Giá trị": ckpt.get("epochs", 30)},
            {"Thuộc tính": "mAP@0.5 test", "Giá trị": f"{metrics.get('map50', 0.953857):.4f}"},
            {"Thuộc tính": "mAP@0.5:0.95 test", "Giá trị": f"{metrics.get('map50_95', 0.805803):.4f}"},
            {"Thuộc tính": "Precision / Recall", "Giá trị": f"{metrics.get('precision', 0.9084):.4f} / {metrics.get('recall', 0.9564):.4f}"},
            {"Thuộc tính": "FPS đo trong output", "Giá trị": f"{metrics.get('fps', 113.2):.1f}"},
            {"Thuộc tính": "Dung lượng", "Giá trị": f"{metrics.get('model_size_mb', 6.25):.2f} MB"},
        ]
    )


def render_metric_cards(metrics: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("mAP@0.5", f"{metrics.get('map50', 0):.4f}")
    cols[1].metric("mAP@0.5:0.95", f"{metrics.get('map50_95', 0):.4f}")
    cols[2].metric("FPS", f"{metrics.get('fps', 0):.1f}")
    cols[3].metric("Model size", f"{metrics.get('model_size_mb', 0):.2f} MB")


def render_plot_grid(paths: list[Path], captions: list[str]) -> None:
    existing = [(p, c) for p, c in zip(paths, captions) if p.exists()]
    if not existing:
        st.info("Chưa tìm thấy biểu đồ đã xuất trong results/outputs.")
        return

    for i in range(0, len(existing), 2):
        cols = st.columns(2)
        for col, (path, caption) in zip(cols, existing[i : i + 2]):
            col.image(str(path), caption=caption, width="stretch")


def sidebar_controls(metrics: dict) -> dict:
    with st.sidebar:
        st.subheader("Model demo")
        st.code(display_path(DEFAULT_WEIGHTS))
        conf = st.slider("Confidence", 0.05, 0.95, DEFAULT_CONF, 0.05)
        iou = st.slider("IoU", 0.10, 0.90, DEFAULT_IOU, 0.05)
        imgsz = st.select_slider(
            "Image size",
            options=[320, 416, 512, 640, 768, 960, 1280, 1536],
            value=DEFAULT_IMGSZ,
        )
        duration = st.slider("Session duration", 10, 15, 10, 1)
        inference_fps = st.slider("Inference FPS cap", 1, 15, 8, 1)

        with st.expander("Video thực tế (dashcam)", expanded=True):
            st.caption(
                "Dữ liệu train nghiêng nặng về biển cận cảnh, nên biển nhỏ ở xa là điểm yếu. "
                "Các tuỳ chọn dưới đây bù lại điều đó khi chạy video đường thật."
            )
            drop_bottom_pct = st.slider("Bỏ phần dưới khung hình (%)", 0, 50, DEFAULT_DROP_BOTTOM_PCT, 5)
            min_hits = st.slider("Số frame xác nhận", 1, 5, VIDEO_STABILIZER_MIN_HITS, 1)
            use_tiling = st.checkbox("Tiled inference (biển rất nhỏ)", value=False)
            tile = st.select_slider("Tile size", options=[512, 640, 768], value=DEFAULT_TILE_SIZE)
            overlap = st.slider("Tile overlap", 0.0, 0.4, DEFAULT_TILE_OVERLAP, 0.05)
            if use_tiling:
                st.warning("Tiling chạy chậm hơn nhiều lần — chỉ nên dùng cho clip ngắn.")

        if metrics:
            st.subheader("Kết quả model")
            st.metric("mAP50", f"{metrics.get('map50', 0):.3f}")
            st.metric("mAP50-95", f"{metrics.get('map50_95', 0):.3f}")
            st.metric("FPS", f"{metrics.get('fps', 0):.1f}")

    return {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "duration": duration,
        "inference_fps": inference_fps,
        "drop_bottom_pct": drop_bottom_pct,
        "min_hits": min_hits,
        "use_tiling": use_tiling,
        "tile": tile,
        "overlap": overlap,
    }


def render_overview_tab() -> None:
    st.subheader("Bài toán")
    st.write(
        "Đề tài giải quyết bài toán phát hiện biển báo giao thông từ ảnh/video. "
        "Đầu ra gồm hộp bao, nhãn lớp và độ tin cậy. Nhóm không chỉ đo một con số mAP, "
        "mà tách rõ hai khâu: định vị biển báo và phân loại đúng loại biển."
    )

    st.subheader("Cách tiếp cận")
    roadmap = pd.DataFrame(
        [
            {"Giai đoạn": "E0 - Mô hình nền", "Câu hỏi": "Mốc so sánh là bao nhiêu?", "Cách làm": "Train YOLOv8n và DETR-R50"},
            {"Giai đoạn": "E1 - Kiến trúc nhẹ", "Câu hỏi": "Có model nhẹ hơn mà không kém hơn không?", "Cách làm": "Thử YOLO11n, SSDLite, D-FINE"},
            {"Giai đoạn": "E2 - Lượng tử hoá", "Câu hỏi": "Có thể nén model tốt nhất mà không train lại không?", "Cách làm": "FP16, ONNX INT8, OpenVINO INT8"},
            {"Giai đoạn": "E3 - Chưng cất", "Câu hỏi": "Student nhỏ có học được từ teacher lớn không?", "Cách làm": "YOLO26s dạy YOLO26n"},
        ]
    )
    st.dataframe(roadmap, hide_index=True, width="stretch")


def render_data_analysis_tab() -> None:
    st.subheader("Bộ dữ liệu")
    cols = st.columns(4)
    cols[0].metric("Ảnh", "4,969")
    cols[1].metric("Vật thể", "6,012")
    cols[2].metric("Lớp", "15")
    cols[3].metric("Test sạch", "573 ảnh")

    st.dataframe(dataset_split_table(), hide_index=True, width="stretch")

    st.subheader("Nhận xét từ phân tích dữ liệu")
    st.markdown(
        """
- Mất cân bằng lớp rất rõ: lớp phổ biến nhất có 787 mẫu, lớp hiếm nhất Speed Limit 10 chỉ có 22 mẫu.
- 35.3% vật thể thuộc nhóm nhỏ theo chuẩn COCO, nên biển ở xa là điểm khó chính.
- Ảnh nguồn đồng nhất 416x416; tăng input size quá cao chủ yếu là phóng to, không tạo thêm chi tiết.
- Có rò rỉ dữ liệu giữa các tập: 65/638 ảnh test có bản sao trong train, nên report đo thêm cột "sạch".
- Các speed sign dễ nhầm vì khác nhau chủ yếu ở chữ số nhỏ bên trong biển.
"""
    )

    render_plot_grid(
        [
            report_asset("results", "eda", "class_distribution.png"),
            report_asset("results", "eda", "bbox_size_categories.png"),
            report_asset("results", "eda", "object_center_heatmap.png"),
            report_asset("results", "eda", "image_resolution.png"),
        ],
        [
            "Phân bố lớp",
            "Kích thước bounding box",
            "Heatmap vị trí vật thể",
            "Độ phân giải ảnh",
        ],
    )


def render_experiments_tab() -> None:
    st.subheader("Các hướng đã thử và kết quả")
    df = experiment_table()
    st.dataframe(df, hide_index=True, width="stretch")

    st.subheader("Đọc kết quả theo từng thí nghiệm")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**E0 - Baseline**

YOLOv8n đạt mAP cao và tốc độ tốt. DETR-R50 thấp vì chưa hội tụ trong 10 epoch, không phải vì transformer chắc chắn kém.

**E1 - Kiến trúc nhẹ**

YOLO11n nhẹ hơn nhưng kém YOLOv8n. SSDLite định vị kém với vật thể nhỏ. D-FINE định vị tốt hơn SSDLite nhưng phân loại yếu hơn.
"""
        )
    with c2:
        st.markdown(
            """
**E2 - Lượng tử hoá**

FP16 gần như không mất độ chính xác. ONNX INT8 nén rất tốt nhưng chậm trong runtime thử nghiệm. OpenVINO INT8 nhanh hơn ONNX INT8 nhưng giảm AP vật thể nhỏ.

**E3 - Chưng cất tri thức**

YOLO26n chưng cất kém hơn YOLO26n đối chứng. Kết quả âm này được giữ lại vì nó chỉ ra điều kiện áp dụng chưa phù hợp.
"""
        )

    st.subheader("Notebook và mức khớp với report")
    st.dataframe(notebook_source_table(), hide_index=True, width="stretch")


def render_selected_model_tab(metrics: dict, ckpt: dict) -> None:
    st.subheader("Model được chọn cho demo")
    render_metric_cards(metrics)
    st.dataframe(selected_model_table(metrics, ckpt), hide_index=True, width="stretch")


def result_rows(result: Any) -> list[dict]:
    rows = []
    names = result.names
    if result.boxes is None:
        return rows

    for box in result.boxes:
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rows.append(
            {
                "class": names.get(cls_id, str(cls_id)),
                "confidence": round(float(box.conf.item()), 3),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )
    return rows


def bbox_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def class_color(name: str) -> tuple[int, int, int]:
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
    ]
    return palette[abs(hash(name)) % len(palette)]


def draw_detections(frame_bgr, rows: list[dict]):
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]

    for row in rows:
        x1 = int(max(0, min(width - 1, row["x1"])))
        y1 = int(max(0, min(height - 1, row["y1"])))
        x2 = int(max(0, min(width - 1, row["x2"])))
        y2 = int(max(0, min(height - 1, row["y2"])))
        color = class_color(row["class"])
        label = f"{row['class']} {row['confidence']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        label_y = max(y1, text_h + baseline + 4)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - baseline - 4),
            (min(width - 1, x1 + text_w + 8), label_y),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


class DetectionStabilizer:
    """Simple IoU tracker that smooths boxes and survives short missed detections."""

    def __init__(
        self,
        iou_threshold: float = VIDEO_STABILIZER_IOU,
        smoothing: float = VIDEO_STABILIZER_SMOOTHING,
        max_age: int = VIDEO_STABILIZER_MAX_AGE,
        min_hits: int = VIDEO_STABILIZER_MIN_HITS,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.smoothing = smoothing
        self.max_age = max_age
        # A detection must be confirmed on `min_hits` frames before it is drawn. This is
        # what makes a low confidence threshold usable: real signs persist across frames,
        # one-frame noise does not.
        self.min_hits = max(1, min_hits)
        self.tracks: list[dict] = []
        self.next_id = 1

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        existing_track_count = len(self.tracks)

        candidates: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if track["class"] != det["class"]:
                    continue
                iou = bbox_iou(track, det)
                if iou >= self.iou_threshold:
                    candidates.append((iou, ti, di))

        for _, ti, di in sorted(candidates, reverse=True):
            if ti in matched_tracks or di in matched_detections:
                continue
            track = self.tracks[ti]
            det = detections[di]
            keep = self.smoothing
            take = 1.0 - keep
            for key in ("x1", "y1", "x2", "y2"):
                track[key] = round(keep * float(track[key]) + take * float(det[key]), 1)
            track["confidence"] = round(0.55 * float(track["confidence"]) + 0.45 * float(det["confidence"]), 3)
            track["missed"] = 0
            track["hits"] += 1
            matched_tracks.add(ti)
            matched_detections.add(di)

        for ti in range(existing_track_count):
            if ti not in matched_tracks:
                self.tracks[ti]["missed"] += 1

        for di, det in enumerate(detections):
            if di in matched_detections:
                continue
            self.tracks.append(
                {
                    **det,
                    "track_id": self.next_id,
                    "hits": 1,
                    "missed": 0,
                }
            )
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track["missed"] <= self.max_age]
        return self.current()

    def current(self) -> list[dict]:
        rows = []
        for track in self.tracks:
            if track["hits"] < self.min_hits:
                continue
            row = {key: track[key] for key in DETECTION_COLUMNS}
            if track["missed"]:
                row["confidence"] = round(max(0.05, row["confidence"] * (0.85 ** track["missed"])), 3)
            rows.append(row)
        return rows


def nms_rows(rows: list[dict], iou_threshold: float) -> list[dict]:
    """Greedy per-class NMS over detection rows.

    Needed because tiled inference runs the model several times on overlapping crops,
    so the same sign can be reported by 2-4 tiles plus the full-frame pass.
    """
    kept: list[dict] = []
    for cls in {row["class"] for row in rows}:
        group = sorted(
            (row for row in rows if row["class"] == cls),
            key=lambda row: -row["confidence"],
        )
        selected: list[dict] = []
        for candidate in group:
            if all(bbox_iou(candidate, chosen) < iou_threshold for chosen in selected):
                selected.append(candidate)
        kept.extend(selected)
    return sorted(kept, key=lambda row: -row["confidence"])


def tile_origins(total: int, tile: int, step: int) -> list[int]:
    """Start offsets so that tiles of `tile` px cover `total` px with `step` stride.

    The last origin is snapped to `total - tile` so the right/bottom edge is always
    fully covered instead of being cropped off.
    """
    if total <= tile:
        return [0]
    origins = list(range(0, total - tile + 1, step))
    if origins[-1] != total - tile:
        origins.append(total - tile)
    return origins


def apply_roi(frame_bgr, drop_bottom_pct: int):
    """Keep only the top (100 - drop_bottom_pct)% of the frame.

    In dashcam footage the bottom of the frame is road surface and bonnet — no signs
    live there, so cropping it both speeds up inference and removes false positives.
    The crop keeps the top-left origin, so detection coordinates need no offset.
    """
    if drop_bottom_pct <= 0:
        return frame_bgr
    height = frame_bgr.shape[0]
    keep = max(1, int(height * (1.0 - drop_bottom_pct / 100.0)))
    return frame_bgr[:keep]


def tiled_rows(model, frame_bgr, conf: float, iou: float, imgsz: int,
               tile: int, overlap: float) -> list[dict]:
    """Run detection on overlapping tiles plus one full-frame pass, then merge.

    A sign that is 30px in the full frame is ~30px inside a 640px tile too — no
    downscaling — which is the whole point. The extra full-frame pass catches signs
    larger than a single tile, which tiling alone would cut in half.
    """
    height, width = frame_bgr.shape[:2]
    if height <= tile and width <= tile:
        return result_rows(
            model.predict(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        )

    step = max(1, int(tile * (1.0 - overlap)))
    crops, offsets = [], []
    for y in tile_origins(height, tile, step):
        for x in tile_origins(width, tile, step):
            crops.append(frame_bgr[y : y + tile, x : x + tile])
            offsets.append((x, y))

    rows: list[dict] = []
    for i in range(0, len(crops), TILE_BATCH):
        chunk = crops[i : i + TILE_BATCH]
        results = model.predict(chunk, conf=conf, iou=iou, imgsz=tile, verbose=False)
        for result, (off_x, off_y) in zip(results, offsets[i : i + TILE_BATCH]):
            for row in result_rows(result):
                row["x1"] = round(row["x1"] + off_x, 1)
                row["x2"] = round(row["x2"] + off_x, 1)
                row["y1"] = round(row["y1"] + off_y, 1)
                row["y2"] = round(row["y2"] + off_y, 1)
                rows.append(row)

    rows.extend(
        result_rows(model.predict(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0])
    )
    return nms_rows(rows, iou)


def detect_rows(frame_bgr, settings: dict, *, use_roi: bool = True) -> list[dict]:
    """Single entry point for every demo path. Returns rows in frame coordinates."""
    model = load_yolo(str(DEFAULT_WEIGHTS))
    working = apply_roi(frame_bgr, settings["drop_bottom_pct"]) if use_roi else frame_bgr

    if settings["use_tiling"]:
        return tiled_rows(
            model,
            working,
            conf=settings["conf"],
            iou=settings["iou"],
            imgsz=settings["imgsz"],
            tile=settings["tile"],
            overlap=settings["overlap"],
        )

    result = model.predict(
        working,
        conf=settings["conf"],
        iou=settings["iou"],
        imgsz=settings["imgsz"],
        verbose=False,
    )[0]
    return result_rows(result)


def browser_safe_frame(frame_bgr):
    height, width = frame_bgr.shape[:2]
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    return frame_bgr[:even_height, :even_width]


def transcode_for_browser(source_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Không tìm thấy ffmpeg để chuyển video sang H.264. "
            "Khi deploy Streamlit Cloud, thêm `ffmpeg` vào packages.txt rồi redeploy."
        )

    output_path = source_path.with_name(f"{source_path.stem}_browser.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.exists():
        details = (result.stderr or result.stdout or "không có log").strip()
        raise RuntimeError(f"Không chuyển video sang H.264 được: {details[-800:]}")
    return output_path


def summarize_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["class", "detections", "avg_confidence", "max_confidence"])

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("class", as_index=False)
        .agg(
            detections=("class", "size"),
            avg_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
        )
        .sort_values(["detections", "max_confidence"], ascending=False)
    )
    summary["avg_confidence"] = summary["avg_confidence"].round(3)
    summary["max_confidence"] = summary["max_confidence"].round(3)
    return summary


class TrafficSignVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.settings: dict = {
            "conf": DEFAULT_CONF,
            "iou": DEFAULT_IOU,
            "imgsz": DEFAULT_IMGSZ,
            "drop_bottom_pct": DEFAULT_DROP_BOTTOM_PCT,
            "use_tiling": False,
            "tile": DEFAULT_TILE_SIZE,
            "overlap": DEFAULT_TILE_OVERLAP,
        }
        self.duration = 10
        self.max_inference_fps = 8
        self.started_at = time.monotonic()
        self.last_inference_at = 0.0
        self.lock = threading.Lock()
        self.frame_count = 0
        self.processed_frames = 0
        self.detected_rows: list[dict] = []
        self.class_counter: Counter[str] = Counter()
        self.stabilizer = DetectionStabilizer()

    def configure(self, settings: dict) -> None:
        with self.lock:
            self.settings = dict(settings)
            self.duration = settings["duration"]
            self.max_inference_fps = settings["inference_fps"]
            self.stabilizer.min_hits = max(1, settings["min_hits"])

    def reset(self) -> None:
        with self.lock:
            self.started_at = time.monotonic()
            self.last_inference_at = 0.0
            self.frame_count = 0
            self.processed_frames = 0
            self.detected_rows = []
            self.class_counter = Counter()
            self.stabilizer.reset()

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = time.monotonic() - self.started_at
            return {
                "elapsed": elapsed,
                "duration": self.duration,
                "frame_count": self.frame_count,
                "processed_frames": self.processed_frames,
                "detections": sum(self.class_counter.values()),
                "class_counter": dict(self.class_counter),
                "rows": list(self.detected_rows),
                "complete": elapsed >= self.duration,
            }

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        now = time.monotonic()

        with self.lock:
            self.frame_count += 1
            elapsed = now - self.started_at
            should_detect = elapsed <= self.duration and (
                now - self.last_inference_at >= 1.0 / max(self.max_inference_fps, 1)
            )
            settings = dict(self.settings)
            stable_rows = self.stabilizer.current()

        annotated = draw_detections(frame_bgr, stable_rows) if stable_rows else frame_bgr.copy()

        if should_detect:
            rows = detect_rows(frame_bgr, settings)

            with self.lock:
                stable_rows = self.stabilizer.update(rows)
                self.last_inference_at = now
                self.processed_frames += 1
                self.detected_rows.extend(rows)
                self.class_counter.update(row["class"] for row in rows)

            annotated = draw_detections(frame_bgr, stable_rows)

        with self.lock:
            remaining = max(0.0, self.duration - (time.monotonic() - self.started_at))
            complete = remaining <= 0.0

        label = "complete" if complete else f"{remaining:0.1f}s"
        color = (0, 180, 255) if complete else (0, 255, 0)
        cv2.putText(annotated, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def render_webcam_demo(settings: dict) -> None:
    if webrtc_streamer is None:
        st.error("Thiếu webcam dependencies. Cài bằng: pip install streamlit-webrtc av")
        return

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="traffic-sign-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=TrafficSignVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.configure(settings)
        if st.button("Reset webcam session"):
            ctx.video_processor.reset()

        stats_placeholder = st.empty()
        table_placeholder = st.empty()
        caption_placeholder = st.empty()

        while ctx.state.playing:
            stats = ctx.video_processor.snapshot()
            elapsed = min(stats["elapsed"], stats["duration"])
            processed_fps = stats["processed_frames"] / max(elapsed, 0.01)
            top_classes = ", ".join(
                f"{name} ({count})" for name, count in Counter(stats["class_counter"]).most_common(3)
            )

            stats_placeholder.metric("Session", f"{elapsed:0.1f}s / {stats['duration']}s")
            cols = table_placeholder.columns(3)
            cols[0].metric("Processed frames", stats["processed_frames"])
            cols[1].metric("Avg inference FPS", f"{processed_fps:0.1f}")
            cols[2].metric("Detections", stats["detections"])

            if top_classes:
                caption_placeholder.caption(f"Top classes: {top_classes}")
            else:
                caption_placeholder.empty()

            time.sleep(0.5)
            if stats["complete"]:
                break

        rows = ctx.video_processor.snapshot()["rows"]
        summary = summarize_rows(rows)
        if not summary.empty:
            st.dataframe(summary, hide_index=True, width="stretch")


def process_video_file(uploaded_file, settings: dict) -> tuple[Path, pd.DataFrame, dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        input_path = Path(tmp.name)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Không đọc được video. Hãy thử MP4/H.264.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width -= width % 2
    height -= height % 2
    max_frames = int(settings["duration"] * fps)

    output_path = OUTPUT_DIR / f"annotated_{int(time.time())}.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        input_path.unlink(missing_ok=True)
        raise RuntimeError("Không tạo được video output. Hãy thử video MP4/H.264 khác.")

    all_rows: list[dict] = []
    processed = 0
    progress = st.progress(0)
    stabilizer = DetectionStabilizer(min_hits=settings["min_hits"])

    while processed < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_bgr = browser_safe_frame(frame_bgr)
        rows = detect_rows(frame_bgr, settings)
        stable_rows = stabilizer.update(rows)
        annotated = draw_detections(frame_bgr, stable_rows)
        writer.write(annotated)
        all_rows.extend(rows)
        processed += 1
        progress.progress(min(processed / max(max_frames, 1), 1.0))

    cap.release()
    writer.release()
    input_path.unlink(missing_ok=True)
    progress.empty()
    playable_path = transcode_for_browser(output_path)

    stats = {
        "processed_frames": processed,
        "source_fps": round(fps, 2),
        "duration_seconds": round(processed / max(fps, 1), 2),
        "detections": len(all_rows),
        "video_format": "H.264 / yuv420p",
        "imgsz": settings["imgsz"],
        "conf": settings["conf"],
        "roi_drop_bottom_pct": settings["drop_bottom_pct"],
        "tiling": "on" if settings["use_tiling"] else "off",
    }
    return playable_path, summarize_rows(all_rows), stats


def render_video_demo(settings: dict) -> None:
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="video-upload")
    st.caption(
        f"Đang chạy ở imgsz={settings['imgsz']}, conf={settings['conf']:.2f}, "
        f"bỏ {settings['drop_bottom_pct']}% đáy khung hình, "
        f"tiling {'BẬT' if settings['use_tiling'] else 'TẮT'}. "
        "Chỉnh trong sidebar nếu biển ở xa vẫn bị bỏ sót."
    )
    if uploaded_video is None:
        st.info("Upload video ngắn để detect từng frame bằng model đã chọn.")
        return

    if st.button("Process video"):
        with st.spinner("Running YOLO on video frames..."):
            try:
                output_path, summary, stats = process_video_file(uploaded_video, settings)
            except Exception as exc:
                st.error(str(exc))
                return

        cols = st.columns(4)
        cols[0].metric("Frames", stats["processed_frames"])
        cols[1].metric("Source FPS", stats["source_fps"])
        cols[2].metric("Seconds", stats["duration_seconds"])
        cols[3].metric("Detections", stats["detections"])

        if summary.empty:
            st.warning(
                "Không phát hiện biển báo ở ngưỡng confidence hiện tại. "
                "Thử hạ Confidence xuống 0.10, tăng Image size lên 1536, hoặc bật Tiled inference."
            )
        else:
            st.dataframe(summary, hide_index=True, width="stretch")

        video_bytes = output_path.read_bytes()
        st.video(video_bytes, format="video/mp4")
        st.download_button(
            "Download annotated video",
            video_bytes,
            file_name=output_path.name,
            mime="video/mp4",
        )


def render_image_demo(settings: dict) -> None:
    camera_image = st.camera_input("Camera snapshot")
    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"], key="image-upload")
    image_source = camera_image or uploaded_image

    st.caption(
        "Ảnh tĩnh không áp dụng cắt ROI (ảnh có thể được đóng khung bất kỳ), "
        "nhưng vẫn dùng imgsz và tiling từ sidebar."
    )

    if image_source is None:
        st.info("Chụp ảnh hoặc upload ảnh để chạy single-frame detection.")
        return

    image = Image.open(image_source).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    with st.spinner("Detecting traffic signs..."):
        # use_roi=False: a still photo is not necessarily a dashcam frame, so cropping
        # the bottom of it would be wrong.
        rows = detect_rows(frame_bgr, settings, use_roi=False)
        table = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
        annotated = draw_detections(frame_bgr, rows)

    left, right = st.columns([1.4, 1])
    with left:
        st.image(annotated, caption="Detections", channels="BGR", width="stretch")
    with right:
        st.metric("Detected signs", len(table))
        if table.empty:
            st.warning("Không có biển báo nào vượt ngưỡng confidence.")
        else:
            st.dataframe(table, hide_index=True, width="stretch")


def render_demo_tab(settings: dict) -> None:
    st.subheader("Demo nhận diện bằng mô hình tốt nhất")
    st.caption(f"Đang dùng: {display_path(DEFAULT_WEIGHTS)}")
    webcam_tab, video_tab, image_tab = st.tabs(["Webcam realtime", "Video upload", "Image snapshot"])
    with webcam_tab:
        render_webcam_demo(settings)
    with video_tab:
        render_video_demo(settings)
    with image_tab:
        render_image_demo(settings)


def main() -> None:
    st.set_page_config(page_title="Traffic Sign Detection Report", layout="wide")
    st.title("Traffic Sign Detection")

    check_model_ready()
    metrics = load_metrics(str(YOLO_METRICS))
    ckpt = load_checkpoint_summary(str(DEFAULT_WEIGHTS))
    settings = sidebar_controls(metrics)

    overview, data, experiments, selected, demo = st.tabs(
        ["Tổng Quan", "Phân Tích Dữ Liệu", "Thực Nghiệm Mô Hình", "Mô Hình Được Chọn", "Demo Nhận Diện"]
    )
    with overview:
        render_overview_tab()
    with data:
        render_data_analysis_tab()
    with experiments:
        render_experiments_tab()
    with selected:
        render_selected_model_tab(metrics, ckpt)
    with demo:
        render_demo_tab(settings)


if __name__ == "__main__":
    main()
