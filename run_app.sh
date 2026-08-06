#!/usr/bin/env bash
# Launch the Streamlit demo locally, in an isolated venv.
#
#   ./run_app.sh                  # auto-pick an interpreter, port 8501
#   PORT=8600 ./run_app.sh        # different port
#   PYTHON=/path/to/python ./run_app.sh
#   RECREATE=1 ./run_app.sh       # rebuild the venv from scratch
#
# ── Why a venv instead of the ambient python ─────────────────────────────────
# Streamlit >= 1.12.1 declares `Requires-Python >=3.9,!=3.9.7`: the 3.9.7 patch
# release is explicitly BLACKLISTED (a typing bug in that exact build). pyenv's
# 3.9.7 therefore caps you at Streamlit 1.12.0, while this app needs >= 1.49 for
# width="stretch" on st.image/st.dataframe. There is no pin that resolves that —
# the interpreter has to change. So: find a usable one, build .venv-app, install
# there, and leave the system/pyenv site-packages untouched.
#
# Written for macOS's stock bash 3.2 — no associative arrays, no `mapfile`.
set -eu
cd "$(dirname "$0")"

PORT="${PORT:-8501}"
VENV=".venv-app"

# ── Is this interpreter allowed to run modern Streamlit? ─────────────────────
py_ok() {
  "$1" -c 'import sys
v = sys.version_info
ok = v[:2] >= (3, 10) or (v[:2] == (3, 9) and v[:3] != (3, 9, 7))
raise SystemExit(0 if ok else 1)' >/dev/null 2>&1
}

# ── .python-version (pyenv) ──────────────────────────────────────────────────
# Declaring the interpreter in-repo means `cd` into this directory and `python3`
# is already correct. pyenv fails loudly if the version isn't installed, so turn
# that into an actionable message instead of a confusing shim error.
if [ -z "${PYTHON:-}" ] && [ -f .python-version ]; then
  WANT="$(tr -d ' \t\r\n' < .python-version)"
  if [ -n "$WANT" ] && [ -x "$HOME/.pyenv/versions/$WANT/bin/python" ]; then
    PYTHON="$HOME/.pyenv/versions/$WANT/bin/python"
  elif [ -n "$WANT" ] && [ -d "$HOME/.pyenv" ]; then
    echo "ⓘ .python-version yêu cầu $WANT nhưng pyenv chưa cài bản này."
    echo "  Cài bằng:  pyenv install $WANT"
    echo "  Đang thử tìm interpreter khác..."
  fi
fi

BASE_PY=""
if [ -n "${PYTHON:-}" ]; then
  if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "✖ không tìm thấy '$PYTHON'" >&2; exit 1
  fi
  if ! py_ok "$PYTHON"; then
    echo "✖ '$PYTHON' là $("$PYTHON" -V 2>&1) — streamlit hiện đại không cài được trên bản này." >&2
    echo "  Cần Python >= 3.10 (hoặc 3.9.x nhưng KHÔNG phải 3.9.7)." >&2
    exit 1
  fi
  BASE_PY="$PYTHON"
else
  # Prefer newest first. pyenv installs are appended so an explicit python3.12
  # on PATH wins over an old pyenv default.
  CANDS="python3.14 python3.13 python3.12 python3.11 python3.10 python3 /usr/bin/python3"
  if [ -d "$HOME/.pyenv/versions" ]; then
    for d in "$HOME"/.pyenv/versions/*/bin/python3; do
      [ -x "$d" ] && CANDS="$CANDS $d"
    done
  fi
  for c in $CANDS; do
    command -v "$c" >/dev/null 2>&1 || continue
    if py_ok "$c"; then BASE_PY="$c"; break; fi
  done
fi

if [ -z "$BASE_PY" ]; then
  cat >&2 <<'MSG'
✖ Không tìm thấy Python nào chạy được streamlit hiện đại.

  Cần Python >= 3.10 (3.9.x cũng được, trừ đúng bản 3.9.7 bị streamlit chặn).
  Bạn đang dùng pyenv, nên cách nhanh nhất:

      pyenv install 3.11.9
      PYTHON=~/.pyenv/versions/3.11.9/bin/python ./run_app.sh

  Hoặc dùng Homebrew:

      brew install python@3.11
      PYTHON=$(brew --prefix)/bin/python3.11 ./run_app.sh
MSG
  exit 1
fi

echo "▶ interpreter : $("$BASE_PY" -V 2>&1)  ($(command -v "$BASE_PY" || echo "$BASE_PY"))"

# ── venv ─────────────────────────────────────────────────────────────────────
if [ "${RECREATE:-0}" = "1" ]; then rm -rf "$VENV"; fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "▶ tạo venv    : $VENV"
  "$BASE_PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
fi
PY="$VENV/bin/python"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

# ── deps ─────────────────────────────────────────────────────────────────────
need=""
"$PY" - <<'PYEOF' >/dev/null 2>&1 || need="$need streamlit>=1.49"
import streamlit
major, minor = (int(x) for x in streamlit.__version__.split(".")[:2])
raise SystemExit(0 if (major, minor) >= (1, 49) else 1)
PYEOF
"$PY" -c "import ultralytics" >/dev/null 2>&1 || need="$need ultralytics"
"$PY" -c "import cv2"         >/dev/null 2>&1 || need="$need opencv-python"
"$PY" -c "import pandas"      >/dev/null 2>&1 || need="$need pandas"

if [ -n "$need" ]; then
  echo "▶ cài đặt     :$need"
  echo "  (lần đầu mất vài phút — ultralytics kéo theo torch, ~200MB+)"
  "$PY" -m pip install --upgrade $need
fi

"$PY" -c "import streamlit_webrtc" >/dev/null 2>&1 || \
  echo "ⓘ chưa có streamlit-webrtc → tab 'Webcam realtime' báo thiếu deps; Video/Image vẫn chạy."
command -v ffmpeg >/dev/null 2>&1 || \
  echo "⚠ chưa có ffmpeg → tab 'Video upload' lỗi ở bước transcode. Cài: brew install ffmpeg"

# ── weights ──────────────────────────────────────────────────────────────────
"$PY" - <<'PYEOF' || exit 1
import pathlib, sys
root = pathlib.Path.cwd()
found = sorted(root.glob("outputs/*/weights/**/best.pt")) + sorted(root.glob("weights/**/best.pt"))
if not found:
    sys.exit("✖ không tìm thấy best.pt trong outputs/ hay weights/. "
             "Đặt TSD_YOLO_WEIGHTS trỏ tới checkpoint rồi chạy lại.")
print("▶ weights     :", found[-1].relative_to(root))
PYEOF

echo "▶ mở          : http://localhost:$PORT   (Ctrl+C để dừng)"
echo
exec "$PY" -m streamlit run app/streamlit_app.py --server.port "$PORT"
