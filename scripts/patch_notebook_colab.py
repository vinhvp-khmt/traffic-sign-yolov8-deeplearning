"""Patch the notebook's dataset cells for fast, observable Colab runs.

- Put the working dataset on Colab LOCAL disk (/content) instead of Drive (Drive is very
  slow for the ~10k small files in this dataset and shows no progress).
- Download the Kaggle zip, then unzip it ourselves WITH a live progress counter so the long
  extraction step is observable instead of looking frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "traffic_sign_detection_pipeline.ipynb"

# Cell 5 (repo clone + env): move DATA_DIR to fast local disk.
CELL5 = """import os, sys, subprocess

GITHUB_REPO = 'https://github.com/huynhphtloi/traffic-sign-detection-yolo-detr'
DRIVE_ROOT  = '/content/drive/MyDrive/MasterProjects/DeepLearning/traffic-sign-detection-yolo-detr'
REPO        = '/content/traffic-sign-detection-yolo-detr'
# Dataset lives on Colab LOCAL disk for speed (Drive is very slow for ~10k small files).
DATA_DIR    = '/content/dataset'

if not os.path.exists(REPO):
    result = subprocess.run(['git', 'clone', GITHUB_REPO, REPO])
    if result.returncode != 0:
        raise RuntimeError('git clone failed - check the repo URL or network access')
else:
    subprocess.run(['git', '-C', REPO, 'pull'], check=True)

if not os.path.exists(f'{REPO}/src'):
    raise RuntimeError(f'Clone succeeded but src/ not found in {REPO}')

os.environ['TSD_DRIVE_ROOT'] = DRIVE_ROOT
os.environ['TSD_REPO_ROOT']  = REPO
os.environ['TSD_DATA_DIR']   = DATA_DIR
if REPO not in sys.path:
    sys.path.insert(0, REPO)

print('drive root:', DRIVE_ROOT)
print('repo root :', REPO)
print('data dir  :', DATA_DIR, '(local disk - fast)')
print('src found :', os.path.exists(f'{REPO}/src'))
"""

# Cell 11 (download + extract): download zip to local disk, then unzip WITH progress.
CELL11 = """import os, pathlib, zipfile, time
from google.colab import files

DATA_DIR   = os.environ['TSD_DATA_DIR']
DRIVE_ROOT = os.environ['TSD_DRIVE_ROOT']
raw = pathlib.Path(DATA_DIR) / 'raw'
raw.mkdir(parents=True, exist_ok=True)

if DATASET_SOURCE == 'kaggle':
    print('A file picker will appear - select your kaggle.json')
    uploaded = files.upload()
    if 'kaggle.json' not in uploaded:
        raise ValueError('You must upload a file named kaggle.json')
    os.makedirs('/root/.kaggle', exist_ok=True)
    with open('/root/.kaggle/kaggle.json', 'wb') as f:
        f.write(uploaded['kaggle.json'])
    os.chmod('/root/.kaggle/kaggle.json', 0o600)
    print('kaggle.json installed')

    # 1) Download the zip (kaggle shows a % bar) WITHOUT --unzip
    print('\\n[1/2] Downloading dataset zip...')
    !kaggle datasets download -d pkdarabi/cardetection -p {DATA_DIR}/raw

    # 2) Unzip OURSELVES with a live progress counter (the slow step - now observable)
    zips = list(raw.glob('*.zip'))
    if not zips:
        raise FileNotFoundError('No .zip found after download')
    zpath = zips[0]
    print(f'\\n[2/2] Extracting {zpath.name} -> {raw} (local disk)')
    with zipfile.ZipFile(zpath) as zf:
        members = zf.namelist()
        total = len(members)
        t0 = time.time()
        for i, m in enumerate(members, 1):
            zf.extract(m, raw)
            if i % 500 == 0 or i == total:
                pct = 100 * i / total
                rate = i / max(time.time() - t0, 1e-6)
                eta = (total - i) / rate if rate else 0
                print(f'  {pct:5.1f}%  ({i}/{total} files)  ~{eta:4.0f}s left', flush=True)
    print(f'Extracted {total} files in {time.time()-t0:.0f}s')

elif DATASET_SOURCE == 'drive':
    src = pathlib.Path(DRIVE_DATASET_PATH)
    if not src.exists():
        raise FileNotFoundError(
            f'Drive dataset not found: {DRIVE_DATASET_PATH}\\n'
            'Update DRIVE_DATASET_PATH in cell 0.4 to the folder containing data.yaml.'
        )
    link = raw / src.name
    if not link.exists():
        link.symlink_to(src.resolve())
        print(f'Linked  {link} -> {src.resolve()}')
    else:
        print(f'Already linked: {link}')

print()
!ls {DATA_DIR}/raw
"""

PATCHES = {
    "GITHUB_REPO = 'https://github.com/huynhphtloi": CELL5,
    "!kaggle datasets download -d pkdarabi/cardetection -p {DATA_DIR}/raw --unzip": CELL11,
}


def main() -> None:
    nb = json.loads(NB.read_text())
    patched = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for marker, new in PATCHES.items():
            if marker in src:
                cell["source"] = new.splitlines(keepends=True)
                cell["outputs"] = []
                cell["execution_count"] = None
                patched += 1
                break
    NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"patched {patched} cells")


if __name__ == "__main__":
    main()
