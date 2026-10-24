#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Cần Python 3.10–3.12 cho PaddleOCR/VietOCR"'
command -v git >/dev/null || { echo "Cần cài Git trước khi chạy setup." >&2; exit 1; }
"$python_bin" -m venv .venv

if [ ! -d .vendor/vietnam-cccd-ocr/.git ]; then
  mkdir -p .vendor
  git clone https://github.com/namlhn/vietnam-cccd-ocr.git .vendor/vietnam-cccd-ocr
fi
upstream_ref="${OCR_REF:-2d951ac603a43677572ff39ed6a21bdebe622263}"
git -C .vendor/vietnam-cccd-ocr checkout "$upstream_ref"
sh scripts/patch-upstream.sh "$project_dir/.vendor/vietnam-cccd-ocr"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r .vendor/vietnam-cccd-ocr/requirements.txt -r requirements.txt -r requirements-cpu.txt -r requirements-qr.txt
# These distributions share cv2; install the contrib variant last for WeChatQRCode.
.venv/bin/python -m pip install --force-reinstall --no-deps opencv-contrib-python==4.10.0.84
.venv/bin/python scripts/setup-qr-models.py
echo "Đã cài xong. Chạy: .venv/bin/python -m uvicorn identification_demo.web:app --host 0.0.0.0 --port 8888"
