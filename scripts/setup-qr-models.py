#!/usr/bin/env python3
"""Download public QR models, never user images. Safe to rerun."""
import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from identification_demo.wechat_qr import model_directory


WECHAT_URL = "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode"
MODELS = (
    ("detect.prototxt", f"{WECHAT_URL}/detect.prototxt", "6fb4976b32695f9f5c6305c19f12537d"),
    ("detect.caffemodel", f"{WECHAT_URL}/detect.caffemodel", "238e2b2d6f3c18d6c3a30de0c31e23cf"),
    ("sr.prototxt", f"{WECHAT_URL}/sr.prototxt", "69db99927a70df953b471daaba03fbef"),
    ("sr.caffemodel", f"{WECHAT_URL}/sr.caffemodel", "cbfcd60361a73beb8c583eea7e8e6664"),

)


def matches_checksum(path, expected):
    algorithm = "sha256" if len(expected) == 64 else "md5"
    return hashlib.new(algorithm, path.read_bytes()).hexdigest() == expected


def download_models(directory):
    for name, url, expected_hash in MODELS:
        target = directory / name
        if target.is_file() and target.stat().st_size and matches_checksum(target, expected_hash):
            print(f"Ready: {name}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "identification-qr-setup"})
            with urllib.request.urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
                    dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                while block := response.read(1024 * 1024):
                    output.write(block)
                expected_size = response.headers.get("Content-Length")
            if not temporary.stat().st_size or (expected_size and temporary.stat().st_size != int(expected_size)):
                raise ValueError(f"Incomplete download: {name}")
            if not matches_checksum(temporary, expected_hash):
                raise ValueError(f"Checksum mismatch: {name}")
            os.replace(temporary, target)
            print(f"Downloaded: {name}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=model_directory())
    args = parser.parse_args()
    download_models(args.directory)
