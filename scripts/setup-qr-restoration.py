#!/usr/bin/env python3
"""Download and verify the authors' restoration checkpoints before serving requests."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


ASSETS = (
    ("restormer-defocus.pth", "https://drive.usercontent.google.com/download?id=10v8BH3Gktl34TYzPy0x-pAKoRSYKnNZp&export=download&authuser=0&confirm=t",
     "7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e"),
    ("swinir-small-x4.pth", "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
     "09fad24e32ae62722e1a055efde9921328f4137981bab0a42a4a3a806306c58e"),
)


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, url, checksum in ASSETS:
        target = directory / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == checksum:
            print(f"Ready: {name}", flush=True)
            continue
        temporary = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "identification-qr-restoration"})
            with urllib.request.urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temporary = Path(output.name)
                while block := response.read(1024 * 1024):
                    output.write(block)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Model checksum mismatch: {name}")
            os.replace(temporary, target)
            print(f"Downloaded: {name}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(os.getenv("QR_RESTORATION_MODEL_DIR", ".vendor/qr-restoration")))
    download(parser.parse_args().directory)
