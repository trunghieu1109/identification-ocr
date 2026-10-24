"""Fixed public restoration checkpoints; no request-time downloads."""
import os
from pathlib import Path


RESTORATION_MODES = {
    "restormer": ("restormer",),
    "swinir": ("swinir",),
    "both": ("restormer", "swinir"),
}
MODEL_FILES = {"restormer": "restormer-defocus.pth", "swinir": "swinir-small-x4.pth"}
MODEL_LABELS = {"restormer": "Restormer · single-image defocus", "swinir": "SwinIR-S · x4"}
MAX_RESTORATION_SIDE = 384


def restoration_directory():
    return Path(os.getenv("QR_RESTORATION_MODEL_DIR", "/opt/qr-restoration"))
