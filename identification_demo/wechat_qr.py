"""WeChat QR decoding shared by the main API and the comparison tool."""
import importlib
import importlib.util
import os
from pathlib import Path
from threading import Lock

from .qr import QRBusy, QRUnavailable


WECHAT_FILES = ("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")


def model_directory():
    return Path(os.getenv("QR_MODEL_DIR", str(Path(__file__).resolve().parents[1] / ".vendor" / "qr-models")))


def create_wechat_decoder():
    try:
        cv2 = importlib.import_module("cv2")
        importlib.import_module("numpy")
    except ImportError as exc:
        raise QRUnavailable("Chưa cài bộ đọc WeChatQRCode. Hãy cài requirements-qr.txt hoặc build lại Docker.") from exc
    if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
        raise QRUnavailable("OpenCV chưa có WeChatQRCode. Hãy cài opencv-contrib-python theo requirements-qr.txt.")
    directory = model_directory()
    if not all((directory / name).is_file() for name in WECHAT_FILES):
        raise QRUnavailable("Thiếu model WeChatQRCode. Hãy chạy scripts/setup-qr-models.py hoặc build lại Docker.")
    return cv2.wechat_qrcode_WeChatQRCode(*(str(directory / name) for name in WECHAT_FILES))


def decode_wechat(decoder, image):
    import cv2
    import numpy as np
    return decoder.detectAndDecode(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))


class WeChatQRReader:
    """Keep the QR reader interface: upright RGB pixels in, unique texts out."""

    def __init__(self):
        self._lock = Lock()
        self._decoder = None

    def describe(self):
        installed = all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy"))
        models = all((model_directory() / name).is_file() for name in WECHAT_FILES)
        return {"provider": "wechat-qrcode", "available": installed and models}

    def read(self, image):
        if not self._lock.acquire(timeout=2):
            raise QRBusy("Dịch vụ đang đọc một mã QR khác. Hãy thử lại sau ít giây.")
        try:
            if self._decoder is None:
                self._decoder = create_wechat_decoder()
            texts, _points = decode_wechat(self._decoder, image)
            return list(dict.fromkeys(texts))
        finally:
            self._lock.release()
