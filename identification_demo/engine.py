"""Adapter for namlhn/vietnam-cccd-ocr; model imports stay out of the HTTP layer."""
import importlib.util
import os
import sys
from pathlib import Path
from threading import Lock


class OCRUnavailable(RuntimeError):
    pass


class OCRBusy(RuntimeError):
    pass


class NamLhnEngine:
    def __init__(self):
        source = Path(os.getenv("OCR_SOURCE_DIR", ".vendor/vietnam-cccd-ocr")).resolve()
        if (source / "ocr" / "__init__.py").is_file():
            sys.path.insert(0, str(source))
        self.source_available = importlib.util.find_spec("ocr") is not None
        self.device = os.getenv("OCR_DEVICE", "cpu")
        self.model = os.getenv("OCR_MODEL", "vgg_transformer")
        self._extractor = None
        self._lock = Lock()

    def describe(self):
        return {
            "provider": "namlhn/vietnam-cccd-ocr",
            "source_available": self.source_available,
            "model_loaded": self._extractor is not None,
            "device": self.device,
            "model": self.model,
        }

    def extract(self, image_path):
        # A single model instance is reused. Concurrent requests must not mutate it.
        if not self._lock.acquire(blocking=False):
            raise OCRBusy("Dịch vụ đang đọc một ảnh khác. Hãy thử lại sau ít giây.")
        try:
            if self._extractor is None:
                try:
                    from ocr import IDCardExtractor
                    self._extractor = IDCardExtractor(
                        device=self.device,
                        model_name=self.model,
                        weights_path=os.getenv("OCR_WEIGHTS") or None,
                        paddle_lang="vi",
                        max_workers=1,
                    )
                except Exception as exc:
                    raise OCRUnavailable(
                        "Chưa khởi tạo được OCR. Kiểm tra mã nguồn, thư viện và "
                        "kết nối tải model theo README của dịch vụ."
                    ) from exc
            return self._extractor.extract(str(image_path), verbose=False)
        finally:
            self._lock.release()
