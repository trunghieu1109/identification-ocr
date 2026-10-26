"""Adapter for namlhn/vietnam-cccd-ocr; model imports stay out of the HTTP layer."""
import importlib.util
import os
import sys
from pathlib import Path
from threading import Lock
from time import perf_counter


class OCRUnavailable(RuntimeError):
    pass


class OCRBusy(RuntimeError):
    pass


class OCRMultipleDocuments(OCRUnavailable):
    code = "multiple_documents"

    def __init__(self, detected_regions=None):
        self.detected_regions = detected_regions
        count = f"{detected_regions} vùng" if detected_regions else "nhiều vùng"
        super().__init__(
            f"GPU phát hiện {count} nghi là thẻ và chưa chọn được một vùng duy nhất. "
            "Có thể model nhận diện nhầm; hãy cắt ảnh sát một thẻ rồi thử lại."
        )


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
        from .work_queue import ModelQueue
        self._queue = ModelQueue("ocr-cpu")

    def describe(self):
        return {
            "provider": "namlhn/vietnam-cccd-ocr",
            "source_available": self.source_available,
            "model_loaded": self._extractor is not None,
            "device": self.device,
            "model": self.model,
            "queue": self._queue.describe(),
        }

    def _initialize(self):
        if self._extractor is None:
            try:
                import torch
                torch.set_num_threads(max(1, int(os.getenv("OCR_CPU_THREADS", "4"))))
                from ocr import IDCardExtractor
                self._extractor = IDCardExtractor(
                    device=self.device, model_name=self.model,
                    weights_path=os.getenv("OCR_WEIGHTS") or None,
                    paddle_lang="vi", max_workers=1,
                )
            except Exception as exc:
                raise OCRUnavailable(
                    "Chưa khởi tạo được OCR. Kiểm tra mã nguồn, thư viện và "
                    "kết nối tải model theo README của dịch vụ."
                ) from exc

    def initialize(self):
        self._queue.run(self._initialize)

    def close(self):
        self._queue.close()

    def extract(self, image_path):
        result, _timings = self._queue.run(self._extract_path, image_path)
        return result

    def _extract_path(self, image_path):
        # A single model instance is reused. Concurrent requests must not mutate it.
        if not self._lock.acquire(blocking=False):
            raise OCRBusy("Dịch vụ đang đọc một ảnh khác. Hãy thử lại sau ít giây.")
        try:
            self._initialize()
            return self._extractor.extract(str(image_path), verbose=False)
        finally:
            self._lock.release()

    def extract_image(self, image, side, needed_fields=None, source_data=None):
        result, timings = self._queue.run(self._extract_pixels, image, side, needed_fields)
        result.setdefault("timings_ms", {}).update(timings)
        return result

    def extract_expiry_image(self, image, side, source_data=None):
        return self.extract_image(image, side, needed_fields=["expiry_date"], source_data=source_data)

    def _extract_pixels(self, image, side, needed_fields):
        from .cpu_pipeline import extract_pixels
        started = perf_counter()
        self._initialize()
        initialized = (perf_counter() - started) * 1000
        result = extract_pixels(self._extractor, image, side, needed_fields)
        result["timings_ms"]["model_init"] = round(initialized, 2)
        return result
