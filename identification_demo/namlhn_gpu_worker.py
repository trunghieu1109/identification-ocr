"""GPU service: Paddle text detection and DocU Seq2Seq recognition."""
import logging
import os
import secrets
import sys
from contextlib import asynccontextmanager, closing
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, File, Header, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .documents import FRONT_FIELDS, BACK_FIELDS
from .engine import NamLhnEngine, OCRBusy, OCRUnavailable, OCRMultipleDocuments
from .processing import MAX_IMAGE_BYTES, InvalidImage, load_image
from .namlhn_docu_pipeline import STRATEGY as DOCU_STRATEGY

logger = logging.getLogger(__name__)


class NamLhnDocUCudaEngine(NamLhnEngine):
    """Load only Paddle detection and one DocU recognizer on the GPU."""
    def __init__(self):
        super().__init__(queue_name="ocr-gpu")
        self.device = "cuda"
        self.model = "vgg_seq2seq"
        self.cuda_available = False
        self.reason = "Đang khởi tạo OCR GPU."
        if os.getenv("NAMLHN_GPU_MODEL", self.model) != self.model:
            raise ValueError("GPU chỉ hỗ trợ model vgg_seq2seq")
        self.text_enhancement = os.getenv("NAMLHN_GPU_TEXT_ENHANCEMENT", "adaptive")
        if self.text_enhancement not in {"off", "adaptive"}:
            raise ValueError("NAMLHN_GPU_TEXT_ENHANCEMENT must be off or adaptive")
        self._detector = None
        self._recognizer = None
        self.card_enhancement = os.getenv("NAMLHN_GPU_CARD_ENHANCEMENT", "adaptive")
        if self.card_enhancement not in {"adaptive", "off"}:
            raise ValueError("NAMLHN_GPU_CARD_ENHANCEMENT must be adaptive/off")

    def _initialize(self):
        if self._detector is not None and self._recognizer is not None:
            return
        import torch
        import paddle
        from paddleocr import TextDetection

        self.cuda_available = bool(torch.cuda.is_available() and paddle.device.is_compiled_with_cuda()
                                   and paddle.device.cuda.device_count() > 0)
        if not self.cuda_available:
            self.reason = "namlhn kết hợp DocU cần CUDA của cả Paddle và PyTorch."
            raise OCRUnavailable(self.reason)
        torch.set_num_threads(max(1, int(os.getenv("OCR_CPU_THREADS", "4"))))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        source = Path(os.getenv("DOCU_SOURCE_DIR", "/opt/docu"))
        weights = Path(os.getenv("OCR_WEIGHTS", f"/app/model/recognition/{self.model}.pth"))
        if not weights.is_file() or not (source / "configs/vietocr" / f"{self.model}.yml").is_file():
            self.reason = "Thiếu checkpoint hoặc cấu hình cục bộ của bộ đọc DocU."
            raise OCRUnavailable(self.reason)
        sys.path.insert(0, str(source))
        from core.recognition.vietocr_recognizer import VietOCRRecognizer

        paddle.set_device("gpu:0")
        detector = TextDetection(model_name="PP-OCRv6_medium_det", device="gpu:0",
                                 limit_side_len=960, thresh=.2, box_thresh=.5, unclip_ratio=2.,
                                 enable_mkldnn=False)
        recognizer = VietOCRRecognizer(config_name=self.model, weights_path=str(weights), device="cuda:0")
        if not str(next(recognizer.predictor.model.parameters()).device).startswith("cuda"):
            raise OCRUnavailable("Bộ đọc DocU chưa chạy trên CUDA.")
        paddle.ones([1]).sum().numpy()
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()
        self._detector, self._recognizer = detector, recognizer
        self.reason = ""

    def _extract_pixels(self, image, side, needed_fields):
        from .namlhn_docu_pipeline import extract_pixels
        started = perf_counter()
        self._initialize()
        initialized_ms = (perf_counter() - started) * 1000
        result = extract_pixels(self._detector, self._recognizer, image, side, self.text_enhancement,
                                card_enhancement=self.card_enhancement)
        result["timings_ms"]["model_init"] = round(initialized_ms, 2)
        return result

    def _extract_path(self, image_path):
        with closing(load_image(Path(image_path).read_bytes())) as image:
            return self._extract_pixels(image, "auto", None)

    def describe(self):
        ready = self._detector is not None and self._recognizer is not None
        return {**super().describe(), "device": "cuda:0", "cuda_available": self.cuda_available,
                "reason": self.reason, "recognition_model": self.model, "model_loaded": ready,
                "pipeline_ready": ready and self.cuda_available, "strategy": DOCU_STRATEGY,
                "detection_model": "PP-OCRv6_medium_det", "recognition_provider": "docu",
                "recognition_device": "cuda:0",
                "text_enhancement": self.text_enhancement,
                "card_crop": "off", "card_enhancement": self.card_enhancement,
                "coordinate_space": "original_image"}


def create_namlhn_gpu_app(engine=None):
    engine = engine if engine is not None else NamLhnDocUCudaEngine()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            await run_in_threadpool(engine.initialize)
        except Exception:
            logger.exception("Namlhn GPU initialization failed")
        yield
        engine.close()

    app = FastAPI(title="Namlhn GPU OCR", lifespan=lifespan)
    token = os.getenv("NAMLHN_GPU_OCR_API_KEY", "")

    def failure(message, status, headers=None):
        return JSONResponse({"success": False, "error": message}, status_code=status,
                            headers={"Cache-Control": "no-store", **(headers or {})})

    @app.get("/api/v1/health/ready")
    def health():
        info = engine.describe()
        return JSONResponse(info, status_code=200 if info["pipeline_ready"] else 503,
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/extract")
    async def extract(file: UploadFile = File(...), doc_type_hint: Literal["cccd_front", "cccd_back"] = "cccd_front",
                      fields: str | None = None, authorization: str | None = Header(None)):
        try:
            if token and not secrets.compare_digest(authorization or "", f"Bearer {token}"):
                return failure("Không có quyền truy cập namlhn GPU.", 401)
            data = await file.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                return failure("Mỗi ảnh tối đa 10 MB.", 413)
            side = doc_type_hint.removeprefix("cccd_")
            allowed = {key for key, *_ in (FRONT_FIELDS if side == "front" else BACK_FIELDS)}
            allowed.add("expiry_date")
            needed = fields.split(",") if fields is not None else None
            if needed is not None and (not needed or not set(needed) <= allowed):
                raise InvalidImage("Danh sách trường OCR không hợp lệ.")

            def run():
                started = perf_counter()
                with closing(load_image(data)) as image:
                    decoded_ms = (perf_counter() - started) * 1000
                    result = engine.extract_image(image, side, needed_fields=needed)
                result.setdefault("timings_ms", {})["worker_image_decode"] = round(decoded_ms, 2)
                return {"success": True, **result}

            return JSONResponse(await run_in_threadpool(run), headers={"Cache-Control": "no-store"})
        except OCRMultipleDocuments as exc:
            return JSONResponse({"success": False, "error": str(exc), "error_code": exc.code,
                                 "detected_regions": exc.detected_regions}, status_code=422,
                                headers={"Cache-Control": "no-store"})
        except InvalidImage as exc:
            return failure(str(exc), 422)
        except OCRBusy as exc:
            return failure(str(exc), 429, {"Retry-After": "5"})
        except OCRUnavailable as exc:
            return failure(str(exc), 503)
        except Exception:
            logger.exception("Namlhn GPU OCR failed")
            return failure("Chưa xử lý được ảnh bằng namlhn GPU.", 503)
        finally:
            await file.close()

    return app


app = create_namlhn_gpu_app()
