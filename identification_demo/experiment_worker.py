"""CPU-only DocAligner + FastMRZ worker for main GPU and experiment routes."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .engine import OCRBusy
from .experiment_mrz_reader import MRZReader
from .namlhn_gpu_engine import NamLhnGPUEngine
from .processing import MAX_IMAGE_BYTES, InvalidImage
from .wechat_qr import WeChatQRReader
from .work_queue import ModelQueue


class ExperimentEngine:
    def __init__(self):
        self.models = Path(os.getenv("CCCD_EXPERIMENT_MODEL_DIR", "/opt/cccd-experiment-models"))
        self.queue = ModelQueue("cccd-experiment", capacity=2, wait_timeout=180)
        self.gpu = NamLhnGPUEngine()
        self.qr = WeChatQRReader()
        self.aligner = self.mrz = None

    def _initialize(self):
        if self.aligner is None:
            from .experiment_alignment import DocAligner
            aligner = DocAligner(self.models)
            mrz = MRZReader(self.models)
            self.aligner, self.mrz = aligner, mrz

    def initialize(self):
        self.queue.run(self._initialize)

    def describe(self):
        return {"experimental": True, "available": self.aligner is not None and self.mrz is not None,
                "alignment": "DocAligner/lcnet050", "mrz": "FastMRZ/2.1.2 + Vietnam TD1 validation",
                "device": "cpu", "ocr": "existing namlhn-gpu", "queue": self.queue.describe(),
                "execution": "parallel_qr_mrz_ocr", "mrz_attempt_limit": 3, "mrz_retry_budget_ms": 900}

    def extract(self, images, **options):
        def run():
            from .experiment_pipeline import process_experiment
            self._initialize()
            return process_experiment(images, self.gpu, self.qr, self.aligner, self.mrz, **options)
        result, timings = self.queue.run(run)
        result["worker_queue_ms"] = timings["queue_wait"]
        return result

    def close(self):
        self.queue.close()
        self.gpu.close()


def create_app(engine=None):
    backend = engine if engine is not None else ExperimentEngine()

    @asynccontextmanager
    async def lifespan(_app):
        await run_in_threadpool(backend.initialize)
        yield
        backend.close()

    app = FastAPI(title="CCCD experiment · DocAligner + FastMRZ", lifespan=lifespan)

    def authorized(authorization):
        import secrets
        key = os.getenv("CCCD_EXPERIMENT_API_KEY", "")
        return not key or secrets.compare_digest(authorization or "", f"Bearer {key}")

    @app.get("/health")
    def health(authorization: str | None = Header(None)):
        if not authorized(authorization):
            return JSONResponse({"success": False, "error": "Unauthorized"}, status_code=401)
        return backend.describe()

    @app.post("/extract")
    async def extract(front_image: UploadFile = File(...), back_image: UploadFile = File(...),
                      align: bool = Form(True), front_rotation: int = Form(0), back_rotation: int = Form(0),
                      diagnostics: bool = Form(True), include_back_ocr: bool = Form(False),
                      authorization: str | None = Header(None)):
        try:
            if not authorized(authorization):
                return JSONResponse({"success": False, "error": "Unauthorized"}, status_code=401)
            if front_rotation not in (0, 90, 180, 270) or back_rotation not in (0, 90, 180, 270):
                raise InvalidImage("Góc xoay phải là 0, 90, 180 hoặc 270 độ.")
            images = {}
            for side, upload in (("front", front_image), ("back", back_image)):
                data = await upload.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    return JSONResponse({"success": False, "error": "Mỗi ảnh tối đa 10 MB."}, status_code=413)
                if not data:
                    raise InvalidImage("Hãy chọn đủ ảnh mặt trước và mặt sau.")
                images[side] = data
            result = await run_in_threadpool(backend.extract, images, align=align,
                                            rotations={"front": front_rotation, "back": back_rotation})
            if not diagnostics:
                from .gpu_parallel import main_result
                result = {**main_result(result, include_back_ocr), "experimental": True}
            result["request_id"] = uuid4().hex
            return result
        except InvalidImage as exc:
            return JSONResponse({"success": False, "error": str(exc)}, status_code=422)
        except OCRBusy as exc:
            return JSONResponse({"success": False, "error": str(exc)}, status_code=429, headers={"Retry-After": "5"})
        except Exception:
            return JSONResponse({"success": False, "error": "Bộ xử lý thử nghiệm gặp lỗi."}, status_code=503)
        finally:
            await front_image.close()
            await back_image.close()

    @app.middleware("http")
    async def no_store(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


app = create_app()
