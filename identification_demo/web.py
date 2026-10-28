"""Multipart API and same-origin demo UI. Uploaded photos are temporary."""
import logging
import asyncio
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from uuid import uuid4
from typing import Literal
from time import perf_counter

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .engine import NamLhnEngine, OCRBusy, OCRUnavailable
from .documents import FRONT_FIELDS
from .qr import QRBusy, QRUnavailable
from .wechat_qr import WeChatQRReader
from .processing import MAX_IMAGE_BYTES, InvalidImage, process_image, process_qr_image
from .pipeline import process_card
from .namlhn_gpu_engine import NamLhnGPUEngine
from .request_timing import RequestTimingMiddleware, request_timings
from .experiment_api import ExperimentClient, register_experiment_routes
from .gpu_parallel import ParallelUnavailable, process_parallel_card

FIELDS = FRONT_FIELDS
logger = logging.getLogger(__name__)


def create_app(engine=None, qr_reader=None, gpu_engine=None, namlhn_gpu_engine=None, experiment_backend=None):
    default_quality = os.getenv("OCR_DEFAULT_QUALITY", "medium")
    if default_quality not in {"medium", "high"}:
        raise ValueError("OCR_DEFAULT_QUALITY must be medium or high")
    mrz_option = os.getenv("GPU_MRZ_ENABLED", "0").strip().lower()
    if mrz_option not in {"0", "1", "false", "true", "off", "on", "no", "yes"}:
        raise ValueError("GPU_MRZ_ENABLED must be 0 or 1")
    default_mrz = mrz_option in {"1", "true", "on", "yes"}

    @asynccontextmanager
    async def lifespan(app):
        async def warm_cpu():
            try:
                if hasattr(app.state.ocr_engine, "initialize"):
                    await run_in_threadpool(app.state.ocr_engine.initialize)
            except Exception:
                logger.warning("CPU warmup failed; initialization will be retried on request")
        task = asyncio.create_task(warm_cpu()) if default_quality == "medium" else None
        yield
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        backends = (app.state.ocr_engine, app.state.gpu_engine, app.state.experiment_backend)
        for backend in {id(backend): backend for backend in backends}.values():
            if hasattr(backend, "close"):
                backend.close()

    app = FastAPI(title="Identification demo — OCR CCCD", version="0.9.0", lifespan=lifespan)
    app.state.ocr_engine = engine if engine is not None else NamLhnEngine()
    app.state.qr_reader = qr_reader if qr_reader is not None else WeChatQRReader()
    app.state.gpu_engine = (gpu_engine if gpu_engine is not None else
                            namlhn_gpu_engine if namlhn_gpu_engine is not None else NamLhnGPUEngine())
    app.state.experiment_backend = experiment_backend if experiment_backend is not None else ExperimentClient()
    register_experiment_routes(app, app.state.experiment_backend)
    static = Path(__file__).parent / "static"
    app.mount("/assets", StaticFiles(directory=static), name="assets")

    @app.middleware("http")
    async def no_store(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path == "/" or request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "engine": app.state.ocr_engine.describe(),
            "qr": app.state.qr_reader.describe(),
            "ocr_profiles": {
                "medium": app.state.ocr_engine.describe(),
                "high": app.state.gpu_engine.describe(),
            },
            "default_quality": default_quality,
            "gpu_mrz": {"default_enabled": default_mrz,
                        **({"worker": app.state.experiment_backend.describe()} if default_mrz else {})},
            "max_image_bytes": MAX_IMAGE_BYTES,
            "supported_formats": ["image/jpeg", "image/png"],
            "supported_sides": ["front", "back"],
        }

    @app.post("/api/cccd/extract", summary="Đọc hai mặt CCCD, ưu tiên QR và bổ sung bằng OCR")
    async def extract_card(request: Request, front_image: UploadFile = File(...), back_image: UploadFile = File(...),
                           quality: Literal["medium", "high"] = Form(default_quality),
                           include_back_ocr: bool = Form(False), use_mrz: bool | None = Form(None)):
        request_id = uuid4().hex
        try:
            images = {}
            for side, upload in (("front", front_image), ("back", back_image)):
                label = "Mặt trước" if side == "front" else "Mặt sau"
                data = await upload.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    return failure(f"{label}: mỗi ảnh tối đa 10 MB.", 413, request_id)
                if not data:
                    return failure(f"{label}: ảnh đang trống. Hãy chọn lại ảnh CCCD.", 422, request_id)
                images[side] = data
            selected_engine = app.state.gpu_engine if quality == "high" else app.state.ocr_engine
            dispatched = perf_counter()
            worker_started = worker_finished = dispatched

            def run_pipeline():
                nonlocal worker_started, worker_finished
                worker_started = perf_counter()
                try:
                    parallel = quality == "high" and (default_mrz if use_mrz is None else use_mrz)
                    if parallel:
                        try:
                            return process_parallel_card(images, app.state.experiment_backend, include_back_ocr)
                        except ParallelUnavailable:
                            logger.warning("Parallel GPU worker unavailable; using classic GPU flow")
                    result = process_card(images, selected_engine, app.state.qr_reader, quality,
                                          include_back_ocr=include_back_ocr)
                    if quality == "high":
                        result["gpu_pipeline"] = {"requested": "parallel" if parallel else "classic", "used": "classic", "fallback": parallel}
                        if parallel:
                            result["warnings"].append("Bộ xử lý MRZ chưa sẵn sàng; đã dùng luồng GPU QR + OCR trước đó.")
                    return result
                finally:
                    worker_finished = perf_counter()

            result = await run_in_threadpool(run_pipeline)
            result["request_id"] = request_id
            result["request_timings_ms"] = request_timings(
                request.scope.get("state", {}).get("request_timing", {}),
                dispatched, worker_started, worker_finished,
            )
            return result
        except InvalidImage as exc:
            return failure(str(exc), 422, request_id)
        except OCRBusy as exc:
            return failure(str(exc), 429, request_id, {"Retry-After": "5"})
        except OCRUnavailable as exc:
            return failure(str(exc), 503, request_id)
        except Exception:
            logger.exception("Combined CCCD extraction failed; request_id=%s", request_id)
            return failure("Chưa xử lý được hai ảnh CCCD. Hãy thử lại hoặc kiểm tra log dịch vụ.", 503, request_id)
        finally:
            await front_image.close()
            await back_image.close()


    @app.post("/api/ocr/cccd")
    async def extract(file: UploadFile = File(...), side: Literal["auto", "front", "back"] = Form("auto")):
        request_id = uuid4().hex
        try:
            data = await file.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                return failure("Mỗi ảnh tối đa 10 MB.", 413, request_id)
            if not data:
                return failure("Ảnh đang trống. Hãy chọn lại ảnh CCCD.", 422, request_id)
            result = await run_in_threadpool(process_image, data, app.state.ocr_engine, side, include_expiry=True)
            result["request_id"] = request_id
            return result
        except InvalidImage as exc:
            return failure(str(exc), 422, request_id)
        except OCRBusy as exc:
            return failure(str(exc), 429, request_id, {"Retry-After": "5"})
        except OCRUnavailable as exc:
            logger.exception("OCR initialization failed; request_id=%s", request_id)
            return failure(str(exc), 503, request_id)
        except Exception:
            logger.exception("OCR extraction failed; request_id=%s", request_id)
            return failure("Chưa đọc được ảnh do lỗi OCR. Hãy thử lại hoặc kiểm tra log dịch vụ.",
                           503, request_id)
        finally:
            await file.close()

    @app.post("/api/qr/cccd")
    async def extract_qr(file: UploadFile = File(...)):
        request_id = uuid4().hex
        try:
            data = await file.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                return failure("Mỗi ảnh tối đa 10 MB.", 413, request_id)
            if not data:
                return failure("Ảnh đang trống. Hãy chọn lại ảnh căn cước.", 422, request_id)
            result = await run_in_threadpool(process_qr_image, data, app.state.qr_reader)
            result["request_id"] = request_id
            return result
        except InvalidImage as exc:
            return failure(str(exc), 422, request_id)
        except QRBusy as exc:
            return failure(str(exc), 429, request_id, {"Retry-After": "5"})
        except QRUnavailable as exc:
            return failure(str(exc), 503, request_id)
        except Exception:
            logger.exception("QR extraction failed; request_id=%s", request_id)
            return failure("Chưa đọc được mã QR do lỗi xử lý. Hãy thử lại hoặc kiểm tra log dịch vụ.",
                           503, request_id)
        finally:
            await file.close()


    @app.exception_handler(HTTPException)
    async def http_error(_request, exc):
        return failure(str(exc.detail), exc.status_code, uuid4().hex)

    app.add_middleware(RequestTimingMiddleware)
    return app


def failure(message, status, request_id, headers=None):
    return JSONResponse(
        {"success": False, "error": message, "request_id": request_id},
        status_code=status, headers=headers,
    )


app = create_app()
