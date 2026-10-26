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
from .qr_experiments import QRExperiments, QRPairExperiments, process_qr_experiment, process_qr_pair_experiment
from .processing import MAX_IMAGE_BYTES, InvalidImage, process_image, process_qr_image
from .pipeline import process_card
from .expiry_experiments import process_expiry_experiment
from .gpu_engine import DocUGPUEngine
from .namlhn_gpu_engine import NamLhnGPUEngine
from .docu_region_engine import DocURegionGPUEngine, DocUExpandedGPUEngine
from .ocr_experiments import ENGINES, process_ocr_experiment
from .request_timing import RequestTimingMiddleware, request_timings

FIELDS = FRONT_FIELDS
logger = logging.getLogger(__name__)


def create_app(engine=None, qr_reader=None, gpu_engine=None, qr_experiments=None, qr_pair_experiments=None,
               namlhn_gpu_engine=None, docu_region_engine=None, docu_expanded_engine=None, docu_gpu_engine=None):
    default_quality = os.getenv("OCR_DEFAULT_QUALITY", "medium")
    if default_quality not in {"medium", "high"}:
        raise ValueError("OCR_DEFAULT_QUALITY must be medium or high")

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
        backends = (app.state.ocr_engine, app.state.gpu_engine, app.state.namlhn_gpu_engine,
                    app.state.docu_gpu_engine, app.state.docu_region_engine, app.state.docu_expanded_engine)
        for backend in {id(backend): backend for backend in backends}.values():
            if hasattr(backend, "close"):
                backend.close()

    app = FastAPI(title="Identification demo — OCR CCCD", version="0.7.0", lifespan=lifespan)
    app.state.ocr_engine = engine if engine is not None else NamLhnEngine()
    app.state.qr_reader = qr_reader if qr_reader is not None else WeChatQRReader()
    app.state.namlhn_gpu_engine = namlhn_gpu_engine if namlhn_gpu_engine is not None else NamLhnGPUEngine()
    app.state.gpu_engine = gpu_engine if gpu_engine is not None else app.state.namlhn_gpu_engine
    app.state.docu_gpu_engine = docu_gpu_engine if docu_gpu_engine is not None else DocUGPUEngine()
    app.state.docu_region_engine = docu_region_engine if docu_region_engine is not None else DocURegionGPUEngine()
    app.state.docu_expanded_engine = docu_expanded_engine if docu_expanded_engine is not None else DocUExpandedGPUEngine()
    app.state.qr_experiments = qr_experiments if qr_experiments is not None else QRExperiments()
    app.state.qr_pair_experiments = qr_pair_experiments if qr_pair_experiments is not None else QRPairExperiments()
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
            "qr_experiments": app.state.qr_experiments.describe(),
            "ocr_profiles": {
                "medium": app.state.ocr_engine.describe(),
                "high": app.state.gpu_engine.describe(),
            },
            "default_quality": default_quality,
            "max_image_bytes": MAX_IMAGE_BYTES,
            "supported_formats": ["image/jpeg", "image/png"],
            "supported_sides": ["front", "back"],
        }

    @app.post("/api/cccd/extract", summary="Đọc hai mặt CCCD, ưu tiên QR và bổ sung bằng OCR")
    async def extract_card(request: Request, front_image: UploadFile = File(...), back_image: UploadFile = File(...),
                           quality: Literal["medium", "high"] = Form(default_quality),
                           include_back_ocr: bool = Form(False)):
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
                    return process_card(images, selected_engine, app.state.qr_reader, quality,
                                        include_back_ocr=include_back_ocr)
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

    @app.get("/api/cccd/experiment/health")
    def experiment_health():
        backends = {"namlhn_cpu": app.state.ocr_engine, "namlhn_gpu": app.state.namlhn_gpu_engine,
                    "docu_gpu": app.state.docu_gpu_engine, "docu_gpu_regions": app.state.docu_region_engine,
                    "docu_gpu_v3": app.state.docu_expanded_engine}
        return {"engines": {name: {**ENGINES[name], **backend.describe()} for name, backend in backends.items()}}

    @app.post("/api/cccd/experiment/extract", summary="Thử OCR đầy đủ với namlhn CPU/GPU hoặc DocU GPU")
    async def experiment_card(request: Request, front_image: UploadFile = File(...), back_image: UploadFile = File(...),
                              engine_name: Literal["namlhn_cpu", "namlhn_gpu", "docu_gpu", "docu_gpu_regions", "docu_gpu_v3"] = Form("docu_gpu", alias="engine"),
                              include_ocr_text: bool = Form(False), use_qr: bool = Form(True)):
        request_id = uuid4().hex
        try:
            images = {}
            for side, upload in (("front", front_image), ("back", back_image)):
                label = "Mặt trước" if side == "front" else "Mặt sau"
                data = await upload.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    return failure(f"{label}: mỗi ảnh tối đa 10 MB.", 413, request_id)
                if not data:
                    return failure(f"{label}: ảnh đang trống. Hãy chọn lại ảnh căn cước.", 422, request_id)
                images[side] = data
            selected = {"namlhn_cpu": app.state.ocr_engine, "namlhn_gpu": app.state.namlhn_gpu_engine,
                        "docu_gpu": app.state.docu_gpu_engine, "docu_gpu_regions": app.state.docu_region_engine,
                        "docu_gpu_v3": app.state.docu_expanded_engine}[engine_name]
            dispatched = perf_counter()
            worker_started = worker_finished = dispatched

            def run():
                nonlocal worker_started, worker_finished
                worker_started = perf_counter()
                try:
                    return process_ocr_experiment(images, selected, app.state.qr_reader, engine_name,
                                                  include_ocr_text, use_qr)
                finally:
                    worker_finished = perf_counter()

            result = await run_in_threadpool(run)
            result["request_id"] = request_id
            result["request_timings_ms"] = request_timings(
                request.scope.get("state", {}).get("request_timing", {}),
                dispatched, worker_started, worker_finished)
            return result
        except InvalidImage as exc:
            return failure(str(exc), 422, request_id)
        except OCRBusy as exc:
            return failure(str(exc), 429, request_id, {"Retry-After": "5"})
        except OCRUnavailable as exc:
            return failure(str(exc), 503, request_id)
        except Exception:
            logger.exception("Full OCR experiment failed; engine=%s request_id=%s", engine_name, request_id)
            return failure("Chưa xử lý được lần thử OCR này. Hãy thử lại hoặc kiểm tra bộ xử lý đã chọn.", 503, request_id)
        finally:
            await front_image.close()
            await back_image.close()

    @app.post("/api/cccd/expiry/experiment", summary="Thử đọc ngày hết hạn từ cả hai mặt căn cước")
    async def experiment_expiry(front_image: UploadFile = File(...), back_image: UploadFile = File(...),
                                quality: Literal["medium", "high"] = Form(default_quality)):
        request_id = uuid4().hex
        try:
            images = {}
            for side, upload in (("front", front_image), ("back", back_image)):
                label = "Mặt trước" if side == "front" else "Mặt sau"
                data = await upload.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    return failure(f"{label}: mỗi ảnh tối đa 10 MB.", 413, request_id)
                if not data:
                    return failure(f"{label}: ảnh đang trống. Hãy chọn lại ảnh căn cước.", 422, request_id)
                images[side] = data
            selected_engine = app.state.gpu_engine if quality == "high" else app.state.ocr_engine
            result = await run_in_threadpool(process_expiry_experiment, images, selected_engine, quality)
            result["request_id"] = request_id
            return result
        except InvalidImage as exc:
            return failure(str(exc), 422, request_id)
        except OCRBusy as exc:
            return failure(str(exc), 429, request_id, {"Retry-After": "5"})
        except OCRUnavailable as exc:
            return failure(str(exc), 503, request_id)
        except Exception:
            logger.exception("Expiry experiment failed; request_id=%s", request_id)
            return failure("Chưa chạy được thử nghiệm ngày hết hạn. Hãy thử lại hoặc kiểm tra dịch vụ.", 503, request_id)
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

    @app.post("/api/qr/experiment", summary="Thử các bộ đọc QR và phục hồi ảnh bằng Restormer / SwinIR")
    async def experiment_qr(file: UploadFile = File(...),
                            method: Literal["zxing", "wechat", "qrdet", "wechat_parallel", "wechat_restormer", "wechat_swinir", "wechat_both"] = Form(...)):
        request_id = uuid4().hex
        try:
            data = await file.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                return failure("Mỗi ảnh tối đa 10 MB.", 413, request_id)
            if not data:
                return failure("Ảnh đang trống. Hãy chọn lại ảnh QR.", 422, request_id)
            result = await run_in_threadpool(process_qr_experiment, data, app.state.qr_experiments, method)
            result["request_id"] = request_id
            return result
        except InvalidImage as exc:
            return failure(str(exc), 422, request_id)
        except QRBusy as exc:
            return failure(str(exc), 429, request_id, {"Retry-After": "5"})
        except QRUnavailable as exc:
            return failure(str(exc), 503, request_id)
        except Exception:
            logger.exception("QR experiment failed; method=%s request_id=%s", method, request_id)
            return failure("Chưa chạy được phương án QR này. Hãy kiểm tra thư viện, model hoặc thử lại.", 503, request_id)
        finally:
            await file.close()

    @app.post("/api/qr/experiment/pair", summary="Thử QR hai mặt, chạy song song hoặc tuần tự")
    async def experiment_qr_pair(front_image: UploadFile = File(...), back_image: UploadFile = File(...),
                                 method: Literal["zxing", "wechat", "qrdet", "wechat_parallel", "wechat_restormer", "wechat_swinir", "wechat_both"] = Form(...),
                                 execution: Literal["parallel", "sequential"] = Form("parallel")):
        request_id = uuid4().hex
        try:
            images = {}
            for side, upload in (("front", front_image), ("back", back_image)):
                label = "Mặt trước" if side == "front" else "Mặt sau"
                data = await upload.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    return failure(f"{label}: mỗi ảnh tối đa 10 MB.", 413, request_id)
                if not data:
                    return failure(f"{label}: ảnh đang trống. Hãy chọn lại ảnh QR.", 422, request_id)
                images[side] = data
            result = await run_in_threadpool(process_qr_pair_experiment, images, app.state.qr_pair_experiments,
                                            method, execution)
            result["request_id"] = request_id
            return result
        except InvalidImage as exc:
            return failure(str(exc), 422, request_id)
        except QRBusy as exc:
            return failure(str(exc), 429, request_id, {"Retry-After": "5"})
        except QRUnavailable as exc:
            return failure(str(exc), 503, request_id)
        except Exception:
            logger.exception("QR pair experiment failed; method=%s request_id=%s", method, request_id)
            return failure("Chưa chạy được thử nghiệm QR hai mặt. Hãy kiểm tra bộ đọc hoặc thử lại.", 503, request_id)
        finally:
            await front_image.close()
            await back_image.close()

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
