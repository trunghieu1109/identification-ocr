"""Multipart API and same-origin demo UI. Uploaded photos are temporary."""
import logging
from pathlib import Path
from uuid import uuid4
from typing import Literal

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .engine import NamLhnEngine, OCRBusy, OCRUnavailable
from .documents import FRONT_FIELDS
from .qr import QRReader, QRBusy, QRUnavailable
from .processing import MAX_IMAGE_BYTES, InvalidImage, process_image, process_qr_image
from .pipeline import process_card

FIELDS = FRONT_FIELDS
logger = logging.getLogger(__name__)


def create_app(engine=None, qr_reader=None):
    app = FastAPI(title="Identification demo — OCR CCCD", version="0.3.0")
    app.state.ocr_engine = engine if engine is not None else NamLhnEngine()
    app.state.qr_reader = qr_reader if qr_reader is not None else QRReader()
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
            "max_image_bytes": MAX_IMAGE_BYTES,
            "supported_formats": ["image/jpeg", "image/png"],
            "supported_sides": ["front", "back"],
        }

    @app.post("/api/cccd/extract", summary="Đọc hai mặt CCCD, ưu tiên QR và bổ sung bằng OCR")
    async def extract_card(front_image: UploadFile = File(...), back_image: UploadFile = File(...)):
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
            result = await run_in_threadpool(process_card, images, app.state.ocr_engine, app.state.qr_reader)
            result["request_id"] = request_id
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
            result = await run_in_threadpool(process_image, data, app.state.ocr_engine, side)
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

    return app


def failure(message, status, request_id, headers=None):
    return JSONResponse(
        {"success": False, "error": message, "request_id": request_id},
        status_code=status, headers=headers,
    )


app = create_app()
