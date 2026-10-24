"""Multipart API and same-origin demo UI. Uploaded photos are temporary."""
import io
import logging
import os
import tempfile
import warnings
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from typing import Literal

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .engine import NamLhnEngine, OCRBusy, OCRUnavailable
from .documents import FRONT_FIELDS, BACK_FIELDS, detect_side, text_rows, parse_back
from .qr import QR_FIELDS, QRReader, QRBusy, QRUnavailable, parse_cccd_qr

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
FIELDS = FRONT_FIELDS
logger = logging.getLogger(__name__)


class InvalidImage(ValueError):
    pass


def load_image(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as original:
                if original.format not in {"JPEG", "PNG"}:
                    raise InvalidImage("Chỉ hỗ trợ ảnh JPG hoặc PNG.")
                if original.width * original.height > MAX_IMAGE_PIXELS:
                    raise InvalidImage("Ảnh quá lớn. Hãy dùng ảnh dưới 20 megapixel.")
                # Both readers receive upright pixels, matching the preview.
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise InvalidImage("Không đọc được ảnh. Hãy chọn một ảnh JPG hoặc PNG hợp lệ.") from exc
    return image


def process_image(data, engine, side="auto"):
    started = perf_counter()

    # Never use the original filename as a filesystem path. The entire directory
    # is removed on success and failure, including any upstream sidecar outputs.
    with load_image(data) as image, tempfile.TemporaryDirectory(prefix="cccd-ocr-", dir=os.getenv("OCR_UPLOAD_TMP_DIR")) as directory:
        image_path = Path(directory) / "card.png"
        image.save(image_path)
        raw = engine.extract(image_path)

    if not isinstance(raw, dict):
        raise TypeError("OCR provider returned an invalid result")
    rows, backup_rows = text_rows(raw), text_rows(raw, "paddle_text_lines")
    detected = detect_side(rows + backup_rows)
    if side != "auto" and detected != "unknown" and side != detected:
        actual = "mặt sau" if detected == "back" else "mặt trước"
        raise InvalidImage(f"Ảnh có dấu hiệu là {actual}. Hãy chọn đúng mặt thẻ hoặc Tự nhận diện.")
    side = detected if side == "auto" and detected != "unknown" else "front" if side == "auto" else side
    schema = BACK_FIELDS if side == "back" else FRONT_FIELDS
    if side == "back":
        fields = parse_back(rows, backup_rows)
    else:
        fields = {}
        for key, _label, upstream in schema:
            value = raw.get(upstream)
            if value is not None and not isinstance(value, (str, int, float)):
                raise TypeError("OCR field must be text")
            fields[key] = str(value).strip() if value is not None else ""
    raw_text = "\n".join(rows or backup_rows)
    count = sum(bool(value) for value in fields.values())
    return {
        "success": True,
        "status": "complete" if count == len(schema) else "partial" if count else "unstructured" if raw_text else "no_text",
        "side": side,
        "detected_side": detected,
        "fields": fields,
        "raw_text": raw_text,
        "recognized_fields": count,
        "total_fields": len(schema),
        "processing_time_ms": round((perf_counter() - started) * 1000),
    }


def process_qr_image(data, reader):
    started = perf_counter()
    with load_image(data) as image:
        texts = reader.read(image)
    fields = {key: "" for key, _ in QR_FIELDS}
    if not texts:
        status = "no_qr"
        message = "Chưa đọc được mã QR. Hãy chụp rõ toàn bộ mã, tránh lóa hoặc thử ảnh cắt riêng vùng QR."
    elif len(texts) > 1:
        status = "multiple_qr"
        message = "Ảnh có nhiều mã QR khác nhau. Hãy cắt ảnh chỉ còn mã QR của một căn cước rồi thử lại."
    else:
        parsed = parse_cccd_qr(texts[0])
        if parsed is None:
            status = "unsupported_qr"
            message = "Mã QR chưa có cấu trúc CCCD được hỗ trợ. Hãy thử mã QR trên thẻ CCCD."
        else:
            fields = parsed
            required = [value for key, value in fields.items() if key != "old_id_number"]
            status = "complete" if all(required) else "partial"
            message = "Đã đọc thông tin từ mã QR. Số CMND cũ có thể để trống; hãy đối chiếu với thẻ gốc."
            if status == "partial":
                message = "Một số thông tin trong mã QR bị thiếu hoặc không hợp lệ. Hãy đối chiếu với thẻ gốc."
    return {
        "success": True,
        "source": "qr",
        "status": status,
        "fields": fields,
        "message": message,
        "recognized_fields": sum(bool(value) for value in fields.values()),
        "total_fields": len(QR_FIELDS),
        "processing_time_ms": round((perf_counter() - started) * 1000),
    }


def create_app(engine=None, qr_reader=None):
    app = FastAPI(title="Identification demo — OCR CCCD", version="0.2.1")
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
