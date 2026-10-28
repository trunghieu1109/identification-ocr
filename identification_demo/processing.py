"""Shared upload validation and individual OCR/QR processing."""
import io
import os
import tempfile
import warnings
from contextlib import closing, nullcontext
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps, UnidentifiedImageError

from .documents import FRONT_FIELDS, BACK_FIELDS, detect_side, text_rows, parse_back
from .qr import QR_FIELDS, parse_cccd_qr
from .ocr_debug import back_ocr_debug

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


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


def process_image(data, engine, side="auto", needed_fields=None, source_data=None, include_expiry=False):
    started = perf_counter()

    # Never use the original filename as a filesystem path. The entire directory
    # is removed on success and failure, including any upstream sidecar outputs.
    context = nullcontext(data) if isinstance(data, Image.Image) else closing(load_image(data))
    with context as image:
        extract_image = getattr(engine, "extract_image", None)
        if extract_image:
            raw = extract_image(image, side, needed_fields=needed_fields,
                                source_data=source_data if source_data is not None else data if isinstance(data, bytes) else None)
        else:
            with tempfile.TemporaryDirectory(prefix="cccd-ocr-", dir=os.getenv("OCR_UPLOAD_TMP_DIR")) as directory:
                image_path = Path(directory) / "card.png"
                image.save(image_path)
                extract_document = getattr(engine, "extract_document", None)
                raw = extract_document(image_path, side) if extract_document else engine.extract(image_path)

    if not isinstance(raw, dict):
        raise TypeError("OCR provider returned an invalid result")
    rows, backup_rows = text_rows(raw), text_rows(raw, "paddle_text_lines")
    detected = raw.get("document_side") or detect_side(rows + backup_rows)
    if detected == "unknown":
        detected = detect_side(rows + backup_rows)
    if side != "auto" and detected != "unknown" and side != detected:
        actual = "mặt sau" if detected == "back" else "mặt trước"
        raise InvalidImage(f"Ảnh có dấu hiệu là {actual}. Hãy chọn đúng mặt thẻ hoặc Tự nhận diện.")
    side = detected if side == "auto" and detected != "unknown" else "front" if side == "auto" else side
    schema = BACK_FIELDS if side == "back" else FRONT_FIELDS
    if "structured_fields" in raw:
        structured = raw["structured_fields"]
        if not isinstance(structured, dict):
            raise TypeError("OCR fields must be an object")
        fields = {key: structured.get(key, "") for key, *_ in schema}
        if any(not isinstance(value, str) for value in fields.values()):
            raise TypeError("OCR field must be text")
    elif side == "back":
        fields = parse_back(raw.get("text_lines", []), raw.get("paddle_text_lines", []))
    else:
        fields = {}
        for key, _label, upstream in schema:
            value = raw.get(upstream)
            if value is not None and not isinstance(value, (str, int, float)):
                raise TypeError("OCR field must be text")
            fields[key] = str(value).strip() if value is not None else ""
    if side == "front" and not fields.get("id_number") and rows:
        from .namlhn_identity import extract_id_number
        original_lines = raw.get("text_lines", [])
        evidence = (original_lines if original_lines and all(isinstance(line, dict) and isinstance(line.get("text"), str)
                                                             for line in original_lines) else [{"text": text} for text in rows])
        value, _box, validation = extract_id_number(evidence, with_details=True)
        if value:
            fields["id_number"] = value
            raw = {**raw, "id_number_validation": validation}
    expiry = None
    if include_expiry:
        from .expiry import read_expiry
        fields["expiry_date"], expiry = read_expiry(raw, side)
    raw_text = "\n".join(rows or backup_rows)
    count = sum(bool(value) for value in fields.values())
    complete = all(fields[key] for key, *_ in schema)
    if expiry is not None and side == "front":
        complete = complete and (bool(fields["expiry_date"]) or expiry["expiry_type"] == "unlimited")
    result = {
        "success": True,
        "status": "complete" if complete else "partial" if count else "unstructured" if raw_text else "no_text",
        "side": side,
        "detected_side": detected,
        "fields": fields,
        "raw_text": raw_text,
        "recognized_fields": count,
        "total_fields": len(fields),
        "processing_time_ms": round((perf_counter() - started) * 1000),
    }
    if expiry is not None:
        result["expiry"] = expiry
    for key in ("recovery", "enhancement", "timings_ms", "work", "authority_region", "image_preprocessing", "id_number_validation"):
        if isinstance(raw.get(key), dict):
            result[key] = raw[key]
    if result.get("authority_region", {}).get("status") == "partial":
        result["status"] = "partial"
    if side == "back":
        result["back_ocr_debug"] = raw.get("back_ocr_debug") or back_ocr_debug(
            raw.get("initial_text_lines", raw.get("text_lines", [])), raw.get("text_lines", []))
    return result


def process_qr_image(data, reader):
    started = perf_counter()
    with closing(load_image(data)) as image:
        texts = reader.read(image)
    result = qr_result(texts)
    result["processing_time_ms"] = round((perf_counter() - started) * 1000)
    return result


def qr_result(texts):
    """Summarize decoded QR content without exposing its raw payload."""
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
    }

