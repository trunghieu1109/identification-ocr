"""Read both card sides, preferring QR values and filling gaps with OCR."""
import logging
from contextlib import ExitStack
from time import perf_counter

from .documents import FRONT_FIELDS, BACK_FIELDS
from .engine import OCRBusy, OCRUnavailable
from .processing import InvalidImage, load_image, process_image
from .qr import QRBusy, parse_cccd_qr


CARD_FIELDS = (
    "id_number", "old_id_number", "name", "date_of_birth", "gender",
    "nationality", "place_of_origin", "place_of_residence", "issue_date",
    "issuing_authority", "identifying_features",
)
OPTIONAL_FIELDS = {"old_id_number"}
logger = logging.getLogger(__name__)


def _complete_qr(fields):
    return all(value for key, value in fields.items() if key not in OPTIONAL_FIELDS)


def process_card(images, engine, reader):
    started = perf_counter()
    fields = dict.fromkeys(CARD_FIELDS, "")
    sources = dict.fromkeys(CARD_FIELDS)
    warnings = []
    failures = []
    qr_sides = {}
    ocr_sides = {}
    candidates = []

    # Validate both inputs before calling either reader, including the second
    # image when the first image already contains a usable QR code.
    with ExitStack() as stack:
        loaded = {}
        for side, data in images.items():
            try:
                loaded[side] = stack.enter_context(load_image(data))
            except InvalidImage as exc:
                label = "Mặt trước" if side == "front" else "Mặt sau"
                raise InvalidImage(f"{label}: {exc}") from exc
        for side, image in loaded.items():
            try:
                texts = reader.read(image)
            except Exception as exc:
                logger.warning("Combined extraction: QR reader failed on %s (%s)", side, type(exc).__name__)
                failures.append(exc)
                qr_sides[side] = "error"
                warnings.append(f"Chưa đọc được QR ở {'mặt trước' if side == 'front' else 'mặt sau'}; đã thử bổ sung bằng OCR.")
                continue
            parsed = [value for text in texts if (value := parse_cccd_qr(text)) is not None]
            if parsed:
                qr_sides[side] = "complete" if any(_complete_qr(value) for value in parsed) else "partial"
            else:
                qr_sides[side] = "unsupported_qr" if texts else "no_qr"
            candidates.extend((side, value) for value in parsed)

    # The same code can appear on both photos. Complementary values for the
    # same card are allowed; conflicting values must never be mixed.
    for side, candidate in candidates:
        if any(fields[key] and value and fields[key] != value for key, value in candidate.items()):
            raise InvalidImage("Các mã QR có thông tin căn cước không khớp. Hãy gửi hai mặt của cùng một thẻ và chỉ giữ một mã QR trên mỗi ảnh.")
        for key, value in candidate.items():
            if value and not fields[key]:
                fields[key], sources[key] = value, f"qr_{side}"

    qr_count = sum(bool(value) for value in fields.values())
    if candidates:
        qr_fields = {key: fields[key] for key in candidates[0][1]}
        qr_status = "complete" if _complete_qr(qr_fields) else "partial"
    elif "error" in qr_sides.values():
        qr_status = "error"
    elif "unsupported_qr" in qr_sides.values():
        qr_status = "unsupported_qr"
    else:
        qr_status = "no_qr"
    if not candidates and not warnings:
        warnings.append("Không đọc được thông tin CCCD từ QR; đã dùng OCR trên hai mặt.")

    for side, schema in (("front", FRONT_FIELDS), ("back", BACK_FIELDS)):
        needed = [key for key, *_ in schema if not fields[key]]
        if not needed:
            ocr_sides[side] = {"status": "skipped", "filled_fields": []}
            continue
        try:
            result = process_image(images[side], engine, side)
        except InvalidImage as exc:
            label = "Mặt trước" if side == "front" else "Mặt sau"
            raise InvalidImage(f"{label}: {exc}") from exc
        except Exception as exc:
            logger.warning("Combined extraction: OCR failed on %s (%s)", side, type(exc).__name__)
            failures.append(exc)
            ocr_sides[side] = {"status": "error", "filled_fields": []}
            warnings.append(f"OCR {'mặt trước' if side == 'front' else 'mặt sau'} chưa xử lý được; thông tin đã đọc được vẫn được giữ lại.")
            continue
        filled = []
        for key in needed:
            value = result["fields"].get(key, "")
            if value:
                fields[key], sources[key] = value, f"ocr_{side}"
                filled.append(key)
        ocr_sides[side] = {"status": result["status"], "filled_fields": filled}

    count = sum(bool(value) for value in fields.values())
    if not count and failures:
        if any(isinstance(exc, (OCRBusy, QRBusy)) for exc in failures):
            raise OCRBusy("Dịch vụ đang bận và chưa đọc được thông tin. Hãy thử lại sau ít giây.")
        raise OCRUnavailable("Chưa đọc được thông tin do lỗi bộ xử lý QR/OCR. Hãy thử lại hoặc kiểm tra dịch vụ.")
    missing = [key for key in CARD_FIELDS if not fields[key] and key not in OPTIONAL_FIELDS]
    status = "complete" if not missing else "partial" if count else "no_text"
    return {
        "success": True,
        "source": "qr_ocr",
        "status": status,
        "fields": fields,
        "field_sources": sources,
        "missing_fields": missing,
        "qr": {"status": qr_status, "recognized_fields": qr_count, "sides": qr_sides},
        "ocr": ocr_sides,
        "warnings": warnings,
        "recognized_fields": count,
        "total_fields": len(CARD_FIELDS),
        "processing_time_ms": round((perf_counter() - started) * 1000),
    }
