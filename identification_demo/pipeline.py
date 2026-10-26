"""Read both card sides, preferring QR values and filling gaps with OCR."""
import logging
from contextlib import ExitStack
from time import perf_counter

from .documents import FRONT_FIELDS, BACK_FIELDS
from .engine import OCRBusy, OCRUnavailable, OCRMultipleDocuments
from .processing import InvalidImage, load_image, process_image
from .qr import QRBusy, parse_cccd_qr


CARD_FIELDS = (
    "id_number", "old_id_number", "name", "date_of_birth", "gender",
    "nationality", "place_of_origin", "place_of_residence", "issue_date",
    "issuing_authority", "identifying_features", "expiry_date",
)
OPTIONAL_FIELDS = {"old_id_number", "identifying_features"}
logger = logging.getLogger(__name__)


def _complete_qr(fields):
    return all(value for key, value in fields.items() if key not in OPTIONAL_FIELDS)


def process_card(images, engine, reader, quality="medium", include_back_ocr=False):
    started = perf_counter()
    fields = dict.fromkeys(CARD_FIELDS, "")
    sources = dict.fromkeys(CARD_FIELDS)
    warnings = []
    failures = []
    qr_sides = {}
    ocr_sides = {}
    candidates = []
    timings = {}
    expiry_candidates = []
    expiry_label_found = False

    # Validate both inputs before calling either reader, including the second
    # image when the first image already contains a usable QR code.
    with ExitStack() as stack:
        loaded = {}
        stage_started = perf_counter()
        for side, data in images.items():
            try:
                loaded[side] = load_image(data)
                stack.callback(loaded[side].close)
            except InvalidImage as exc:
                label = "Mặt trước" if side == "front" else "Mặt sau"
                raise InvalidImage(f"{label}: {exc}") from exc
        timings["image_decode"] = round((perf_counter() - stage_started) * 1000, 2)
        for side, image in loaded.items():
            stage_started = perf_counter()
            try:
                texts = reader.read(image)
            except Exception as exc:
                logger.warning("Combined extraction: QR reader failed on %s (%s)", side, type(exc).__name__)
                failures.append(exc)
                qr_sides[side] = "error"
                warnings.append(f"Chưa đọc được QR ở {'mặt trước' if side == 'front' else 'mặt sau'}; đã thử bổ sung bằng OCR.")
                continue
            finally:
                timings[f"qr_{side}"] = round((perf_counter() - stage_started) * 1000, 2)
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
            needed = [key for key, *_ in schema if not fields[key] and key not in OPTIONAL_FIELDS]
            # QR has no expiry date. Read its caption/value on either side in
            # the existing OCR pass, including when QR filled the other fields.
            needed.append("expiry_date")
            stage_started = perf_counter()
            try:
                result = process_image(loaded[side], engine, side, needed_fields=needed,
                                       source_data=images[side], include_expiry=True)
            except InvalidImage as exc:
                label = "Mặt trước" if side == "front" else "Mặt sau"
                raise InvalidImage(f"{label}: {exc}") from exc
            except Exception as exc:
                logger.warning("Combined extraction: OCR failed on %s (%s)", side, type(exc).__name__)
                failures.append(exc)
                detail = str(exc) if isinstance(exc, (OCRBusy, OCRUnavailable)) else "Lỗi xử lý OCR."
                ocr_sides[side] = {"status": "error", "filled_fields": [], "error": detail}
                if isinstance(exc, OCRMultipleDocuments):
                    ocr_sides[side]["error_code"] = exc.code
                    ocr_sides[side]["detected_regions"] = exc.detected_regions
                mode = "GPU" if quality == "high" else "CPU"
                warnings.append(f"OCR {mode} {'mặt trước' if side == 'front' else 'mặt sau'} chưa xử lý được. {detail} Thông tin đã đọc được vẫn được giữ lại.")
                continue
            finally:
                timings[f"ocr_{side}"] = round((perf_counter() - stage_started) * 1000, 2)
            filled = []
            for key in needed:
                if key == "expiry_date":
                    continue
                value = result["fields"].get(key, "")
                if value:
                    fields[key], sources[key] = value, f"ocr_{side}"
                    filled.append(key)
            ocr_sides[side] = {"status": result["status"], "filled_fields": filled}
            expiry_label_found = expiry_label_found or result["expiry"]["label_found"]
            expiry_candidates.extend({**candidate, "source": f"ocr_{side}", "side": side}
                                     for candidate in result["expiry"]["candidates"])
            for key in ("recovery", "enhancement", "timings_ms", "work", "authority_region", "image_preprocessing", "id_number_validation"):
                if key in result:
                    ocr_sides[side][key] = result[key]
            if "id_number" in filled and result.get("id_number_validation", {}).get("status") in {"unverified", "ambiguous"}:
                warnings.append("Số căn cước OCR chưa đúng định dạng hoặc có nhiều cách đọc; đã giữ lại chữ đọc được để đối chiếu.")
            if side == "back" and include_back_ocr:
                ocr_sides[side]["debug"] = result["back_ocr_debug"]

        expiry_values = {(item["expiry_type"], item["expiry_date"]) for item in expiry_candidates}
        expiry_kind, expiry_value = next(iter(expiry_values)) if len(expiry_values) == 1 else (None, "")
        expiry_source = expiry_candidates[0]["source"] if len(expiry_values) == 1 else None
        expiry = {"status": "conflict" if len(expiry_values) > 1 else "found" if expiry_values else "not_found",
                  "expiry_type": expiry_kind, "label_found": expiry_label_found,
                  "side": expiry_candidates[0]["side"] if expiry_source else None}
        if expiry_value:
            fields["expiry_date"], sources["expiry_date"] = expiry_value, expiry_source
            ocr_sides[expiry["side"]]["filled_fields"].append("expiry_date")
        if expiry["status"] == "conflict":
            warnings.append("Các dòng OCR cho ngày hết hạn khác nhau; chưa chọn một ngày. Hãy đối chiếu ảnh gốc.")

        count = sum(bool(value) for value in fields.values())
        if not count and failures and expiry_kind != "unlimited":
            if any(isinstance(exc, OCRMultipleDocuments) for exc in failures):
                raise InvalidImage("GPU chưa chọn được một vùng thẻ duy nhất và chưa đọc được thông tin. Hãy cắt ảnh sát một thẻ rồi thử lại.")
            if any(isinstance(exc, (OCRBusy, QRBusy)) for exc in failures):
                raise OCRBusy("Dịch vụ đang bận và chưa đọc được thông tin. Hãy thử lại sau ít giây.")
            raise OCRUnavailable("Chưa đọc được thông tin do lỗi bộ xử lý QR/OCR. Hãy thử lại hoặc kiểm tra dịch vụ.")
        missing = [key for key in CARD_FIELDS if not fields[key] and key not in OPTIONAL_FIELDS
                   and not (key == "expiry_date" and expiry_kind == "unlimited")]
        authority_partial = any(stage.get("authority_region", {}).get("status") == "partial"
                                and "issuing_authority" in stage["filled_fields"] for stage in ocr_sides.values())
        if authority_partial:
            warnings.append("Cơ quan cấp mới đọc được một phần; phần đã đọc được vẫn được giữ lại để đối chiếu với thẻ gốc.")
        status = "complete" if not missing and not authority_partial else "partial" if count or expiry_kind == "unlimited" else "no_text"
        total_ms = round((perf_counter() - started) * 1000, 2)
        timings["merge"] = round(max(0, total_ms - sum(timings.values())), 2)
        timings["total"] = total_ms
        ready_info = getattr(engine, "_ready_info", None) or {}
        model = ready_info.get("recognition_model") or getattr(engine, "model", None)
        return {
            "success": True,
            "source": "qr_ocr",
            "quality": quality,
            "ocr_engine": {
                "provider": "namlhn/vietnam-cccd-ocr",
                "device": "cuda" if quality == "high" else "cpu",
                **({"model": model} if model else {}),
                **{key: ready_info[key] for key in ("strategy", "text_enhancement", "recognition_provider", "recognition_device") if key in ready_info},
                "used": any(stage["status"] not in {"error", "skipped"} for stage in ocr_sides.values()),
            },
            "status": status,
            "fields": fields,
            "field_sources": sources,
            "missing_fields": missing,
            "qr": {"status": qr_status, "recognized_fields": qr_count, "sides": qr_sides},
            "ocr": ocr_sides,
            "expiry": expiry,
            "warnings": warnings,
            "recognized_fields": count,
            "total_fields": len(CARD_FIELDS),
            "processing_time_ms": round(total_ms),
            "timings_ms": timings,
        }
