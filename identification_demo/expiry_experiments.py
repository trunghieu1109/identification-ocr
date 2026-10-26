"""Inspect expiry captions, dates and OCR evidence from both card sides."""
import logging
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

from .documents import text_rows
from .engine import OCRBusy, OCRUnavailable
from .expiry import expiry_rows, has_expiry_label, parse_expiry
from .processing import load_image

logger = logging.getLogger(__name__)

def _read(image, data, engine, side):
    if hasattr(engine, "extract_expiry_image"):
        raw = engine.extract_expiry_image(image, side, source_data=data)
    elif hasattr(engine, "extract_image"):
        # Read both sides even when QR or existing fields would skip OCR.
        # No new field is sent to the production GPU worker's field whitelist.
        raw = engine.extract_image(image, side, source_data=data)
    else:
        with tempfile.TemporaryDirectory(prefix="cccd-expiry-", dir=os.getenv("OCR_UPLOAD_TMP_DIR")) as directory:
            path = Path(directory) / "card.png"
            image.save(path)
            raw = engine.extract_document(path, side) if hasattr(engine, "extract_document") else engine.extract(path)
    if not isinstance(raw, dict):
        raise OCRUnavailable("Bộ OCR trả kết quả không hợp lệ.")
    return raw


def process_expiry_experiment(images, engine, quality="medium"):
    started = perf_counter()
    timings, reports, candidates, failures, warnings = {}, {}, [], [], []
    with ExitStack() as stack:
        # Validate both images before either model invocation.
        loaded = {side: stack.enter_context(load_image(data)) for side, data in images.items()}
        timings["image_decode"] = round((perf_counter() - started) * 1000, 2)
        for side in ("front", "back"):
            step_started = perf_counter()
            try:
                raw = _read(loaded[side], images[side], engine, side)
                rows = text_rows(raw)
                backup = text_rows(raw, "paddle_text_lines")
                matches = parse_expiry(raw.get("text_lines", []))
                # Legacy engines can expose both readings. Keep disagreement visible.
                backup_matches = parse_expiry(raw.get("paddle_text_lines", [])) if backup and backup != rows else []
                marked = {item["number"] for match in matches for item in match["evidence"]}
                report = {"status": "found" if matches or backup_matches else "not_found",
                          "expiry_label_found": has_expiry_label(raw.get("text_lines", [])) or has_expiry_label(raw.get("paddle_text_lines", [])),
                          "text_lines": [{"number": i + 1, "text": text, "is_expiry": i + 1 in marked}
                                         for i, text in enumerate(rows)]}
                # Bounds belong to the OCR-processed image; keep them with the
                # text for debugging the caption/value association.
                for output, record in zip(report["text_lines"], expiry_rows(raw.get("text_lines", []))):
                    if record["bbox"]:
                        output["bbox"] = list(record["bbox"])
                if backup:
                    backup_marked = {item["number"] for match in backup_matches for item in match["evidence"]}
                    report["backup_text_lines"] = [
                        {"number": i + 1, "text": text, "is_expiry": i + 1 in backup_marked}
                        for i, text in enumerate(backup)]
                for reading, found in (("primary", matches), ("backup", backup_matches)):
                    for match in found:
                        candidates.append({**match, "source": f"ocr_{side}", "reading": reading})
                reports[side] = report
            except Exception as exc:
                failures.append(exc)
                detail = str(exc) if isinstance(exc, (OCRBusy, OCRUnavailable)) else "Lỗi xử lý OCR."
                reports[side] = {"status": "error", "error": detail, "text_lines": []}
                label = "mặt trước" if side == "front" else "mặt sau"
                warnings.append(f"Chưa đọc được {label}. {detail}")
                logger.warning("Expiry experiment failed; side=%s error=%s", side, type(exc).__name__)
            finally:
                timings[f"ocr_{side}"] = round((perf_counter() - step_started) * 1000, 2)
    if len(failures) == 2:
        if any(isinstance(exc, OCRBusy) for exc in failures):
            raise OCRBusy("OCR đang bận, chưa đọc được ngày hết hạn. Hãy thử lại sau ít giây.")
        raise OCRUnavailable("Chưa đọc được cả hai ảnh. Hãy kiểm tra bộ OCR hoặc thử lại.")

    values = {(item["expiry_type"], item["expiry_date"]) for item in candidates}
    conflict = len(values) > 1
    resolved = len(values) == 1
    kind, value = next(iter(values)) if resolved else (None, "")
    sources = list(dict.fromkeys(item["source"] for item in candidates)) if resolved else []
    status = "conflict" if conflict else "partial" if failures else "complete" if resolved else "not_found"
    if conflict:
        warnings.append("Các dòng OCR cho ngày hết hạn khác nhau; hãy đối chiếu hai ảnh, hệ thống chưa chọn một giá trị.")
    message = ("Có nhiều kết quả khác nhau, cần đối chiếu." if conflict else
               "Thẻ ghi không thời hạn." if kind == "unlimited" else
               "Đã đọc được ngày hết hạn." if resolved else
               "Đã thấy nhãn ngày hết hạn nhưng chưa đọc được ngày ở cạnh nhãn." if any(stage.get("expiry_label_found") for stage in reports.values()) else
               "Chưa tìm thấy ngày hết hạn gắn với nhãn trên hai mặt.")
    if failures:
        message += " Một mặt chưa xử lý được."
    total = round((perf_counter() - started) * 1000, 2)
    timings["total"] = total
    return {
        "success": True, "experiment": "expiry_date", "source": "ocr", "quality": quality,
        "status": status, "message": message,
        "fields": {"expiry_date": value}, "expiry_type": kind,
        "field_sources": {"expiry_date": sources[0] if sources else None},
        "sources": sources, "candidates": candidates, "ocr": reports, "warnings": warnings,
        "ocr_engine": {"provider": "namlhn/vietnam-cccd-ocr",
                       "device": "cuda" if quality == "high" else "cpu"},
        "processing_time_ms": round(total), "timings_ms": timings,
    }
