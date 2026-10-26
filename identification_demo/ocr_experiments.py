"""Compare OCR engines with full field extraction and optional text diagnostics."""
import logging
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

from .documents import FRONT_FIELDS, BACK_FIELDS
from .engine import OCRBusy, OCRUnavailable, OCRMultipleDocuments
from .expiry import expiry_rows, read_expiry
from .field_recovery import STRATEGY
from .expiry_context import STRATEGY as EXPANDED_STRATEGY
from .pipeline import CARD_FIELDS, _complete_qr
from .processing import InvalidImage, load_image, process_image
from .qr import QRBusy, parse_cccd_qr

logger = logging.getLogger(__name__)
EXPERIMENT_FIELDS = CARD_FIELDS
OPTIONAL_FIELDS = {"old_id_number", "identifying_features"}
ENGINES = {
    "namlhn_cpu": {"label": "namlhn CPU", "provider": "namlhn/vietnam-cccd-ocr", "device": "cpu"},
    "namlhn_gpu": {"label": "namlhn GPU", "provider": "namlhn/vietnam-cccd-ocr", "device": "cuda"},
    "docu_gpu": {"label": "DocU GPU", "provider": "CaoAnhNato/vn-identity-card-ocr", "device": "cuda"},
    "docu_gpu_regions": {"label": "DocU GPU · đọc theo vùng", "provider": "CaoAnhNato/vn-identity-card-ocr",
                         "device": "cuda", "strategy": STRATEGY},
    "docu_gpu_v3": {"label": "DocU GPU · v3 mở rộng box", "provider": "CaoAnhNato/vn-identity-card-ocr",
                    "device": "cuda", "strategy": EXPANDED_STRATEGY},
}


class _CaptureEngine:
    """Retain one request's OCR pass to parse expiry without another inference."""
    def __init__(self, engine):
        self.engine = engine
        self.raw = None

    def extract_image(self, image, side, needed_fields=None, source_data=None):
        method = getattr(self.engine, "extract_experiment_image", None) or getattr(self.engine, "extract_image", None)
        if method:
            raw = method(image, side, needed_fields=needed_fields, source_data=source_data)
        else:
            with tempfile.TemporaryDirectory(prefix="ocr-experiment-", dir=os.getenv("OCR_UPLOAD_TMP_DIR")) as directory:
                path = Path(directory) / "card.png"
                image.save(path)
                raw = self.engine.extract_document(path, side) if hasattr(self.engine, "extract_document") else self.engine.extract(path)
        if not isinstance(raw, dict):
            raise TypeError("OCR provider returned an invalid result")
        self.raw = raw
        return raw


def _expiry(raw):
    return read_expiry(raw, "front")


def process_ocr_experiment(images, engine, reader, engine_name, include_ocr_text=False, use_qr=True):
    started = perf_counter()
    fields = dict.fromkeys(EXPERIMENT_FIELDS, "")
    sources = dict.fromkeys(EXPERIMENT_FIELDS)
    timings, qr_sides, ocr_sides = {}, {}, {}
    warnings, failures, qr_candidates = [], [], []
    expiry = {"status": "not_found", "expiry_type": None, "side": "front", "label_found": False, "candidates": []}
    with ExitStack() as stack:
        step = perf_counter()
        loaded = {}
        for side in ("front", "back"):
            try:
                loaded[side] = stack.enter_context(load_image(images[side]))
            except InvalidImage as exc:
                raise InvalidImage(f"{'Mặt trước' if side == 'front' else 'Mặt sau'}: {exc}") from exc
        timings["image_decode"] = round((perf_counter() - step) * 1000, 2)
        for side, image in loaded.items():
            step = perf_counter()
            if not use_qr:
                qr_sides[side] = "skipped"
                timings[f"qr_{side}"] = 0
                continue
            try:
                texts = reader.read(image)
                parsed = [value for text in texts if (value := parse_cccd_qr(text)) is not None]
                qr_sides[side] = ("complete" if any(_complete_qr(value) for value in parsed) else "partial") if parsed else "unsupported_qr" if texts else "no_qr"
                qr_candidates.extend((side, value) for value in parsed)
            except Exception as exc:
                failures.append(exc)
                qr_sides[side] = "error"
                warnings.append(f"Chưa đọc được QR {'mặt trước' if side == 'front' else 'mặt sau'}; đã thử OCR.")
            finally:
                timings[f"qr_{side}"] = round((perf_counter() - step) * 1000, 2)
        for side, candidate in qr_candidates:
            if any(fields[key] and value and fields[key] != value for key, value in candidate.items()):
                raise InvalidImage("Các mã QR có thông tin căn cước không khớp. Hãy gửi hai mặt của cùng một thẻ.")
            for key, value in candidate.items():
                if value and not fields[key]:
                    fields[key], sources[key] = value, f"qr_{side}"
        qr_count = sum(bool(value) for value in fields.values())
        qr_status = ("complete" if _complete_qr({key: fields[key] for key in qr_candidates[0][1]}) else "partial") if qr_candidates else (
            "skipped" if not use_qr else "error" if "error" in qr_sides.values() else "unsupported_qr" if "unsupported_qr" in qr_sides.values() else "no_qr")
        if use_qr and not qr_candidates and not warnings:
            warnings.append("Không đọc được thông tin từ QR; đã dùng OCR trên hai mặt.")
        for side, schema in (("front", FRONT_FIELDS), ("back", BACK_FIELDS)):
            needed = [key for key, *_ in schema if not fields[key]]
            if side == "front":
                needed.append("expiry_date")
            capture = _CaptureEngine(engine)
            step = perf_counter()
            try:
                result = process_image(loaded[side], capture, side, needed_fields=needed, source_data=images[side])
                raw = capture.raw
                stage_fields = dict(result["fields"])
                if side == "front":
                    stage_fields["expiry_date"], expiry = _expiry(raw)
                filled = []
                for key, value in stage_fields.items():
                    if value and not fields[key]:
                        fields[key], sources[key] = value, f"ocr_{side}"
                        filled.append(key)
                stage = {"status": result["status"], "filled_fields": filled}
                for key in ("timings_ms", "work", "recovery", "enhancement", "authority_region", "image_preprocessing", "id_number_validation"):
                    if key in result:
                        stage[key] = result[key]
                if "id_number" in filled and result.get("id_number_validation", {}).get("status") in {"unverified", "ambiguous"}:
                    warnings.append("Số căn cước OCR chưa đúng định dạng hoặc có nhiều cách đọc; đã giữ lại chữ đọc được để đối chiếu.")
                if isinstance(raw.get("field_recovery"), dict):
                    stage["field_recovery"] = raw["field_recovery"]
                if isinstance(raw.get("text_enhancement"), dict):
                    stage["text_enhancement"] = {key: raw["text_enhancement"][key] for key in
                        ("mode", "method", "status", "attempted_regions", "changed_regions")
                        if key in raw["text_enhancement"]}
                if include_ocr_text:
                    marked = {row["number"] for candidate in expiry["candidates"] if candidate["reading"] == "primary"
                              for row in candidate["evidence"]} if side == "front" else set()
                    stage["debug"] = {"text_lines": [
                        {"number": row["number"], "text": row["text"], "is_expiry": row["number"] in marked,
                         **({"bbox": list(row["bbox"])} if row["bbox"] else {})}
                        for row in expiry_rows(raw.get("text_lines", []))]}
                    if side == "front" and isinstance(raw.get("field_mapping"), dict):
                        numbers = {row["index"]: row["number"] for row in expiry_rows(raw.get("text_lines", []))}
                        stage["debug"]["field_mapping"] = {
                            key: {"value": item["value"], "method": item["method"], "status": item["status"],
                                  "bbox": item["box"],
                                  "label_rows": [numbers[i] for i in item["label_indices"] if i in numbers],
                                  "value_rows": [numbers[i] for i in item["value_indices"] if i in numbers]}
                            for key, item in raw["field_mapping"].items()}
                    if side == "back":
                        stage["debug"]["authority"] = result.get("back_ocr_debug", {})
                    if isinstance(raw.get("region_debug"), list):
                        stage["debug"]["region_reads"] = raw["region_debug"]
                ocr_sides[side] = stage
            except InvalidImage as exc:
                raise InvalidImage(f"{'Mặt trước' if side == 'front' else 'Mặt sau'}: {exc}") from exc
            except Exception as exc:
                failures.append(exc)
                detail = str(exc) if isinstance(exc, (OCRBusy, OCRUnavailable)) else "Lỗi xử lý OCR."
                ocr_sides[side] = {"status": "error", "filled_fields": [], "error": detail}
                if isinstance(exc, OCRMultipleDocuments):
                    ocr_sides[side].update(error_code=exc.code, detected_regions=exc.detected_regions)
                warnings.append(f"{ENGINES[engine_name]['label']} chưa đọc được {'mặt trước' if side == 'front' else 'mặt sau'}. {detail}")
                logger.warning("OCR experiment failed; engine=%s side=%s error=%s", engine_name, side, type(exc).__name__)
            finally:
                timings[f"ocr_{side}"] = round((perf_counter() - step) * 1000, 2)

    count = sum(bool(value) for value in fields.values())
    if not count and failures and expiry["expiry_type"] != "unlimited":
        if any(isinstance(exc, OCRMultipleDocuments) for exc in failures):
            raise InvalidImage("Chưa chọn được một thẻ duy nhất. Hãy cắt ảnh sát một thẻ rồi thử lại.")
        if any(isinstance(exc, (OCRBusy, QRBusy)) for exc in failures):
            raise OCRBusy("Bộ xử lý đang bận và chưa đọc được thông tin. Hãy thử lại sau ít giây.")
        raise OCRUnavailable("Chưa đọc được thông tin do lỗi OCR/QR của phương án thử nghiệm.")
    missing = [key for key in EXPERIMENT_FIELDS if not fields[key] and key not in OPTIONAL_FIELDS
               and not (key == "expiry_date" and expiry["expiry_type"] == "unlimited")]
    partial_authority = any(stage.get("authority_region", {}).get("status") == "partial" for stage in ocr_sides.values())
    if expiry["status"] == "conflict":
        warnings.append("Các dòng OCR mặt trước cho ngày hết hạn khác nhau; chưa chọn một ngày.")
    if partial_authority:
        warnings.append("Cơ quan cấp mới đọc được một phần; hãy đối chiếu ảnh gốc.")
    status = "complete" if not missing and not partial_authority and expiry["status"] != "conflict" else "partial" if count or expiry["expiry_type"] == "unlimited" else "no_text"
    total = round((perf_counter() - started) * 1000, 2)
    timings["merge"] = round(max(0, total - sum(timings.values())), 2)
    timings["total"] = total
    if not include_ocr_text:
        expiry = {key: value for key, value in expiry.items() if key != "candidates"}
    ready_info = getattr(engine, "_ready_info", None) or {}
    model = ready_info.get("recognition_model") or getattr(engine, "model", None)
    return {"success": True, "experiment": "full_ocr", "engine": engine_name,
            "source": "qr_ocr" if use_qr else "ocr", "use_qr": use_qr,
            "quality": "medium" if engine_name == "namlhn_cpu" else "high",
            "status": status, "fields": fields, "field_sources": sources, "missing_fields": missing,
            "recognized_fields": count, "total_fields": len(EXPERIMENT_FIELDS),
            "qr": {"status": qr_status, "recognized_fields": qr_count, "sides": qr_sides},
            "ocr": ocr_sides, "expiry": expiry, "warnings": warnings,
            "ocr_engine": {**ENGINES[engine_name], "model": model,
                           **({"strategy": ready_info["strategy"]} if engine_name == "namlhn_gpu" and ready_info.get("strategy") == "namlhn_docu_v1" else {}),
                           **{key: ready_info[key] for key in ("recognition_provider", "recognition_device") if key in ready_info},
                           **({"text_enhancement": ready_info["text_enhancement"]}
                              if engine_name == "namlhn_gpu" and ready_info.get("text_enhancement") in {"off", "adaptive"} else {}),
                           **({"decoder": ready_info["decoder"]} if ready_info.get("decoder") in {"cuda_batched_v1", "vietocr_default"} else {}),
                           "used": any(stage["status"] != "error" for stage in ocr_sides.values())},
            "processing_time_ms": round(total), "timings_ms": timings}
