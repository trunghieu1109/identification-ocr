"""Compact public API contract for the optional parallel GPU pipeline."""
from .engine import OCRBusy, OCRUnavailable
from .pipeline import CARD_FIELDS, OPTIONAL_FIELDS
from .processing import InvalidImage


class ParallelUnavailable(OCRUnavailable):
    """Transport/contract failure; the caller may use the classic GPU flow."""


def main_result(raw, include_back_ocr=False):
    """Retain extracted fields and diagnostics explicitly allowed by the API.

    Preview images, candidates and full QR/MRZ/OCR text never enter the main
    response. Back OCR text is returned only when requested by the client.
    """
    if raw.get("status") == "identity_conflict":
        raise InvalidImage("Các nguồn QR/MRZ có số căn cước không khớp. Hãy gửi hai mặt của cùng một thẻ.")
    values = raw.get("fields", {})
    if not isinstance(values, dict) or any(not isinstance(values.get(key), str) for key in CARD_FIELDS):
        raise ValueError("Invalid parallel fields")
    fields = {key: values[key] for key in CARD_FIELDS}
    sources = {key: raw.get("field_sources", {}).get(key) for key in CARD_FIELDS}
    # Preserve the main API's existing optional-field policy.
    fields["identifying_features"], sources["identifying_features"] = "", None
    warnings = list(raw.get("warnings", []))
    ocr, qr_sides, alignments = {}, {}, {}
    expiry_label, unlimited_sides, expiry_values = False, [], set()
    qr_required = {"id_number", "name", "date_of_birth", "gender", "place_of_residence", "issue_date"}
    for side in ("front", "back"):
        stage = raw.get("sides", {}).get(side, {})
        reads = stage.get("ocr", [])
        valid = [item for item in reads if item.get("status") not in {"wrong_side", "error", "unavailable"}]
        if not valid and any(item.get("status") == "wrong_side" for item in reads):
            raise InvalidImage("Ảnh có dấu hiệu sai mặt thẻ. Hãy kiểm tra ảnh mặt trước và mặt sau.")
        chosen = valid[-1] if valid else reads[-1] if reads else {}
        filled = [key for key in CARD_FIELDS if fields[key] and (sources[key] or "").startswith(f"ocr_{side}")]
        status = chosen.get("status", "skipped")
        if status in {"unavailable", "wrong_side"}:
            status = "error"
        ocr[side] = {"status": status, "filled_fields": filled, "passes": len(reads)}
        if status == "error":
            ocr[side]["error"] = "Chưa đọc được chữ bằng OCR GPU; giữ thông tin từ QR/MRZ nếu có."
        for key in ("timings_ms", "work", "authority_region", "image_preprocessing", "id_number_validation"):
            if isinstance(chosen.get(key), dict):
                ocr[side][key] = chosen[key]
        if side == "back" and include_back_ocr and isinstance(chosen.get("back_ocr_debug"), dict):
            ocr[side]["debug"] = chosen["back_ocr_debug"]
        for item in reads:
            expiry = item.get("expiry", {})
            expiry_label = expiry_label or bool(expiry.get("label_found"))
            if expiry.get("expiry_type") == "unlimited":
                unlimited_sides.append(side)
                expiry_values.add(("unlimited", ""))
            for candidate in expiry.get("candidates", []):
                expiry_values.add((candidate.get("expiry_type"), candidate.get("expiry_date", "")))
        qr_reads = stage.get("qr", [])
        qr_sides[side] = ("partial" if any(item.get("status") == "found" for item in qr_reads)
                          else "error" if any(item.get("status") == "error" for item in qr_reads) else "no_qr")
        side_qr_fields = {item["field"] for item in raw.get("candidates", [])
                           if item.get("value") and item.get("source", "").startswith(f"qr_{side}_")}
        if qr_required <= side_qr_fields:
            qr_sides[side] = "complete"
        alignment = stage.get("alignment", {})
        alignments[side] = {key: alignment[key] for key in ("status", "confidence", "corners", "output_size") if key in alignment}
    review = [key for key in raw.get("review_fields", []) if key in CARD_FIELDS and key != "identifying_features"]
    expiry_conflict = len(expiry_values) > 1 and sources["expiry_date"] != "mrz_back"
    if expiry_conflict:
        fields["expiry_date"], sources["expiry_date"] = "", None
        for value in ocr.values():
            value["filled_fields"] = [key for key in value["filled_fields"] if key != "expiry_date"]
        expiry = {"status": "conflict", "expiry_type": None, "label_found": expiry_label, "side": None}
        warnings.append("Các dòng OCR cho ngày hết hạn khác nhau; chưa chọn một ngày. Hãy đối chiếu ảnh gốc.")
    elif fields["expiry_date"]:
        expiry = {"status": "found", "expiry_type": "date", "label_found": expiry_label,
                  "side": "back" if (sources["expiry_date"] or "").startswith(("mrz_back", "ocr_back")) else "front"}
    elif unlimited_sides:
        expiry = {"status": "found", "expiry_type": "unlimited", "label_found": expiry_label, "side": unlimited_sides[0]}
    else:
        expiry = {"status": "not_found", "expiry_type": None, "label_found": expiry_label, "side": None}
    count = sum(bool(value) for value in fields.values())
    if not count and any(value["status"] == "error" for value in ocr.values()):
        raise OCRUnavailable("Chưa đọc được thông tin từ QR, MRZ hoặc OCR GPU. Hãy thử lại hoặc kiểm tra dịch vụ GPU.")
    missing = [key for key in CARD_FIELDS if not fields[key] and key not in OPTIONAL_FIELDS
               and not (key == "expiry_date" and expiry["expiry_type"] == "unlimited")]
    authority_partial = ocr["back"].get("authority_region", {}).get("status") == "partial"
    qr_count = sum(bool(fields[key]) and (sources[key] or "").startswith("qr_") for key in CARD_FIELDS)
    mrz = raw.get("sides", {}).get("back", {}).get("mrz", {})
    qr_status = "partial" if qr_count else "error" if "error" in qr_sides.values() else "no_qr"
    if all(fields[key] and (sources[key] or "").startswith("qr_") for key in qr_required):
        qr_status = "complete"
    return {
        "success": True, "source": "qr_ocr", "quality": "high",
        "status": "complete" if not missing and not review and not authority_partial else "partial" if count or unlimited_sides else "no_text",
        "fields": fields, "field_sources": sources, "missing_fields": missing, "review_fields": review,
        "recognized_fields": count, "total_fields": len(CARD_FIELDS),
        "qr": {"status": qr_status, "recognized_fields": qr_count, "sides": qr_sides},
        "ocr": ocr, "expiry": expiry,
        "mrz": {"provider": "fastmrz", "status": mrz.get("status", "not_found"),
                "checks": mrz.get("checks", {}), "attempts": mrz.get("attempts", []),
                "used_fields": [key for key in CARD_FIELDS if (sources[key] or "").startswith("mrz_")],
                "stop_reason": mrz.get("stop_reason")},
        "alignment": alignments,
        "ocr_engine": {"provider": "namlhn/vietnam-cccd-ocr", "device": "cuda", "model": "vgg_seq2seq",
                       "strategy": "namlhn_docu_v1", "used": any(value["status"] not in {"error", "skipped"} for value in ocr.values())},
        "warnings": list(dict.fromkeys(warnings)),
        "processing_time_ms": raw.get("processing_time_ms", 0), "timings_ms": raw.get("timings_ms", {}),
        "stage_intervals_ms": raw.get("stage_intervals_ms", {}), "execution": raw.get("execution", {}),
        "gpu_pipeline": {"requested": "parallel", "used": "parallel", "fallback": False},
    }


def process_parallel_card(images, backend, include_back_ocr=False):
    payload, status = backend.extract(images, diagnostics=False, include_back_ocr=include_back_ocr)
    if not isinstance(payload, dict):
        raise ParallelUnavailable("Bộ xử lý song song trả kết quả không hợp lệ.")
    if status == 429:
        raise OCRBusy("Hàng đợi xử lý căn cước đang bận. Hãy thử lại sau ít giây.")
    if status in {413, 422}:
        raise InvalidImage(payload.get("error") or "Ảnh căn cước không hợp lệ.")
    if status != 200 or payload.get("success") is not True:
        raise ParallelUnavailable("Bộ xử lý QR/MRZ song song chưa sẵn sàng.")
    if (payload.get("source") != "qr_ocr" or not isinstance(payload.get("gpu_pipeline"), dict)
            or payload["gpu_pipeline"].get("used") != "parallel"
            or not isinstance(payload.get("fields"), dict) or set(payload["fields"]) != set(CARD_FIELDS)
            or any(not isinstance(value, str) for value in payload["fields"].values())
            or payload.get("status") not in {"complete", "partial", "no_text"}):
        raise ParallelUnavailable("Bộ xử lý song song trả kết quả không hợp lệ.")
    # The worker used main_result too. A public allowlist protects against a
    # misconfigured/older worker accidentally sending experiment debug data.
    allowed = {"success", "source", "quality", "status", "fields", "field_sources", "missing_fields", "review_fields",
               "recognized_fields", "total_fields", "qr", "ocr", "expiry", "mrz", "alignment", "ocr_engine", "warnings",
               "processing_time_ms", "timings_ms", "stage_intervals_ms", "execution", "gpu_pipeline"}
    result = {key: value for key, value in payload.items() if key in allowed}
    if not include_back_ocr:
        for stage in result.get("ocr", {}).values():
            stage.pop("debug", None)
    return result
