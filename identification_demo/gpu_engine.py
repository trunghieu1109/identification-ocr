"""HTTP adapter for CaoAnhNato/vn-identity-card-ocr running on a GPU worker."""
import os
import re
import io
import math
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter

import httpx

from .documents import FEATURE_LABEL, fold, parse_back, text_rows
from .engine import OCRBusy, OCRUnavailable, OCRMultipleDocuments
from .processing import InvalidImage
from .authority import incomplete_authority
from .ocr_debug import back_ocr_debug


PROVIDER = "CaoAnhNato/vn-identity-card-ocr"
FIELD_MAP = {
    "id_number": "id_number", "full_name": "name", "dob": "date_of_birth",
    "gender": "gender", "nationality": "nationality", "origin_location": "place_of_origin",
    "residence_location": "place_of_residence", "issue_date": "issue_date",
    "issue_authority": "issuing_authority", "characteristics": "identifying_features",
}


def _date(value):
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d%m%Y"):
        try:
            return datetime.strptime(value, pattern).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return ""


def _has_evidence(value, rows, label, standalone=False):
    if not value:
        return False
    value = fold(value)
    rows = [fold(row) for row in rows]
    for index, row in enumerate(rows):
        if standalone and row == value:
            return True
        match = re.search(label, row)
        if match and (re.search(r"\b" + re.escape(value) + r"\b", row[match.end():])
                      or value in rows[index + 1:index + 3]):
            return True
    return False


def normalize_result(payload):
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise OCRUnavailable("OCR GPU chưa xử lý được ảnh. Hãy thử lại hoặc kiểm tra dịch vụ GPU.")
    upstream = payload.get("extracted_fields")
    if not isinstance(upstream, dict):
        raise OCRUnavailable("Dịch vụ OCR GPU trả kết quả không đúng định dạng.")
    fields = {}
    details = payload.get("field_details") or {}
    if not isinstance(details, dict):
        raise OCRUnavailable("Dịch vụ OCR GPU trả chi tiết trường không hợp lệ.")
    for original, target in FIELD_MAP.items():
        value = upstream.get(original, "")
        if isinstance(value, dict):
            value = value.get("normalized_value", value.get("value", ""))
        if not isinstance(value, str):
            raise OCRUnavailable("Dịch vụ OCR GPU trả trường thông tin không hợp lệ.")
        if isinstance(details.get(original), dict) and details[original].get("is_valid") is False:
            value = ""
        fields[target] = value.strip()
    for key in ("date_of_birth", "issue_date"):
        fields[key] = _date(fields[key])
    rows = text_rows(payload)
    # The pinned upstream has default nationality/gender values. Only accept
    # them when the recognized text supports the value.
    if not _has_evidence(fields["nationality"], rows, r"quoc\s*tich|nationality"):
        fields["nationality"] = ""
    if not _has_evidence(fields["gender"], rows, r"gioi\s*tinh|\bsex\b|gender", standalone=True):
        fields["gender"] = ""
    # Upstream replaces several authority names with a fixed institution. Keep
    # the actual printed authority by parsing the GPU-recognized text instead.
    back = parse_back(payload.get("text_lines", []))
    # Incompleteness requests another read; it must never erase observed text.
    fields["issuing_authority"] = back["issuing_authority"]
    fields["issue_date"] = fields["issue_date"] or back["issue_date"]
    feature = back["identifying_features"] or fields["identifying_features"]
    fields["identifying_features"] = feature if FEATURE_LABEL.sub("", fold(feature)).strip(" :/;,-") else ""
    doc_type = str(payload.get("doc_type", "unknown")).lower()
    side = {"cccd_front": "front", "cccd_back": "back"}.get(doc_type, "unknown")
    if doc_type not in {"cccd_front", "cccd_back", "unknown", "card"}:
        raise OCRUnavailable("OCR GPU nhận diện giấy tờ chưa thuộc loại CCCD được hỗ trợ.")
    result = {"structured_fields": fields, "document_side": side, "text_lines": rows}
    if side == "back" or "initial_text_lines" in payload:
        final = payload.get("text_lines", [])
        result["back_ocr_debug"] = back_ocr_debug(payload.get("initial_text_lines", final), final,
                                                payload.get("authority_search_crop"))
    region = payload.get("authority_region")
    if isinstance(region, dict) and region.get("method") in {"layout", "text_order"}:
        if region.get("status") == "ambiguous":
            result["authority_region"] = {"status": "ambiguous", "method": region["method"]}
        else:
            bbox, size, count = region.get("bbox"), region.get("image_size"), region.get("line_count")
            if (isinstance(bbox, (list, tuple)) and len(bbox) == 4
                    and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in bbox)
                    and isinstance(size, (list, tuple)) and len(size) == 2
                    and all(type(value) is int and value > 0 for value in size)
                    and type(count) is int and 0 < count <= 8):
                result["authority_region"] = {"method": region["method"], "bbox": list(bbox), "image_size": list(size),
                                               "line_count": count, "anchor": region.get("anchor") if region.get("anchor") in {"title", "label", "institution"} else "institution"}
    if fields["issuing_authority"] and incomplete_authority(fields["issuing_authority"]):
        result.setdefault("authority_region", {"method": "text_order"})["status"] = "partial"
    for key, allowed in {
        "timings_ms": {"queue_wait", "inference", "worker_image_decode", "iqa", "segmentation", "rotation",
                       "detection", "recognition", "metadata", "authority_refinement", "recovery"},
        "work": {"primary_regions", "reread_regions", "full_image_attempts", "skipped_duplicate_passes"},
    }.items():
        values = payload.get(key)
        if isinstance(values, dict):
            result[key] = {name: round(value, 2) for name, value in values.items()
                           if name in allowed and type(value) in (int, float) and math.isfinite(value) and value >= 0}
    enhancement = payload.get("enhancement")
    if isinstance(enhancement, dict) and enhancement.get("target") == "issuing_authority":
        result["enhancement"] = {"target": "issuing_authority"}
        for key in ("reread_lines", "changed_lines", "recovered_lines"):
            value = enhancement.get(key)
            if key != "recovered_lines" or key in enhancement:
                result["enhancement"][key] = value if type(value) is int and 0 <= value <= 32 else 0
        for key in ("contrast_enhancement", "expanded_crop", "recovery_attempted"):
            result["enhancement"][key] = enhancement.get(key) is True
    recovery = payload.get("recovery")
    if isinstance(recovery, dict) and recovery.get("method") in {"segmented", "full_image"}:
        count, rotation, attempts = recovery.get("detected_regions"), recovery.get("rotation_degrees"), recovery.get("attempted_rotations", [])
        result["recovery"] = {
            "method": recovery["method"],
            "detected_regions": count if type(count) is int and 0 <= count <= 1000 else None,
            "rotation_degrees": rotation if type(rotation) is int and rotation in (0, 90, 180, 270) else None,
            "attempted_rotations": [value for value in attempts if type(value) is int and value in (0, 90, 180, 270)] if isinstance(attempts, list) else [],
        }
    return result


class DocUGPUEngine:
    def __init__(self, base_url=None, transport=None):
        self.base_url = (base_url if base_url is not None else os.getenv("GPU_OCR_URL", "")).rstrip("/")
        self.transport = transport
        self.timeout = float(os.getenv("GPU_OCR_TIMEOUT", "120"))
        self.api_key = os.getenv("GPU_OCR_API_KEY", "")
        self._http = None
        self._guard = Lock()
        self._ready_info = None
        self._ready_until = 0

    def _client(self):
        with self._guard:
            if self._http is None:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                self._http = httpx.Client(base_url=self.base_url, transport=self.transport, headers=headers,
                                          timeout=httpx.Timeout(self.timeout, connect=5), trust_env=False)
            return self._http

    def close(self):
        if self._http is not None:
            self._http.close()

    def _ready(self, client):
        if self._ready_info is not None and perf_counter() < self._ready_until:
            return self._ready_info
        response = client.get("/api/v1/health/ready", timeout=3)
        response.raise_for_status()
        info = response.json()
        if (not isinstance(info, dict) or info.get("pipeline_ready") is not True or not str(info.get("device", "")).startswith("cuda")
                or info.get("cuda_available") is False):
            raise OCRUnavailable("Mức cao chưa sẵn sàng: dịch vụ GPU chưa có CUDA hoặc chưa tải đủ model.")
        self._ready_info, self._ready_until = info, perf_counter() + 2
        return info

    def describe(self):
        info = {"provider": PROVIDER, "device": "cuda", "configured": bool(self.base_url),
                "available": False, "model_loaded": False}
        if not self.base_url:
            return {**info, "reason": "Chưa cấu hình dịch vụ OCR GPU."}
        try:
            ready = self._ready(self._client())
            description = {**info, "available": True, "model_loaded": True, "device": ready["device"], "queue": ready.get("queue")}
            for key in ("recognition_model", "authority_recognition_model"):
                if ready.get(key) in {"vgg_transformer", "vgg_seq2seq"}:
                    description[key] = ready[key]
            return description
        except Exception:
            return {**info, "reason": "Dịch vụ GPU chưa sẵn sàng; kiểm tra CUDA và các model."}

    def extract_document(self, image_path, side):
        return self._extract_data(Path(image_path), side)

    def extract_image(self, image, side, needed_fields=None, source_data=None):
        if source_data is None:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            source_data = buffer.getvalue()
        return self._extract_data(source_data, side, needed_fields)

    def _extract_data(self, data, side, needed_fields=None):
        if not self.base_url:
            raise OCRUnavailable("Mức cao chưa sẵn sàng: chưa cấu hình dịch vụ OCR GPU.")
        try:
            client = self._client()
            self._ready(client)
            if isinstance(data, Path):
                data = data.read_bytes()
            params = {"doc_type_hint": f"cccd_{side}"}
            if needed_fields is not None:
                params["fields"] = ",".join(needed_fields)
            started = perf_counter()
            response = client.post("/api/v1/extract", params=params,
                                   files={"file": ("card.jpg" if data[:2] == b"\xff\xd8" else "card.png", data,
                                                   "image/jpeg" if data[:2] == b"\xff\xd8" else "image/png")})
            if response.status_code == 429:
                raise OCRBusy("Dịch vụ OCR GPU đang bận. Hãy thử lại sau ít giây.")
            if response.status_code == 422:
                try:
                    detail = response.json()
                except ValueError:
                    detail = {}
                if isinstance(detail, dict) and detail.get("error_code") == "multiple_documents":
                    count = detail.get("detected_regions")
                    count = count if type(count) is int and 2 <= count <= 1000 else None
                    raise OCRMultipleDocuments(count)
                raise InvalidImage("Dịch vụ OCR GPU từ chối ảnh nhưng chưa trả nguyên nhân cụ thể. Hãy thử lại bằng ảnh JPG/PNG rõ nét.")
            response.raise_for_status()
            result = normalize_result(response.json())
            result.setdefault("timings_ms", {})["worker_round_trip"] = round((perf_counter() - started) * 1000, 2)
            return result
        except (OCRBusy, OCRUnavailable, InvalidImage):
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            self._ready_until = 0
            raise OCRUnavailable("Không kết nối hoặc đọc được kết quả từ OCR GPU. Hãy kiểm tra dịch vụ GPU.") from exc
