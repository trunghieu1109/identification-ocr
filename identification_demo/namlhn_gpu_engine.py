"""HTTP client for the main Paddle + DocU Seq2Seq GPU worker."""
import io
import os
from threading import Lock
from time import perf_counter

import httpx

from .engine import OCRBusy, OCRUnavailable, OCRMultipleDocuments
from .processing import InvalidImage


class NamLhnGPUEngine:
    def __init__(self, base_url=None, transport=None):
        self.base_url = (base_url if base_url is not None else os.getenv("NAMLHN_GPU_OCR_URL", "")).rstrip("/")
        self.transport = transport
        self.timeout = float(os.getenv("GPU_OCR_TIMEOUT", "120"))
        self.api_key = os.getenv("NAMLHN_GPU_OCR_API_KEY", "")
        self.model = "vgg_seq2seq"
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
                or info.get("cuda_available") is False
                or info.get("strategy") != "namlhn_docu_v1"
                or info.get("recognition_model") != self.model):
            raise OCRUnavailable("Mức cao chưa sẵn sàng: dịch vụ GPU chưa có CUDA hoặc chưa tải đủ model.")
        self._ready_info, self._ready_until = info, perf_counter() + 2
        return info

    def describe(self):
        info = {"provider": "namlhn/vietnam-cccd-ocr", "device": "cuda", "configured": bool(self.base_url),
                "available": False, "model_loaded": False, "recognition_model": self.model}
        if not self.base_url:
            return {**info, "reason": "Chưa cấu hình dịch vụ OCR GPU."}
        try:
            ready = self._ready(self._client())
            return {**info, "available": True, "model_loaded": True, "device": ready["device"],
                    "queue": ready.get("queue"),
                    **{key: ready[key] for key in ("recognition_model", "strategy", "recognition_provider",
                        "recognition_device", "card_crop", "card_enhancement", "text_enhancement") if key in ready}}
        except Exception:
            return {**info, "reason": "Dịch vụ GPU chưa sẵn sàng; kiểm tra CUDA và các model."}

    def extract_image(self, image, side, needed_fields=None, source_data=None):
        if source_data is None:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            source_data = buffer.getvalue()
        return self._extract_data(source_data, side, needed_fields)

    def _normalize_payload(self, payload):
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise OCRUnavailable("namlhn GPU trả kết quả không hợp lệ.")
        fields = payload.get("structured_fields")
        if not isinstance(fields, dict) or any(not isinstance(value, str) for value in fields.values()):
            raise OCRUnavailable("namlhn GPU trả trường thông tin không hợp lệ.")
        if payload.get("document_side") not in {"front", "back", "unknown"}:
            raise OCRUnavailable("namlhn GPU trả mặt thẻ không hợp lệ.")
        return {key: payload[key] for key in (
            "structured_fields", "document_side", "text_lines", "paddle_text_lines", "initial_text_lines",
            "timings_ms", "work", "authority_region", "back_ocr_debug", "strategy", "coordinate_space", "text_enhancement", "field_mapping",
            "image_preprocessing", "id_number_validation",
        ) if key in payload}

    def _extract_data(self, data, side, needed_fields=None):
        if not self.base_url:
            raise OCRUnavailable("Mức cao chưa sẵn sàng: chưa cấu hình dịch vụ OCR GPU.")
        try:
            client = self._client()
            self._ready(client)
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
            payload = response.json()
            result = self._normalize_payload(payload)
            result.setdefault("timings_ms", {})["worker_round_trip"] = round((perf_counter() - started) * 1000, 2)
            return result
        except (OCRBusy, OCRUnavailable, InvalidImage):
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            self._ready_until = 0
            raise OCRUnavailable("Không kết nối hoặc đọc được kết quả từ OCR GPU. Hãy kiểm tra dịch vụ GPU.") from exc
