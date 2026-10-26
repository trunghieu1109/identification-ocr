"""Client for the namlhn CUDA worker used by the main GPU profile."""
import io
import os

from .engine import OCRUnavailable
from .gpu_engine import DocUGPUEngine


class NamLhnGPUEngine(DocUGPUEngine):
    def __init__(self, base_url=None, transport=None):
        super().__init__(base_url if base_url is not None else os.getenv("NAMLHN_GPU_OCR_URL", ""), transport)
        self.api_key = os.getenv("NAMLHN_GPU_OCR_API_KEY", os.getenv("GPU_OCR_API_KEY", ""))
        default_model = "vgg_seq2seq" if os.getenv("NAMLHN_GPU_PIPELINE", "docu") == "docu" else os.getenv("OCR_MODEL", "vgg_transformer")
        self.model = os.getenv("NAMLHN_GPU_MODEL", default_model)

    def describe(self):
        info = super().describe()
        ready = self._ready_info or {}
        return {**info, "provider": "namlhn/vietnam-cccd-ocr",
                "recognition_model": ready.get("recognition_model", info.get("recognition_model", self.model)),
                **({"strategy": ready["strategy"]} if ready.get("strategy") == "namlhn_docu_v1" else {}),
                **{key: ready[key] for key in ("recognition_provider", "recognition_device", "card_crop", "card_enhancement") if key in ready},
                **({"text_enhancement": self._ready_info["text_enhancement"]}
                   if (self._ready_info or {}).get("text_enhancement") in {"off", "adaptive"} else {}),
                "decoder": (self._ready_info or {}).get("decoder", "vietocr_default")}

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

    def extract_experiment_image(self, image, side, needed_fields=None, source_data=None):
        if source_data is None:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            source_data = buffer.getvalue()
        return self._extract_data(source_data, side, needed_fields, include_layout=True)
