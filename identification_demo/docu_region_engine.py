"""HTTP adapter for the isolated DocU field-recovery experiment."""
import io
import math
import os

from .field_recovery import STRATEGY
from .engine import OCRUnavailable
from .gpu_engine import DocUGPUEngine
from .expiry_context import STRATEGY as EXPANDED_STRATEGY


class DocURegionGPUEngine(DocUGPUEngine):
    strategy = STRATEGY
    def __init__(self, base_url=None, transport=None):
        super().__init__(base_url if base_url is not None else os.getenv("DOCU_REGIONS_GPU_URL", ""), transport)
        self.api_key = os.getenv("DOCU_REGIONS_GPU_API_KEY", os.getenv("GPU_OCR_API_KEY", ""))

    def describe(self):
        return {**super().describe(), "strategy": self.strategy}

    def _ready(self, client):
        info = super()._ready(client)
        if info.get("strategy") != self.strategy:
            raise OCRUnavailable("Worker DocU đọc theo vùng chưa chạy đúng phiên bản thử nghiệm.")
        return info

    def extract_experiment_image(self, image, side, needed_fields=None, source_data=None):
        if source_data is None:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            source_data = buffer.getvalue()
        return self._extract_data(source_data, side, needed_fields, include_layout=True)

    def _normalize_payload(self, payload):
        result = super()._normalize_payload(payload)
        value = payload.get("timings_ms", {}).get("field_refinement")
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            result.setdefault("timings_ms", {})["field_refinement"] = round(value, 2)
        metadata = payload.get("field_recovery")
        if (isinstance(metadata, dict) and metadata.get("strategy") == self.strategy
                and metadata.get("target") in {"expiry_date", "issue_date"}
                and metadata.get("status") in {"skipped", "recovered", "not_found", "error"}):
            result["field_recovery"] = {key: metadata[key] for key in
                ("strategy", "target", "status", "reason", "anchor", "reread_regions", "elapsed_ms", "bbox") if key in metadata}
        if isinstance(payload.get("region_debug"), list):
            result["region_debug"] = payload["region_debug"][:3]
        return result


class DocUExpandedGPUEngine(DocURegionGPUEngine):
    strategy = EXPANDED_STRATEGY

    def __init__(self, base_url=None, transport=None):
        root = base_url if base_url is not None else os.getenv('DOCU_REGIONS_GPU_URL', '')
        super().__init__(root.rstrip('/')+'/expanded' if root else '', transport)
