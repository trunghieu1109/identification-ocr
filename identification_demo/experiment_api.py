"""Lazy HTTP boundary for the optional parallel CCCD worker and diagnostics."""
import os
from threading import Lock
from uuid import uuid4

import httpx
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .processing import MAX_IMAGE_BYTES


class ExperimentClient:
    def __init__(self, base_url=None, transport=None):
        self.base_url = (base_url if base_url is not None else os.getenv("CCCD_EXPERIMENT_URL", "")).rstrip("/")
        self.transport = transport
        self._http = None
        self._lock = Lock()

    def _client(self):
        with self._lock:
            if self._http is None:
                key = os.getenv("CCCD_EXPERIMENT_API_KEY", "")
                self._http = httpx.Client(base_url=self.base_url, transport=self.transport, trust_env=False,
                    timeout=httpx.Timeout(240, connect=5), headers={"Authorization": f"Bearer {key}"} if key else {})
            return self._http

    def describe(self):
        if not self.base_url:
            return {"experimental": True, "available": False, "reason": "Chưa bật worker thử nghiệm."}
        try:
            info = self._client().get("/health", timeout=3).raise_for_status().json()
            if not isinstance(info, dict) or info.get("experimental") is not True:
                raise ValueError("Invalid experiment health")
            return info
        except (httpx.HTTPError, ValueError):
            return {"experimental": True, "available": False, "reason": "Worker thử nghiệm chưa sẵn sàng."}

    def extract(self, images, **options):
        if not self.base_url:
            return {"success": False, "error": "Chưa bật worker thử nghiệm."}, 503
        try:
            response = self._client().post("/extract", data=options, files={
                side + "_image": (side + ".png", data, "application/octet-stream") for side, data in images.items()})
            payload = response.json()
            if not isinstance(payload, dict) or (response.is_success and (payload.get("experimental") is not True
                    or not isinstance(payload.get("fields"), dict))):
                raise ValueError("Invalid experiment response")
            return payload, response.status_code
        except (httpx.HTTPError, ValueError):
            return {"success": False, "error": "Không kết nối được bộ xử lý thử nghiệm. Hãy thử lại."}, 503

    def close(self):
        if self._http is not None:
            self._http.close()


def register_experiment_routes(app, backend):
    @app.get("/api/cccd/experiment/health")
    def experiment_health():
        return backend.describe()

    @app.post("/api/cccd/experiment/extract", summary="Thử nghiệm DocAligner + FastMRZ + QR/OCR")
    async def experiment_extract(front_image: UploadFile = File(...), back_image: UploadFile = File(...),
                                 align: bool = Form(True), front_rotation: int = Form(0), back_rotation: int = Form(0)):
        request_id = uuid4().hex
        try:
            if front_rotation not in (0, 90, 180, 270) or back_rotation not in (0, 90, 180, 270):
                return JSONResponse({"success": False, "error": "Góc xoay không hợp lệ.", "request_id": request_id}, status_code=422)
            images = {}
            for side, upload in (("front", front_image), ("back", back_image)):
                data = await upload.read(MAX_IMAGE_BYTES + 1)
                if not data or len(data) > MAX_IMAGE_BYTES:
                    return JSONResponse({"success": False, "error": "Cần đủ hai ảnh JPG/PNG, tối đa 10 MB mỗi ảnh.",
                                         "request_id": request_id}, status_code=413 if data else 422)
                images[side] = data
            payload, status = await run_in_threadpool(backend.extract, images, align=align,
                front_rotation=front_rotation, back_rotation=back_rotation)
            payload["request_id"] = request_id
            return JSONResponse(payload, status_code=status, headers={"Retry-After": "5"} if status == 429 else None)
        finally:
            await front_image.close()
            await back_image.close()
