"""Lazy, bounded GPU restoration for experimental QR crops only."""
import io
import json
import logging
from contextlib import closing
from time import perf_counter
from typing import Literal

from fastapi import File, Form, Header, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from .engine import OCRBusy
from .processing import InvalidImage, MAX_IMAGE_BYTES, load_image
from .qr import QRUnavailable
from .restoration_assets import (MAX_RESTORATION_SIDE, MODEL_FILES, MODEL_LABELS,
                                 RESTORATION_MODES, restoration_directory)
from .work_queue import ModelQueue


logger = logging.getLogger(__name__)


def tiled_restore(model, pixels, scale, tile, overlap):
    """Average overlapping tiles, retaining exact dimensions at image edges."""
    import torch
    import torch.nn.functional as F
    height, width = pixels.shape[-2:]
    padded = F.pad(pixels, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
    h, w = padded.shape[-2:]
    if max(h, w) <= tile:
        return model(padded)[..., :height * scale, :width * scale]
    th, tw = min(h, tile), min(w, tile)
    ys = list(range(0, h - th, tile - overlap)) + [h - th]
    xs = list(range(0, w - tw, tile - overlap)) + [w - tw]
    output = torch.zeros((1, 3, h * scale, w * scale), device=pixels.device, dtype=pixels.dtype)
    weights = torch.zeros((1, 1, h * scale, w * scale), device=pixels.device, dtype=pixels.dtype)
    for y in ys:
        for x in xs:
            restored = model(padded[..., y:y + th, x:x + tw])
            sy, sx, eh, ew = y * scale, x * scale, th * scale, tw * scale
            output[..., sy:sy + eh, sx:sx + ew] += restored
            weights[..., sy:sy + eh, sx:sx + ew] += 1
    return (output / weights)[..., :height * scale, :width * scale]


class QRRestorationEngine:
    def __init__(self, queue=None):
        self._queue = queue if queue is not None else ModelQueue("qr-restoration", capacity=4)
        self._owns_queue = queue is None
        self._models = {}

    def close(self):
        if self._owns_queue:
            self._queue.close()

    def describe(self):
        import torch
        cuda = torch.cuda.is_available()
        directory = restoration_directory()
        models = {name: {"label": MODEL_LABELS[name], "available": cuda and (directory / filename).is_file(),
                         "model_loaded": name in self._models} for name, filename in MODEL_FILES.items()}
        return {"device": "cuda:0", "cuda_available": cuda, "models": models,
                "max_input_side": MAX_RESTORATION_SIDE, "queue": self._queue.describe()}

    def _load(self, name):
        if name in self._models:
            return self._models[name]
        import torch
        if not torch.cuda.is_available():
            raise QRUnavailable("Thử nghiệm phục hồi QR cần GPU CUDA.")
        path = restoration_directory() / MODEL_FILES[name]
        if not path.is_file():
            raise QRUnavailable("Thiếu model phục hồi QR. Hãy chạy scripts/setup-qr-restoration.py và build lại worker GPU.")
        if name == "restormer":
            from .restoration_models.restormer import Restormer
            model = Restormer()
        else:
            from .restoration_models.swinir import SwinIR
            model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
                           depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                           mlp_ratio=2, upsampler="pixelshuffledirect", resi_connection="1conv")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["params"], strict=True)
        model.eval().requires_grad_(False).to("cuda:0")
        self._models[name] = model
        return model

    def restore(self, image, mode):
        if mode not in RESTORATION_MODES:
            raise ValueError("Unknown restoration mode")
        if max(image.size) > MAX_RESTORATION_SIDE:
            raise InvalidImage(f"Vùng QR phục hồi tối đa {MAX_RESTORATION_SIDE} pixel mỗi cạnh.")
        result, queue_timings = self._queue.run(self._restore, image, mode)
        result[1]["queue_wait_ms"] = queue_timings["queue_wait"]
        return result

    def _restore(self, image, mode):
        import numpy as np
        import torch
        steps, newly_loaded = [], []
        initialized = perf_counter()
        for name in RESTORATION_MODES[mode]:
            if name not in self._models:
                newly_loaded.append(name)
            self._load(name)
        torch.cuda.synchronize()
        initialize_ms = (perf_counter() - initialized) * 1000
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.
            pixels = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to("cuda:0")
            for name in RESTORATION_MODES[mode]:
                started = perf_counter()
                pixels = tiled_restore(self._models[name], pixels, 1 if name == "restormer" else 4,
                                       256 if name == "restormer" else 128, 32 if name == "restormer" else 16).clamp_(0, 1)
                torch.cuda.synchronize()
                steps.append({"model": name, "label": MODEL_LABELS[name],
                              "processing_time_ms": round((perf_counter() - started) * 1000, 2)})
            output = (pixels.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.).round().astype("uint8")
        info = {"device": "cuda:0", "mode": mode, "steps": steps, "newly_loaded": newly_loaded,
                "initialize_ms": round(initialize_ms, 2),
                "gpu_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
                "input_size": list(image.size), "output_size": [output.shape[1], output.shape[0]]}
        return Image.fromarray(output), info


def add_restoration_routes(app, engine, token):
    """Add internal endpoints without changing the existing OCR endpoints."""
    import secrets
    from starlette.concurrency import run_in_threadpool

    def error(message, code):
        return JSONResponse({"success": False, "error": message}, status_code=code,
                            headers={"Retry-After": "5"} if code == 429 else None)

    def allowed(authorization):
        return not token or secrets.compare_digest(authorization or "", f"Bearer {token}")

    @app.get("/api/v1/qr/restoration/health")
    def health(authorization: str | None = Header(None)):
        if not allowed(authorization):
            return error("Không có quyền truy cập phục hồi QR.", 401)
        return engine.describe()

    @app.post("/api/v1/qr/restoration")
    async def restore(file: UploadFile = File(...), mode: Literal["restormer", "swinir", "both"] = Form(...),
                      authorization: str | None = Header(None)):
        try:
            if not allowed(authorization):
                return error("Không có quyền truy cập phục hồi QR.", 401)
            data = await file.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                return error("Vùng QR tải lên quá lớn.", 413)
            def process():
                with closing(load_image(data)) as image:
                    restored, info = engine.restore(image, mode)
                with closing(restored), io.BytesIO() as stream:
                    restored.save(stream, "PNG")
                    return stream.getvalue(), info
            output, info = await run_in_threadpool(process)
            return Response(output, media_type="image/png", headers={"X-QR-Restoration": json.dumps(info, ensure_ascii=True)})
        except InvalidImage as exc:
            return error(str(exc), 422)
        except OCRBusy as exc:
            return error(str(exc), 429)
        except QRUnavailable as exc:
            return error(str(exc), 503)
        except Exception:
            logger.exception("QR restoration experiment failed")
            return error("Chưa phục hồi được vùng QR. Hãy kiểm tra model hoặc GPU.", 503)
        finally:
            await file.close()
