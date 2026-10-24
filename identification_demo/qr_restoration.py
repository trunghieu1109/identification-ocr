"""Locate QR crops on original pixels, restore on GPU, then decode with WeChat."""
import base64
import io
import json
import os
from math import ceil, floor
from time import monotonic, perf_counter

import httpx
from PIL import Image

from .qr import QRBusy, QRUnavailable
from .restoration_assets import MAX_RESTORATION_SIDE, RESTORATION_MODES
from .wechat_qr import create_wechat_decoder, decode_wechat


RESTORATION_METHODS = {"wechat_restormer": "restormer", "wechat_swinir": "swinir", "wechat_both": "both"}
MAX_QR_CROPS = 4


class QRRestorationClient:
    def __init__(self, url=None, transport=None):
        self.url = (url if url is not None else os.getenv("QR_RESTORATION_URL", os.getenv("GPU_OCR_URL", ""))).rstrip("/")
        self.headers = {"Authorization": "Bearer " + os.getenv("GPU_OCR_API_KEY")} if os.getenv("GPU_OCR_API_KEY") else {}
        self.transport = transport
        self._health = None
        self._health_at = 0

    def describe(self):
        if not self.url:
            return {"models": {}, "message": "Cần bật worker GPU để thử phục hồi ảnh QR."}
        if self._health is not None and monotonic() - self._health_at < 10:
            return self._health
        try:
            with httpx.Client(transport=self.transport, timeout=2) as client:
                response = client.get(self.url + "/api/v1/qr/restoration/health", headers=self.headers)
                response.raise_for_status()
                info = response.json()
                if not isinstance(info, dict) or not isinstance(info.get("models"), dict):
                    raise ValueError("Invalid restoration health")
                self._health = info
        except (httpx.HTTPError, ValueError):
            self._health = {"models": {}, "message": "Worker phục hồi QR chưa sẵn sàng. Hãy bật/cập nhật dịch vụ GPU."}
        self._health_at = monotonic()
        return self._health

    def restore(self, image, mode):
        if not self.url:
            raise QRUnavailable("Chưa cấu hình worker phục hồi QR. Hãy bật dịch vụ GPU.")
        with io.BytesIO() as stream:
            image.save(stream, "PNG")
            data = stream.getvalue()
        try:
            with httpx.Client(transport=self.transport, timeout=httpx.Timeout(120, connect=5)) as client:
                response = client.post(self.url + "/api/v1/qr/restoration", headers=self.headers,
                                       data={"mode": mode}, files={"file": ("qr-crop.png", data, "image/png")})
            if response.status_code == 429:
                raise QRBusy("Worker GPU đang bận. Hãy thử lại sau ít giây.")
            if response.status_code != 200:
                raise QRUnavailable("Worker chưa phục hồi được QR. Hãy kiểm tra model và dịch vụ GPU.")
            info = json.loads(response.headers.get("X-QR-Restoration", "null"))
            if not isinstance(info, dict) or info.get("mode") != mode or not isinstance(info.get("steps"), list):
                raise ValueError("Invalid restoration metadata")
            with Image.open(io.BytesIO(response.content)) as output:
                expected_scale = 4 if "swinir" in RESTORATION_MODES[mode] else 1
                if output.format != "PNG" or output.size != (image.width * expected_scale, image.height * expected_scale):
                    raise ValueError("Invalid restored crop")
                restored = output.convert("RGB")
            return restored, info
        except (httpx.HTTPError, ValueError, OSError) as exc:
            raise QRUnavailable("Chưa nhận được ảnh phục hồi hợp lệ từ GPU. Hãy thử lại hoặc kiểm tra dịch vụ.") from exc


def thumbnail(image):
    with image.copy() as preview, io.BytesIO() as stream:
        preview.thumbnail((384, 384))
        preview.save(stream, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def candidate_regions(detector, image):
    """Scan original plus overlapping tiles; merge detections in original coordinates."""
    import numpy as np
    width, height = image.size
    boxes = [(0, 0, width, height)]
    if max(width, height) > 1600:
        tw, th = round(width * .6), round(height * .6)
        boxes += [(x, y, x + tw, y + th) for y in (0, height - th) for x in (0, width - tw)]
    candidates = []
    for left, top, right, bottom in boxes:
        with image.crop((left, top, right, bottom)) as view:
            detected = detector.detect(image=np.asarray(view), is_bgr=False)
        for region in detected:
            x1, y1, x2, y2 = map(float, region["bbox_xyxy"])
            box = [max(0., x1 + left), max(0., y1 + top), min(float(width), x2 + left), min(float(height), y2 + top)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            candidates.append({"box": box, "confidence": float(region["confidence"])})
    unique = []
    def overlap(first, second):
        width = max(0., min(first[2], second[2]) - max(first[0], second[0]))
        height = max(0., min(first[3], second[3]) - max(first[1], second[1]))
        area = min((first[2] - first[0]) * (first[3] - first[1]),
                   (second[2] - second[0]) * (second[3] - second[1]))
        return width * height / area
    for candidate in sorted(candidates, key=lambda value: value["confidence"], reverse=True):
        if not any(overlap(candidate["box"], previous["box"]) >= .65 for previous in unique):
            unique.append(candidate)
    return unique


class WeChatRestoration:
    def __init__(self, detector, client):
        self.detector = detector
        self.client = client
        self.decoder = create_wechat_decoder()

    def read(self, image, mode):
        started = perf_counter()
        candidates = candidate_regions(self.detector, image)
        stage_timings = {"detect_regions": round((perf_counter() - started) * 1000, 2), "restore": 0., "decode": 0.}
        texts, regions, reports = [], [], []
        previews = None
        for candidate in candidates[:MAX_QR_CROPS]:
            x1, y1, x2, y2 = candidate["box"]
            margin = max(8, round(max(x2 - x1, y2 - y1) * .12))
            box = (max(0, floor(x1) - margin), max(0, floor(y1) - margin),
                   min(image.width, ceil(x2) + margin), min(image.height, ceil(y2) + margin))
            with image.crop(box) as crop:
                source_size = crop.size
                crop.thumbnail((MAX_RESTORATION_SIDE, MAX_RESTORATION_SIDE), Image.Resampling.LANCZOS)
                scale_x, scale_y = crop.width / source_size[0], crop.height / source_size[1]
                restoring = perf_counter()
                restored, info = self.client.restore(crop, mode)
                stage_timings["restore"] += (perf_counter() - restoring) * 1000
                try:
                    decoding = perf_counter()
                    decoded, points = decode_wechat(self.decoder, restored)
                    stage_timings["decode"] += (perf_counter() - decoding) * 1000
                    texts.extend(decoded)
                    report = {**info, "bbox": list(box), "source_size": list(source_size),
                              "decoded": bool(decoded), "confidence": round(candidate["confidence"], 4)}
                    reports.append(report)
                    if previews is None:
                        previews = {"before": thumbnail(crop), "after": thumbnail(restored)}
                    if decoded:
                        factor_x = restored.width / crop.width * scale_x
                        factor_y = restored.height / crop.height * scale_y
                        for polygon in points:
                            regions.append({"points": [[float(x) / factor_x + box[0], float(y) / factor_y + box[1]]
                                                       for x, y in polygon.reshape(-1, 2)], "decoded": True})
                    else:
                        regions.append({"points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                        "decoded": False, "confidence": round(candidate["confidence"], 4)})
                finally:
                    restored.close()
        return {"texts": list(dict.fromkeys(texts)), "regions": regions,
                "restoration": {"mode": mode, "device": "cuda:0", "detected_regions": len(candidates),
                                "processed_regions": len(reports), "region_limit": MAX_QR_CROPS,
                                "truncated": len(candidates) > MAX_QR_CROPS, "crops": reports,
                                "timings_ms": {key: round(value, 2) for key, value in stage_timings.items()},
                                "previews": previews}}
