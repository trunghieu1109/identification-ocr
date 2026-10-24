"""Independent QR experiments; never fall back to another detector or OCR."""
import importlib
import importlib.util
from contextlib import closing
from threading import Lock
from time import perf_counter

from PIL import Image

from .processing import load_image, qr_result
from .qr import QRBusy, QRUnavailable
from .qr_parallel import WeChatParallel
from .qr_restoration import QRRestorationClient, RESTORATION_METHODS, WeChatRestoration
from .wechat_qr import WECHAT_FILES, create_wechat_decoder, decode_wechat, model_directory


METHODS = {"zxing": "ZXing", "wechat": "WeChatQRCode", "qrdet": "QRDet + ZXing",
           "wechat_restormer": "WeChat + Restormer", "wechat_swinir": "WeChat + SwinIR",
           "wechat_both": "WeChat + Restormer + SwinIR"}
LEGACY_METHODS = {"wechat_parallel": "WeChat song song"}
QRDET_RELEASE = "https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release"


def _barcode_text(barcode):
    try:
        return barcode.bytes.decode("utf-8")
    except UnicodeDecodeError:
        return barcode.text


def _zxing_scan(backend, image):
    return backend.read_barcodes(image, formats=backend.BarcodeFormat.QRCode,
                                 try_rotate=True, try_downscale=True, return_errors=True)


class QRExperiments:
    def __init__(self):
        self._lock = Lock()
        self._backends = {}
        self._restoration = QRRestorationClient()

    def describe(self):
        directory = model_directory()
        dependencies = {"zxing": ("zxingcpp",), "wechat": ("cv2",),
                        "qrdet": ("qrdet", "zxingcpp", "cv2"), "wechat_parallel": ("cv2", "numpy")}
        model_files = {"zxing": (), "wechat": WECHAT_FILES,
                       "qrdet": ("qrdet/qrdet-s.pt", "qrdet/current_release.txt"), "wechat_parallel": WECHAT_FILES}
        result = {}
        remote = self._restoration.describe()
        for method, label in METHODS.items():
            if method in RESTORATION_METHODS:
                from .restoration_assets import RESTORATION_MODES
                installed = all(importlib.util.find_spec(name) is not None for name in ("qrdet", "cv2"))
                models = all((directory / name).is_file() for name in WECHAT_FILES + model_files["qrdet"])
                names = RESTORATION_MODES[RESTORATION_METHODS[method]]
                ready = installed and models and all(remote.get("models", {}).get(name, {}).get("available") for name in names)
                result[method] = {"label": label, "available": bool(ready), "model_loaded": bool(ready) and all(
                    remote["models"][name].get("model_loaded", False) for name in names),
                    "message": "Sẵn sàng · GPU" if ready else remote.get("message", "Cần model phục hồi QR trên worker GPU.")}
                continue
            installed = all(importlib.util.find_spec(name) is not None for name in dependencies[method])
            models = all((directory / name).is_file() for name in model_files[method])
            available = installed and models
            result[method] = {
                "label": label, "available": available, "model_loaded": method in self._backends,
                "message": "Sẵn sàng" if available else "Cần cài thư viện QR và tải model; xem hướng dẫn Thử nghiệm QR trong README.",
            }
        return result

    def _initialize(self, method):
        if method in self._backends:
            return self._backends[method]
        directory = model_directory()
        try:
            if method == "zxing":
                backend = importlib.import_module("zxingcpp")
            elif method == "wechat":
                backend = create_wechat_decoder()
            elif method == "wechat_parallel":
                backend = WeChatParallel()
            elif method in RESTORATION_METHODS:
                backend = WeChatRestoration(self._initialize("qrdet"), self._restoration)
            else:
                module = importlib.import_module("qrdet")
                importlib.import_module("zxingcpp")
                weights = directory / "qrdet"
                marker = weights / "current_release.txt"
                # QRDet otherwise downloads weights itself, without a timeout.
                if not (weights / "qrdet-s.pt").is_file() or not marker.is_file() or marker.read_text() != QRDET_RELEASE:
                    raise QRUnavailable("Thiếu model QRDet. Hãy chạy scripts/setup-qr-models.py hoặc build lại Docker.")
                backend = module.QRDetector(model_size="s", conf_th=0.5, weights_folder=str(weights))
                backend.model.to("cpu")
        except ImportError as exc:
            raise QRUnavailable("Chưa cài thư viện thử nghiệm QR. Hãy cài requirements-qr.txt hoặc build lại Docker.") from exc
        self._backends[method] = backend
        return backend

    def read(self, image, method):
        if method not in METHODS and method not in LEGACY_METHODS:
            raise ValueError("Unknown QR experiment")
        if not self._lock.acquire(timeout=2):
            raise QRBusy("Đang chạy một thử nghiệm QR khác. Hãy thử lại sau ít giây.")
        try:
            cold_start = method not in self._backends
            started = perf_counter()
            backend = self._initialize(method)
            initialized = perf_counter()
            if method == "wechat_parallel":
                output = backend.read(image)
            elif method in RESTORATION_METHODS:
                output = backend.read(image, RESTORATION_METHODS[method])
            else:
                texts, regions = getattr(self, f"_read_{method}")(backend, image)
                output = {"texts": texts, "regions": regions}
            finished = perf_counter()
            return {
                **output, "texts": list(dict.fromkeys(output["texts"])), "cold_start": cold_start,
                "timings_ms": {"initialize": round((initialized - started) * 1000, 2),
                               "detect_decode": round((finished - initialized) * 1000, 2)},
            }
        finally:
            self._lock.release()

    def _read_zxing(self, backend, image):
        barcodes = _zxing_scan(backend, image)
        texts, regions = [], []
        for barcode in barcodes:
            if barcode.valid:
                texts.append(_barcode_text(barcode))
            position = barcode.position
            regions.append({"points": [[point.x, point.y] for point in (
                position.top_left, position.top_right, position.bottom_right, position.bottom_left)],
                "decoded": bool(barcode.valid)})
        return texts, regions

    def _read_wechat(self, backend, image):
        texts, points = decode_wechat(backend, image)
        # This API returns quadrangles for decoded codes, not every failed detection.
        regions = [{"points": polygon.reshape(-1, 2).tolist(), "decoded": True} for polygon in points]
        return list(texts), regions

    def _read_qrdet(self, backend, image):
        import cv2
        import numpy as np
        zxing = importlib.import_module("zxingcpp")
        rgb = np.asarray(image)
        detections = backend.detect(image=rgb, is_bgr=False)
        texts, regions = [], []
        for detection in detections:
            quad = np.asarray(detection["padded_quad_xy"], dtype=np.float32)
            # Order the four corners around their center, keeping adjacent points
            # adjacent even for a rotated or perspective-distorted QR.
            center = quad.mean(axis=0)
            quad = quad[np.argsort(np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0]))]
            edges = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
            size = min(2048, max(32, int(np.ceil(edges.max()))))
            target = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(quad, target)
            crop = cv2.warpPerspective(rgb, transform, (size, size), borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(255, 255, 255))
            # Add the quiet zone lost when a detector tightly crops the symbol.
            border = max(8, size // 10)
            crop = cv2.copyMakeBorder(crop, border, border, border, border, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            with closing(Image.fromarray(crop)) as cropped:
                decoded = [_barcode_text(code) for code in _zxing_scan(zxing, cropped) if code.valid]
            texts.extend(decoded)
            regions.append({"points": quad.tolist(), "confidence": round(float(detection["confidence"]), 4),
                            "decoded": bool(decoded)})
        return texts, regions


def process_qr_experiment(data, experiments, method):
    started = perf_counter()
    with closing(load_image(data)) as image:
        image_decoded = perf_counter()
        output = experiments.read(image, method)
        dimensions = {"width": image.width, "height": image.height}
    texts = output.pop("texts")
    result = qr_result(texts)
    result.update(output)
    result.update(source="qr_experiment", method=method, method_label={**METHODS, **LEGACY_METHODS}[method],
                  decoded_count=len(texts), image=dimensions)
    if not texts and result["regions"]:
        result["status"] = "detected_not_decoded"
        result["message"] = "Đã tìm thấy vùng QR nhưng chưa giải mã được nội dung."
    result["timings_ms"]["image_decode"] = round((image_decoded - started) * 1000, 2)
    result["processing_time_ms"] = round((perf_counter() - started) * 1000, 2)
    return result
