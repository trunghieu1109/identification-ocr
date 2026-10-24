import io
import json
import types

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from identification_demo.engine import OCRBusy
from identification_demo.qr import QRBusy, QRUnavailable
from identification_demo.qr_restoration import QRRestorationClient, WeChatRestoration, candidate_regions
from identification_demo.qr_restoration_worker import QRRestorationEngine, add_restoration_routes, tiled_restore
from identification_demo.processing import InvalidImage
from identification_demo.work_queue import ModelQueue


def png(size=(80, 64)):
    stream = io.BytesIO()
    Image.new("RGB", size, "white").save(stream, "PNG")
    return stream.getvalue()


def test_client_sends_only_crop_with_selected_mode_and_auth(monkeypatch):
    monkeypatch.setenv("GPU_OCR_API_KEY", "qr-secret")
    requests = []
    def handler(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer qr-secret"
        assert b'swinir' in request.content and b'qr-crop.png' in request.content
        return httpx.Response(200, content=png((320, 256)), headers={"X-QR-Restoration": json.dumps({"mode": "swinir", "steps": []})})
    with Image.new("RGB", (80, 64)) as image:
        restored, info = QRRestorationClient("http://gpu.test", httpx.MockTransport(handler)).restore(image, "swinir")
    assert restored.size == (320, 256) and info["mode"] == "swinir"
    restored.close()
    assert len(requests) == 1 and requests[0].url.path == "/api/v1/qr/restoration"


@pytest.mark.parametrize("code,exception", [(429, QRBusy), (401, QRUnavailable), (503, QRUnavailable)])
def test_client_worker_failures_do_not_leak_details(code, exception):
    transport = httpx.MockTransport(lambda _: httpx.Response(code, json={"error": "private worker detail"}))
    with Image.new("RGB", (50, 50)) as image, pytest.raises(exception) as exc:
        QRRestorationClient("http://gpu.test", transport).restore(image, "restormer")
    assert "private worker detail" not in str(exc.value)


@pytest.mark.parametrize("content,metadata", [(b"invalid PNG", {"mode": "both", "steps": []}),
    (png(), {"mode": "both", "steps": []}), (png(), None), (png(), {"mode": "other", "steps": []})])
def test_client_rejects_invalid_images_dimensions_and_metadata(content, metadata):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=content, headers={"X-QR-Restoration": json.dumps(metadata)}))
    with Image.new("RGB", (80, 64)) as image, pytest.raises(QRUnavailable):
        QRRestorationClient("http://gpu.test", transport).restore(image, "both")


def test_health_is_cached_and_does_not_call_inference():
    calls = []
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"models": {"restormer": {"available": True, "model_loaded": False}}})
    client = QRRestorationClient("http://gpu.test", httpx.MockTransport(handler))
    assert client.describe() == client.describe()
    assert calls == ["/api/v1/qr/restoration/health"]


def test_unconfigured_worker_has_actionable_error():
    client = QRRestorationClient("")
    assert client.describe()["models"] == {}
    with pytest.raises(QRUnavailable, match="GPU"):
        client.restore(None, "restormer")


def test_candidate_deduplication_preserves_distinct_qr_regions():
    pytest.importorskip("numpy")
    detection = lambda box, confidence: {"bbox_xyxy": box, "confidence": confidence}
    detector = types.SimpleNamespace(detect=lambda **kwargs: [detection([10, 20, 80, 90], .9),
        detection([12, 22, 78, 88], .8), detection([150, 20, 200, 70], .7)])
    with Image.new("RGB", (300, 200)) as image:
        candidates = candidate_regions(detector, image)
    assert len(candidates) == 2
    assert candidates[0]["confidence"] == .9


def test_restored_regions_map_back_to_source_and_source_is_not_modified(monkeypatch):
    import identification_demo.qr_restoration as module
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(module, "candidate_regions", lambda *_: [{"box": [100, 100, 500, 500], "confidence": .9}])
    monkeypatch.setattr(module, "create_wechat_decoder", lambda: object())
    seen = []
    def decode(_decoder, image):
        seen.append(image.size)
        return ["valid decoded text"], [np.array([[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]])]
    monkeypatch.setattr(module, "decode_wechat", decode)
    class Worker:
        def restore(self, image, mode):
            assert mode == "both" and max(image.size) == 384
            return image.resize((image.width * 4, image.height * 4)), {"mode": mode, "steps": []}
    with Image.new("RGB", (800, 600), (10, 20, 30)) as image:
        before = image.tobytes()
        result = WeChatRestoration(object(), Worker()).read(image, "both")
        assert image.tobytes() == before
    assert seen == [(1536, 1536)]
    assert result["regions"][0]["points"] == [[52., 52.], [548., 52.], [548., 548.], [52., 548.]]
    assert result["restoration"]["crops"][0]["source_size"] == [496, 496]
    assert result["restoration"]["previews"]["after"].startswith("data:image/jpeg;base64,")


def test_no_detected_region_never_runs_restoration_or_returns_original(monkeypatch):
    import identification_demo.qr_restoration as module
    monkeypatch.setattr(module, "candidate_regions", lambda *_: [])
    monkeypatch.setattr(module, "create_wechat_decoder", lambda: object())
    worker = types.SimpleNamespace(restore=lambda *_: pytest.fail("No crop must mean no restoration inference"))
    with Image.new("RGB", (100, 100)) as image:
        result = WeChatRestoration(object(), worker).read(image, "restormer")
    assert result["texts"] == [] and result["regions"] == []
    assert result["restoration"]["processed_regions"] == 0
    assert result["restoration"]["previews"] is None


def test_tiling_preserves_pixels_and_size_for_scaled_non_square_inputs():
    torch = pytest.importorskip("torch")
    pixels = torch.rand(1, 3, 53, 83)
    model = lambda image: torch.nn.functional.interpolate(image, scale_factor=4, mode="nearest")
    with torch.inference_mode():
        output = tiled_restore(model, pixels, scale=4, tile=32, overlap=8)
    assert output.shape == (1, 3, 212, 332)
    assert torch.allclose(output, model(pixels), atol=1e-6)


def test_oversized_crop_is_rejected_before_model_loading():
    engine = QRRestorationEngine()
    with Image.new("RGB", (385, 100)) as image, pytest.raises(InvalidImage):
        engine.restore(image, "both")
    assert engine._models == {}
    engine.close()


def test_restoration_reuses_ocr_queue_without_closing_it():
    queue = ModelQueue("test-shared-gpu")
    engine = QRRestorationEngine(queue)
    engine.close()
    value, _ = queue.run(lambda: "still running")
    assert value == "still running"
    queue.close()


class FakeGPU:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
    def describe(self):
        return {"models": {"restormer": {"available": True, "model_loaded": False}}}
    def restore(self, image, mode):
        self.calls.append((image.size, mode))
        if self.error:
            raise self.error
        return image.copy(), {"mode": mode, "steps": []}


def worker_app(engine, token=""):
    app = FastAPI()
    add_restoration_routes(app, engine, token)
    return TestClient(app)


def test_worker_internal_endpoint_returns_png_and_metadata():
    engine = FakeGPU()
    response = worker_app(engine).post("/api/v1/qr/restoration", data={"mode": "restormer"}, files={"file": ("crop.png", png(), "image/png")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert json.loads(response.headers["x-qr-restoration"])["mode"] == "restormer"
    assert engine.calls == [((80, 64), "restormer")]


def test_worker_internal_auth_invalid_input_and_unknown_mode_do_not_run_model():
    engine = FakeGPU()
    client = worker_app(engine, "test-token")
    assert client.get("/api/v1/qr/restoration/health").status_code == 401
    assert client.post("/api/v1/qr/restoration", data={"mode": "restormer"}, files={"file": ("c.png", png())}).status_code == 401
    assert client.post("/api/v1/qr/restoration", headers={"authorization": "Bearer test-token"}, data={"mode": "restormer"}, files={"file": ("c.png", b"bad")}).status_code == 422
    assert client.post("/api/v1/qr/restoration", data={"mode": "unknown"}, files={"file": ("c.png", png())}).status_code == 422
    assert not engine.calls


@pytest.mark.parametrize("failure,code", [(QRUnavailable("Missing weights"), 503), (OCRBusy("busy"), 429), (RuntimeError("private detail"), 503)])
def test_worker_failure_statuses_are_actionable_and_sanitized(failure, code):
    response = worker_app(FakeGPU(failure)).post("/api/v1/qr/restoration", data={"mode": "restormer"}, files={"file": ("c.png", png())})
    assert response.status_code == code
    assert "private detail" not in response.text
