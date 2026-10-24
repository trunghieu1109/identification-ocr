import importlib
import types

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from identification_demo.qr import QRBusy, QRUnavailable
from identification_demo.wechat_qr import WECHAT_FILES, WeChatQRReader
from identification_demo.web import create_app
from test_pipeline import Engine, Reader, PAYLOAD, image_data


@pytest.fixture
def wechat_models(tmp_path, monkeypatch):
    monkeypatch.setenv("QR_MODEL_DIR", str(tmp_path))
    for name in WECHAT_FILES:
        (tmp_path / name).write_bytes(b"model placeholder for mocked constructor")
    return tmp_path


@pytest.fixture
def wechat_backend(wechat_models, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    backend = types.SimpleNamespace(detectAndDecode=lambda pixels: ((PAYLOAD,), ()))
    calls = []
    def initialize(*paths):
        calls.append(paths)
        return backend
    monkeypatch.setattr(cv2, "wechat_qrcode_WeChatQRCode", initialize)
    return backend, calls


def test_reader_preserves_unicode_deduplicates_and_reuses_loaded_model(wechat_backend, wechat_models):
    backend, loads = wechat_backend
    received = []
    def decode(pixels):
        received.append(pixels.copy())
        return (PAYLOAD, PAYLOAD), ()
    backend.detectAndDecode = decode
    reader = WeChatQRReader()
    with Image.new("RGB", (200, 100), (10, 20, 30)) as image:
        for _ in range(2):
            assert reader.read(image) == [PAYLOAD]
            assert not reader._lock.locked()
    assert loads == [tuple(str(wechat_models / name) for name in WECHAT_FILES)]
    assert len(received) == 2
    assert received[0].shape == (100, 200, 3)
    assert received[0][0, 0].tolist() == [30, 20, 10]


def test_describe_retains_health_shape_without_loading_model(wechat_backend, wechat_models):
    _backend, loads = wechat_backend
    reader = WeChatQRReader()
    assert reader.describe() == {"provider": "wechat-qrcode", "available": True}
    assert loads == [] and reader._decoder is None
    (wechat_models / WECHAT_FILES[0]).unlink()
    assert reader.describe() == {"provider": "wechat-qrcode", "available": False}


def test_missing_models_fail_cleanly_and_loading_can_be_retried(wechat_backend, wechat_models):
    _backend, loads = wechat_backend
    missing = wechat_models / WECHAT_FILES[0]
    missing.unlink()
    reader = WeChatQRReader()
    with Image.new("RGB", (50, 50)) as image:
        for _ in range(2):
            with pytest.raises(QRUnavailable, match="setup-qr-models.py"):
                reader.read(image)
            assert not reader._lock.locked() and reader._decoder is None
        assert loads == []
        missing.write_bytes(b"model restored")
        assert reader.read(image) == [PAYLOAD]
    assert len(loads) == 1


def test_missing_dependency_does_not_try_zxing_or_qrdet(monkeypatch):
    imported = []
    def unavailable(name):
        imported.append(name)
        raise ImportError(name)
    monkeypatch.setattr(importlib, "import_module", unavailable)
    reader = WeChatQRReader()
    with pytest.raises(QRUnavailable, match="requirements-qr.txt"):
        reader.read(None)
    assert imported == ["cv2"]
    assert not reader._lock.locked() and reader._decoder is None


def test_opencv_without_contrib_has_actionable_error(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda _: types.SimpleNamespace())
    reader = WeChatQRReader()
    with pytest.raises(QRUnavailable, match="opencv-contrib-python"):
        reader.read(None)
    assert not reader._lock.locked()


def test_decoding_failure_unlocks_and_cached_model_can_be_reused(wechat_backend):
    backend, loads = wechat_backend
    reader = WeChatQRReader()
    def broken(_):
        raise RuntimeError("decoder failure")
    backend.detectAndDecode = broken
    with Image.new("RGB", (50, 50)) as image:
        with pytest.raises(RuntimeError, match="decoder failure"):
            reader.read(image)
        assert not reader._lock.locked()
        backend.detectAndDecode = lambda _: ((PAYLOAD,), ())
        assert reader.read(image) == [PAYLOAD]
    assert len(loads) == 1


def test_concurrent_main_requests_retain_busy_error():
    reader = WeChatQRReader()
    reader._lock.acquire()
    try:
        with pytest.raises(QRBusy):
            reader.read(None)
    finally:
        reader._lock.release()


def without_measurements(value):
    if isinstance(value, dict):
        return {key: without_measurements(item) for key, item in value.items()
                if key not in {"request_id", "processing_time_ms", "timings_ms", "request_timings_ms"}}
    if isinstance(value, list):
        return [without_measurements(item) for item in value]
    return value


def response_shape(value):
    if isinstance(value, dict):
        return {key: response_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [response_shape(item) for item in value]
    return type(value).__name__


@pytest.mark.parametrize("quality", ["medium", "high"])
def test_main_api_contract_and_qr_priority_match_previous_reader(wechat_backend, quality):
    backend, _loads = wechat_backend
    backend.detectAndDecode = lambda pixels: ((PAYLOAD,) if pixels.shape[1] == 300 else (), ())
    previous = TestClient(create_app(Engine(), Reader(), Engine()))
    current = TestClient(create_app(Engine(), gpu_engine=Engine()))
    assert isinstance(current.app.state.qr_reader, WeChatQRReader)
    files = {"front_image": ("front.png", image_data(), "image/png"),
             "back_image": ("back.png", image_data("back"), "image/png")}
    before = previous.post("/api/cccd/extract", data={"quality": quality}, files=files)
    after = current.post("/api/cccd/extract", data={"quality": quality}, files=files)
    assert before.status_code == after.status_code == 200
    assert without_measurements(before.json()) == without_measurements(after.json())
    assert response_shape(before.json()) == response_shape(after.json())
    assert after.json()["field_sources"]["name"] == "qr_front"
    assert after.json()["field_sources"]["nationality"] == "ocr_front"
    assert after.json()["field_sources"]["issuing_authority"] == "ocr_back"


@pytest.mark.parametrize("texts", [[PAYLOAD], [], ["https://example.com"],
    [PAYLOAD, "https://example.com"], [PAYLOAD.replace("15032021", "")]])
def test_qr_api_contract_matches_previous_reader_for_each_outcome(wechat_backend, texts):
    backend, _loads = wechat_backend
    backend.detectAndDecode = lambda _: (texts, ())
    previous = TestClient(create_app(Engine(), Reader(front=texts)))
    current = TestClient(create_app(Engine()))
    upload = {"file": ("qr.png", image_data(), "image/png")}
    before = previous.post("/api/qr/cccd", files=upload)
    after = current.post("/api/qr/cccd", files=upload)
    assert before.status_code == after.status_code == 200
    assert without_measurements(before.json()) == without_measurements(after.json())
    assert response_shape(before.json()) == response_shape(after.json())
    assert after.headers["cache-control"] == before.headers["cache-control"] == "no-store"
