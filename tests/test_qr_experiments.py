import io
import types

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from identification_demo.qr import QRBusy, QRUnavailable
from identification_demo.qr_experiments import METHODS, QRExperiments, process_qr_experiment
from identification_demo.web import MAX_IMAGE_BYTES, create_app

PAYLOAD = "001099000001|012345678|NGUYỄN VĂN THỬ|01021999|Nam|Phường Thử, Hà Nội|15032021"


class NoOCR:
    def describe(self):
        return {"source_available": False, "model_loaded": False}

    def extract(self, *args, **kwargs):
        raise AssertionError("QR experiments must not invoke OCR")


class FakeExperiments:
    def __init__(self, texts=None, regions=None, error=None):
        self.texts = [PAYLOAD] if texts is None else texts
        self.regions = regions or []
        self.error = error
        self.calls = []

    def describe(self):
        return {name: {"available": True, "model_loaded": False} for name in METHODS}

    def read(self, image, method):
        self.calls.append((method, image.size, image.mode))
        if self.error:
            raise self.error
        return {"texts": self.texts, "regions": self.regions, "cold_start": True,
                "timings_ms": {"initialize": 1.2, "detect_decode": 3.4}}


def pixels(fmt="PNG", exif=None):
    output = io.BytesIO()
    Image.new("RGB", (300, 190), "white").save(output, fmt, **({"exif": exif} if exif else {}))
    return output.getvalue()


def post(client, method="zxing", data=None):
    return client.post("/api/qr/experiment", data={"method": method},
                       files={"file": ("../../image.png", pixels() if data is None else data, "image/png")})


@pytest.mark.parametrize("method", METHODS)
def test_only_selected_method_runs_without_ocr_and_raw_payload(method):
    experiments = FakeExperiments()
    client = TestClient(create_app(NoOCR(), qr_experiments=experiments))
    response = post(client, method)
    assert response.status_code == 200
    result = response.json()
    assert experiments.calls == [(method, (300, 190), "RGB")]
    assert result["method"] == method and result["source"] == "qr_experiment"
    assert result["status"] == "complete" and result["decoded_count"] == 1
    assert result["fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert result["cold_start"] and result["request_id"]
    assert result["timings_ms"]["detect_decode"] == 3.4
    assert result["timings_ms"]["initialize"] == 1.2
    assert result["timings_ms"]["image_decode"] >= 0
    assert response.headers["cache-control"] == "no-store"
    assert {"texts", "raw_qr", "raw_text", "qr_codes"}.isdisjoint(result)
    assert PAYLOAD not in response.text


@pytest.mark.parametrize("texts,regions,status", [
    ([], [], "no_qr"),
    ([], [{"points": [[0, 0], [20, 0], [20, 20], [0, 20]], "decoded": False}], "detected_not_decoded"),
    (["https://example.com"], [], "unsupported_qr"),
    ([PAYLOAD, "https://example.com"], [], "multiple_qr"),
])
def test_detection_and_decoding_have_distinct_outcomes(texts, regions, status):
    result = process_qr_experiment(pixels(), FakeExperiments(texts, regions), "qrdet")
    assert result["status"] == status
    assert result["decoded_count"] == len(texts)
    assert result["regions"] == regions
    assert not any(result["fields"].values())


@pytest.mark.parametrize("method", ["unknown", "", "ZXing"])
def test_unknown_method_never_invokes_a_reader(method):
    experiments = FakeExperiments()
    response = post(TestClient(create_app(NoOCR(), qr_experiments=experiments)), method)
    assert response.status_code == 422
    assert not experiments.calls


@pytest.mark.parametrize("data,status", [(b"", 422), (b"not pixels", 422),
    (pixels("GIF"), 422), (b"x" * (MAX_IMAGE_BYTES + 1), 413)])
def test_invalid_upload_is_rejected_before_experiment(data, status):
    experiments = FakeExperiments()
    response = post(TestClient(create_app(NoOCR(), qr_experiments=experiments)), data=data)
    assert response.status_code == status
    assert not experiments.calls


@pytest.mark.parametrize("error,status", [
    (QRUnavailable("Thiếu model"), 503), (QRBusy("Đang bận"), 429),
    (RuntimeError("private runtime failure"), 503),
])
def test_failures_are_isolated_and_do_not_invoke_another_method(error, status):
    experiments = FakeExperiments(error=error)
    response = post(TestClient(create_app(NoOCR(), qr_experiments=experiments)), "wechat")
    assert response.status_code == status
    assert [call[0] for call in experiments.calls] == ["wechat"]
    assert "private runtime failure" not in response.text
    if status == 429:
        assert response.headers["retry-after"] == "5"


def test_experiment_applies_exif_orientation():
    experiments = FakeExperiments()
    exif = Image.Exif()
    exif[274] = 6
    result = process_qr_experiment(pixels("JPEG", exif), experiments, "wechat")
    assert result["image"] == {"width": 190, "height": 300}
    assert experiments.calls == [("wechat", (190, 300), "RGB")]


def test_health_does_not_load_models(monkeypatch, tmp_path):
    monkeypatch.setenv("QR_MODEL_DIR", str(tmp_path))
    experiments = QRExperiments()
    monkeypatch.setattr(experiments, "_initialize", lambda _: pytest.fail("Health must not initialize QR models"))
    health = TestClient(create_app(NoOCR(), qr_experiments=experiments)).get("/api/health").json()
    assert set(health["qr_experiments"]) == set(METHODS)
    assert not any(method["model_loaded"] for method in health["qr_experiments"].values())


def test_failed_initialization_unlocks_and_can_be_retried(monkeypatch):
    experiments = QRExperiments()
    def unavailable(_):
        raise QRUnavailable("Thử lại")
    monkeypatch.setattr(experiments, "_initialize", unavailable)
    for _ in range(2):
        with pytest.raises(QRUnavailable):
            experiments.read(None, "wechat")
        assert not experiments._lock.locked()
    assert experiments._backends == {}


def test_concurrent_model_access_is_rejected():
    experiments = QRExperiments()
    experiments._lock.acquire()
    try:
        with pytest.raises(QRBusy):
            experiments.read(None, "qrdet")
    finally:
        experiments._lock.release()


def test_zxing_failed_symbols_are_not_accepted_as_decoded_payloads():
    point = types.SimpleNamespace(x=1, y=2)
    position = types.SimpleNamespace(top_left=point, top_right=point, bottom_right=point, bottom_left=point)
    codes = [types.SimpleNamespace(valid=False, position=position, bytes=b"invalid", text="invalid"),
             types.SimpleNamespace(valid=True, position=position, bytes=PAYLOAD.encode(), text="wrong encoding")]
    backend = types.SimpleNamespace(BarcodeFormat=types.SimpleNamespace(QRCode="QR"),
                                    read_barcodes=lambda *args, **kwargs: codes)
    texts, regions = QRExperiments()._read_zxing(backend, None)
    assert texts == [PAYLOAD]
    assert [region["decoded"] for region in regions] == [False, True]


def test_qrdet_crop_rectifies_pixels_and_decodes_unicode_without_full_image_scan():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("zxingcpp")
    qrcode = pytest.importorskip("qrcode")
    qr = qrcode.make(PAYLOAD, box_size=5, border=0).convert("RGB")
    image = Image.new("RGB", (1100, 700), "white")
    image.paste(qr, (90, 120))
    quad = np.array([[90, 120], [90 + qr.width, 120], [90 + qr.width, 120 + qr.height],
                     [90, 120 + qr.height]], dtype=np.float32)
    backend = types.SimpleNamespace(detect=lambda **kwargs: [{"padded_quad_xy": quad, "confidence": 0.97}])
    texts, regions = QRExperiments()._read_qrdet(backend, image)
    assert texts == [PAYLOAD]
    assert regions[0]["decoded"] and regions[0]["confidence"] == 0.97


def test_parallel_experiment_routes_to_two_branches_without_changing_default_reader(monkeypatch):
    import identification_demo.qr_experiments as module
    from identification_demo.wechat_qr import WeChatQRReader
    class Parallel:
        def read(self, image):
            return {"texts": [PAYLOAD], "regions": [], "selected_branch": "processed",
                    "branches": {"original": {"status": "no_qr"}, "processed": {"status": "decoded"}}}
    monkeypatch.setattr(module, "WeChatParallel", Parallel)
    client = TestClient(create_app(NoOCR()))
    assert isinstance(client.app.state.qr_reader, WeChatQRReader)
    assert client.app.state.qr_reader._decoder is None
    response = post(client, "wechat_parallel")
    assert response.status_code == 200
    result = response.json()
    assert result["selected_branch"] == "processed"
    assert result["branches"]["original"]["status"] == "no_qr"
    assert result["fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert {"texts", "raw_qr", "raw_text", "qr_codes"}.isdisjoint(result)
    assert PAYLOAD not in response.text
    assert client.app.state.qr_reader._decoder is None


@pytest.mark.parametrize("method,mode", [("wechat_restormer", "restormer"), ("wechat_swinir", "swinir"), ("wechat_both", "both")])
def test_restoration_methods_only_run_in_experiments(monkeypatch, method, mode):
    import identification_demo.qr_experiments as module
    calls = []
    class Restoration:
        def __init__(self, detector, client):
            assert detector is not None
        def read(self, image, selected):
            calls.append(selected)
            return {"texts": [PAYLOAD], "regions": [], "restoration": {"mode": selected}}
    monkeypatch.setattr(module, "WeChatRestoration", Restoration)
    experiments = QRExperiments()
    experiments._backends["qrdet"] = object()
    client = TestClient(create_app(NoOCR(), qr_experiments=experiments))
    result = post(client, method).json()
    assert calls == [mode]
    assert result["success"] and result["method"] == method
    assert result["restoration"]["mode"] == mode
    assert client.app.state.qr_reader._decoder is None
