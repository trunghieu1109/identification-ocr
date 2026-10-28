import pytest
from fastapi.testclient import TestClient

from identification_demo.engine import NamLhnEngine, OCRBusy, OCRUnavailable, OCRMultipleDocuments
from identification_demo.namlhn_gpu_engine import NamLhnGPUEngine
from identification_demo.namlhn_gpu_worker import NamLhnDocUCudaEngine, create_namlhn_gpu_app
from identification_demo.processing import MAX_IMAGE_BYTES
from test_pipeline import image_data


class Worker:
    def __init__(self, error=None):
        self.initialized = False
        self.closed = False
        self.error = error
        self.calls = []

    def initialize(self):
        self.initialized = True

    def close(self):
        self.closed = True

    def describe(self):
        return {"pipeline_ready": self.initialized, "cuda_available": True, "device": "cuda:0"}

    def extract_image(self, image, side, needed_fields=None):
        self.calls.append((side, needed_fields, image.mode))
        if self.error:
            raise self.error
        return {"structured_fields": {"name": "TEST"}, "document_side": side,
                "text_lines": ["Có giá trị đến: 25/10/2036"]}


def submit(client, data=None, side="front", fields=None, headers=None):
    params = {"doc_type_hint": f"cccd_{side}"}
    if fields is not None:
        params["fields"] = fields
    return client.post("/api/v1/extract", params=params, headers=headers or {},
                       files={"file": ("../../card.png", image_data() if data is None else data, "image/png")})


def test_worker_warms_its_model_and_accepts_expiry_on_either_side():
    worker = Worker()
    with TestClient(create_namlhn_gpu_app(worker)) as client:
        assert client.get("/api/v1/health/ready").json()["pipeline_ready"]
        response = submit(client, fields="name,expiry_date")
        assert response.status_code == 200 and response.json()["success"]
        assert response.json()["timings_ms"]["worker_image_decode"] >= 0
        assert response.headers["cache-control"] == "no-store"
        assert submit(client, side="back", fields="expiry_date").status_code == 200
        assert submit(client, fields="unknown").status_code == 422
    assert worker.calls == [("front", ["name", "expiry_date"], "RGB"), ("back", ["expiry_date"], "RGB")] and worker.closed


@pytest.mark.parametrize("data,status", [(b"", 422), (b"invalid", 422), (b"x" * (MAX_IMAGE_BYTES + 1), 413)])
def test_invalid_images_do_not_reach_the_gpu(data, status):
    worker = Worker()
    with TestClient(create_namlhn_gpu_app(worker)) as client:
        assert submit(client, data=data).status_code == status
    assert not worker.calls


@pytest.mark.parametrize("exception,status", [(OCRBusy("busy"), 429), (OCRUnavailable("unavailable"), 503),
                                             (RuntimeError("private details"), 503)])
def test_worker_errors_do_not_leak_details_or_report_success(exception, status):
    worker = Worker(exception)
    with TestClient(create_namlhn_gpu_app(worker)) as client:
        response = submit(client)
    assert response.status_code == status and not response.json()["success"]
    assert "private details" not in response.text
    if status == 429:
        assert response.headers["retry-after"] == "5"


def test_authentication_is_checked_before_model_inference(monkeypatch):
    monkeypatch.setenv("NAMLHN_GPU_OCR_API_KEY", "test-token")
    worker = Worker()
    with TestClient(create_namlhn_gpu_app(worker)) as client:
        assert submit(client).status_code == 401
        assert not worker.calls
        assert submit(client, headers={"Authorization": "Bearer test-token"}).status_code == 200


def test_startup_failure_remains_observable():
    class NotReady(Worker):
        def initialize(self):
            raise OCRUnavailable("CUDA unavailable")
    with TestClient(create_namlhn_gpu_app(NotReady())) as client:
        assert client.get("/api/v1/health/ready").status_code == 503


def test_multiple_cards_are_reported_as_a_validation_error():
    with TestClient(create_namlhn_gpu_app(Worker(OCRMultipleDocuments(2)))) as client:
        response = submit(client)
    assert response.status_code == 422
    assert response.json()["error_code"] == "multiple_documents" and response.json()["detected_regions"] == 2


def test_namlhn_worker_cannot_silently_initialize_cpu_models():
    torch = pytest.importorskip("torch")
    paddle = pytest.importorskip("paddle")
    if torch.cuda.is_available() and paddle.device.is_compiled_with_cuda():
        pytest.skip("This check intentionally requires a CPU-only environment")
    engine = NamLhnDocUCudaEngine()
    try:
        with pytest.raises(OCRUnavailable, match="CUDA"):
            engine.initialize()
        assert engine.describe()["pipeline_ready"] is False and engine._extractor is None
    finally:
        engine.close()


def test_removed_card_crop_setting_cannot_reenable_crop_and_enhancement_stays_on(monkeypatch):
    monkeypatch.setenv("NAMLHN_GPU_CARD_CROP", "auto")
    monkeypatch.delenv("NAMLHN_GPU_CARD_ENHANCEMENT", raising=False)
    engine = NamLhnDocUCudaEngine()
    try:
        info = engine.describe()
        assert info["card_crop"] == "off" and info["card_enhancement"] == "adaptive"
    finally:
        engine.close()
