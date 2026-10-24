import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from identification_demo.engine import OCRBusy, OCRUnavailable
from identification_demo.web import FIELDS, MAX_IMAGE_BYTES, create_app


class FakeEngine:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else {name: f"TEST {name}" for _, _, name in FIELDS}
        self.error = error
        self.paths = []
        self.dimensions = None

    def describe(self):
        return {"source_available": True, "model_loaded": False}

    def extract(self, path):
        self.paths.append(Path(path))
        assert path.is_file()
        with Image.open(path) as image:
            self.dimensions = image.size
            assert image.mode == "RGB"
        if self.error:
            raise self.error
        return self.result


def image_data(fmt="PNG", size=(300, 190)):
    stream = io.BytesIO()
    Image.new("RGB", size, "white").save(stream, fmt)
    return stream.getvalue()


def submit(client, data=None, filename="card.png"):
    return client.post("/api/ocr/cccd", files={"file": (filename, image_data() if data is None else data, "image/png")})


def test_complete_result_normalizes_fields_and_removes_temporary_files():
    engine = FakeEngine()
    client = TestClient(create_app(engine))
    response = submit(client, filename="../../ignored.png")
    assert response.status_code == 200
    result = response.json()
    assert result["success"] and result["status"] == "complete"
    assert result["recognized_fields"] == 7
    assert result["fields"]["id_number"] == "TEST ID_number"
    assert result["processing_time_ms"] >= 0 and result["request_id"]
    assert response.headers["cache-control"] == "no-store"
    assert engine.paths[0].name == "card.png"
    assert not engine.paths[0].parent.exists()


@pytest.mark.parametrize("raw,expected,count", [
    ({"Name": "  TÊN KIỂM THỬ  ", "ID_number": None}, "partial", 1),
    ({}, "no_text", 0),
])
def test_missing_fields_remain_empty(raw, expected, count):
    client = TestClient(create_app(FakeEngine(raw)))
    result = submit(client).json()
    assert result["status"] == expected
    assert result["recognized_fields"] == count
    assert result["fields"]["nationality"] == ""
    assert len(result["fields"]) == 7


@pytest.mark.parametrize("data,status", [(b"", 422), (b"not an image", 422),
    (b"x" * (MAX_IMAGE_BYTES + 1), 413), (image_data("GIF"), 422)])
def test_invalid_upload_never_calls_ocr(data, status):
    engine = FakeEngine()
    response = submit(TestClient(create_app(engine)), data)
    assert response.status_code == status
    assert not response.json()["success"]
    assert not engine.paths


@pytest.mark.parametrize("error,status", [
    (OCRUnavailable("OCR chưa cài"), 503),
    (OCRBusy("OCR đang bận"), 429),
    (RuntimeError("internal details"), 503),
])
def test_provider_errors_are_actionable_and_cleanup_always_runs(error, status):
    engine = FakeEngine(error=error)
    response = submit(TestClient(create_app(engine)))
    assert response.status_code == status
    assert not response.json()["success"]
    assert "internal details" not in response.text
    assert not engine.paths[0].parent.exists()
    if status == 429:
        assert response.headers["retry-after"] == "5"


def test_exif_orientation_is_applied_before_ocr():
    image = Image.new("RGB", (300, 190), "white")
    exif = image.getexif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, "JPEG", exif=exif)
    engine = FakeEngine()
    assert submit(TestClient(create_app(engine)), stream.getvalue()).status_code == 200
    assert engine.dimensions == (190, 300)


def test_health_and_frontend_work_without_initializing_models():
    engine = FakeEngine()
    client = TestClient(create_app(engine))
    assert client.get("/api/health").json()["engine"]["model_loaded"] is False
    assert client.get("/").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/style.css").status_code == 200
    assert not engine.paths


def test_missing_file_is_rejected():
    assert TestClient(create_app(FakeEngine())).post("/api/ocr/cccd").status_code == 422


def test_unexpected_provider_output_cannot_be_reported_as_success():
    engine = FakeEngine({"Name": ["invalid"]})
    response = submit(TestClient(create_app(engine)))
    assert response.status_code == 503
    assert not response.json()["success"]


def test_back_is_automatically_parsed_into_its_own_fields():
    engine = FakeEngine({
        "Place_of_residence": "upstream front-parser fallback must not become a back field",
        "text_lines": [
            {"text": "Đặc điểm nhân dạng: Nốt ruồi má trái"},
            {"text": "Ngày cấp: 20/04/2021"},
            {"text": "CỤC TRƯỞNG CỤC CẢNH SÁT"},
            {"text": "QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"},
        ],
    })
    response = submit(TestClient(create_app(engine)))
    assert response.status_code == 200
    report = response.json()
    assert report["side"] == "back"
    assert report["total_fields"] == report["recognized_fields"] == 3
    assert report["fields"]["issue_date"] == "20/04/2021"
    assert "CỤC CẢNH SÁT" in report["fields"]["issuing_authority"]
    assert "place_of_residence" not in report["fields"]
    assert "Ngày cấp" in report["raw_text"]


def test_wrong_explicit_side_is_actionable_and_upload_is_cleaned():
    engine = FakeEngine({"text_lines": [{"text": "Đặc điểm nhân dạng"}, {"text": "Ngày cấp: 20/04/2021"}]})
    client = TestClient(create_app(engine))
    response = client.post("/api/ocr/cccd", data={"side": "front"},
                           files={"file": ("back.png", image_data(), "image/png")})
    assert response.status_code == 422
    assert "mặt sau" in response.json()["error"]
    assert not engine.paths[0].parent.exists()


def test_explicit_back_and_invalid_side():
    client = TestClient(create_app(FakeEngine({"text_lines": [{"text": "Unreadable labels"}]})))
    response = client.post("/api/ocr/cccd", data={"side": "back"},
                           files={"file": ("back.png", image_data(), "image/png")})
    assert response.status_code == 200
    assert response.json()["status"] == "unstructured"
    assert response.json()["side"] == "back"
    assert all(value == "" for value in response.json()["fields"].values())
    invalid = client.post("/api/ocr/cccd", data={"side": "invalid"},
                          files={"file": ("image.png", image_data(), "image/png")})
    assert invalid.status_code == 422
