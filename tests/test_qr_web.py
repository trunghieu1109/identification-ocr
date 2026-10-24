import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from identification_demo.qr import QRBusy, QRUnavailable
from identification_demo.wechat_qr import WeChatQRReader
from identification_demo.web import MAX_IMAGE_BYTES, create_app, process_qr_image


PAYLOAD = "001099000001|012345678|NGUYỄN VĂN THỬ|01021999|Nam|Phường Thử, Hà Nội|15032021"


class NoOCREngine:
    def describe(self):
        return {"source_available": False, "model_loaded": False}

    def extract(self, path):
        raise AssertionError("Reading QR must never initialize OCR")


class FakeReader:
    def __init__(self, texts=None, error=None):
        self.texts = [PAYLOAD] if texts is None else texts
        self.error = error
        self.dimensions = None

    def describe(self):
        return {"available": True}

    def read(self, image):
        assert image.mode == "RGB"
        self.dimensions = image.size
        if self.error:
            raise self.error
        return self.texts


def image_data(fmt="PNG", size=(300, 190), exif=None):
    stream = io.BytesIO()
    Image.new("RGB", size, "white").save(stream, fmt, **({"exif": exif} if exif else {}))
    return stream.getvalue()


def submit(client, data=None):
    return client.post("/api/qr/cccd", files={"file": ("../../card.png", image_data() if data is None else data, "image/png")})


def test_qr_endpoint_works_without_ocr_and_does_not_create_files(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_UPLOAD_TMP_DIR", str(tmp_path))
    reader = FakeReader()
    client = TestClient(create_app(NoOCREngine(), reader))
    response = submit(client)
    assert response.status_code == 200
    result = response.json()
    assert result["source"] == "qr" and result["status"] == "complete"
    assert result["fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert result["fields"]["id_number"] == "001099000001"
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert result["recognized_fields"] == result["total_fields"] == 7
    assert {"raw_qr", "qr_codes", "raw_text"}.isdisjoint(result)
    assert PAYLOAD not in response.text
    assert result["request_id"] and result["processing_time_ms"] >= 0
    assert response.headers["cache-control"] == "no-store"
    assert not list(tmp_path.iterdir())
    health = client.get("/api/health").json()
    assert health["qr"]["available"] and not health["engine"]["model_loaded"]


@pytest.mark.parametrize("texts,status,count", [
    ([], "no_qr", 0), (["https://example.com"], "unsupported_qr", 0),
    ([PAYLOAD, "https://example.com"], "multiple_qr", 0),
    ([PAYLOAD.replace("012345678", "")], "complete", 6),
    ([PAYLOAD.replace("01021999", "31021999")], "partial", 6),
])
def test_qr_outcomes_omit_raw_content_and_do_not_mix_cards(texts, status, count):
    result = process_qr_image(image_data(), FakeReader(texts))
    assert result["status"] == status and result["recognized_fields"] == count
    assert {"raw_qr", "qr_codes", "raw_text"}.isdisjoint(result)
    assert "issue_date" in result["fields"]
    assert result["message"]


@pytest.mark.parametrize("data,status", [
    (b"", 422), (b"not an image", 422), (image_data("GIF"), 422),
    (b"x" * (MAX_IMAGE_BYTES + 1), 413),
    (image_data(size=(5000, 4001)), 422),
])
def test_invalid_uploads_never_reach_the_qr_reader(data, status):
    reader = FakeReader()
    response = submit(TestClient(create_app(NoOCREngine(), reader)), data)
    assert response.status_code == status
    assert not response.json()["success"] and reader.dimensions is None


@pytest.mark.parametrize("error,status", [
    (QRUnavailable("Chưa cài bộ đọc QR"), 503),
    (QRBusy("Đang bận"), 429), (RuntimeError("private decoder details"), 503),
])
def test_qr_errors_have_clear_http_status_and_do_not_leak_details(error, status):
    response = submit(TestClient(create_app(NoOCREngine(), FakeReader(error=error))))
    assert response.status_code == status and not response.json()["success"]
    assert "private decoder details" not in response.text
    if status == 429:
        assert response.headers["retry-after"] == "5"


def test_qr_exif_orientation_is_applied():
    exif = Image.Exif()
    exif[274] = 6
    reader = FakeReader()
    process_qr_image(image_data("JPEG", exif=exif), reader)
    assert reader.dimensions == (190, 300)


def test_qr_missing_upload_is_rejected():
    assert TestClient(create_app(NoOCREngine(), FakeReader())).post("/api/qr/cccd").status_code == 422


@pytest.mark.parametrize("rotation,fmt", [(0, "PNG"), (90, "PNG"), (0, "JPEG")])
def test_real_qr_decoder_extracts_generated_unicode_code(rotation, fmt):
    if not WeChatQRReader().describe()["available"]:
        pytest.skip("WeChatQRCode dependencies and models are required")
    qrcode = pytest.importorskip("qrcode")
    # This integration uses generated QR pixels, never a real person's card.
    qr = qrcode.make(PAYLOAD, box_size=6, border=4).convert("RGB")
    card = Image.new("RGB", (1400, 900), "#edf2ed")
    card.paste(qr, (card.width - qr.width - 60, 100))
    card = card.rotate(rotation, expand=True)
    stream = io.BytesIO()
    card.save(stream, fmt)
    response = submit(TestClient(create_app(NoOCREngine())), stream.getvalue())
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "complete"
    assert result["fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert {"raw_qr", "qr_codes", "raw_text"}.isdisjoint(result)


@pytest.mark.parametrize("encoded,expected,status", [
    ("15032021", "15/03/2021", "complete"),
    ("15/03/2021", "15/03/2021", "complete"),
    ("31022021", "", "partial"),
    ("", "", "partial"),
])
def test_qr_issue_date_is_always_returned_without_raw_payload(encoded, expected, status):
    payload = PAYLOAD.rsplit("|", 1)[0] + "|" + encoded
    client = TestClient(create_app(NoOCREngine(), FakeReader([payload])))
    response = submit(client)
    assert response.status_code == 200
    result = response.json()
    assert result["fields"]["issue_date"] == expected
    assert result["status"] == status
    assert {"raw_qr", "qr_codes", "raw_text"}.isdisjoint(result)
    assert payload not in response.text
