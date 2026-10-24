import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from identification_demo.engine import OCRBusy, OCRUnavailable
from identification_demo.pipeline import CARD_FIELDS
from identification_demo.qr import QRBusy, QRUnavailable
from identification_demo.web import MAX_IMAGE_BYTES, create_app


PAYLOAD = "001099000001|012345678|NGUYỄN VĂN THỬ|01021999|Nam|Phường Thử, Hà Nội|15032021"
FRONT = {
    "ID_number": "999999999999", "Name": "OCR KHÁC QR", "Date_of_birth": "02/02/2001",
    "Gender": "Nữ", "Nationality": "Việt Nam", "Place_of_origin": "Hà Nội",
    "Place_of_residence": "Địa chỉ OCR", "text_lines": ["Họ và tên", "Ngày sinh"],
}
BACK = {"text_lines": [
    "Đặc điểm nhân dạng: Nốt ruồi má trái", "Ngày cấp: 20/04/2021",
    "CỤC TRƯỞNG CỤC CẢNH SÁT", "QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI",
]}


def image_data(side="front", fmt="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (300 if side == "front" else 320, 190), "white").save(stream, fmt)
    return stream.getvalue()


class Engine:
    def __init__(self, front=None, back=None):
        self.results = {"front": FRONT if front is None else front, "back": BACK if back is None else back}
        self.paths = []
        self.calls = []

    def describe(self):
        return {"source_available": True, "model_loaded": False}

    def extract(self, path):
        self.paths.append(Path(path))
        with Image.open(path) as image:
            side = "front" if image.width == 300 else "back"
        self.calls.append(side)
        value = self.results[side]
        if isinstance(value, Exception):
            raise value
        return value


class Reader:
    def __init__(self, front=None, back=None):
        self.results = {"front": [PAYLOAD] if front is None else front, "back": [] if back is None else back}
        self.calls = []

    def describe(self):
        return {"available": True}

    def read(self, image):
        side = "front" if image.width == 300 else "back"
        self.calls.append(side)
        value = self.results[side]
        if isinstance(value, Exception):
            raise value
        return value


def submit(engine=None, reader=None, front=None, back=None):
    client = TestClient(create_app(engine or Engine(), reader or Reader()))
    return client.post("/api/cccd/extract", files={
        "front_image": ("../../front.png", image_data() if front is None else front, "image/png"),
        "back_image": ("../../back.png", image_data("back") if back is None else back, "image/png"),
    })


def test_qr_wins_and_ocr_only_fills_gaps_in_combined_response(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_UPLOAD_TMP_DIR", str(tmp_path))
    engine, reader = Engine(), Reader()
    response = submit(engine, reader)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "complete" and result["source"] == "qr_ocr"
    assert list(result["fields"]) == list(CARD_FIELDS)
    assert result["fields"]["id_number"] == "001099000001"
    assert result["fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert result["fields"]["gender"] == "Nam"
    assert result["fields"]["nationality"] == "Việt Nam"
    assert "CỤC CẢNH SÁT" in result["fields"]["issuing_authority"]
    assert result["field_sources"]["name"] == "qr_front"
    assert result["field_sources"]["nationality"] == "ocr_front"
    assert result["field_sources"]["issuing_authority"] == "ocr_back"
    assert result["ocr"]["front"]["filled_fields"] == ["nationality", "place_of_origin"]
    assert result["ocr"]["back"]["filled_fields"] == ["issuing_authority", "identifying_features"]
    assert result["recognized_fields"] == result["total_fields"] == 11
    assert result["missing_fields"] == result["warnings"] == []
    assert result["request_id"] and result["processing_time_ms"] >= 0
    assert response.headers["cache-control"] == "no-store"
    assert {"raw_text", "raw_qr", "qr_codes"}.isdisjoint(result)
    assert PAYLOAD not in response.text and "text_lines" not in response.text
    assert reader.calls == engine.calls == ["front", "back"]
    assert all(not path.parent.exists() for path in engine.paths)
    assert not list(tmp_path.iterdir())


def test_qr_can_be_on_back_and_identical_codes_are_deduplicated():
    result = submit(reader=Reader([], [PAYLOAD])).json()
    assert result["field_sources"]["issue_date"] == "qr_back"
    assert result["qr"]["sides"] == {"front": "no_qr", "back": "complete"}
    repeated = submit(reader=Reader([PAYLOAD, PAYLOAD], [PAYLOAD])).json()
    assert repeated["recognized_fields"] == 11 and repeated["qr"]["recognized_fields"] == 7


@pytest.mark.parametrize("date", ["", "31022021"])
def test_missing_or_invalid_qr_issue_date_is_filled_from_back_ocr(date):
    partial = PAYLOAD.rsplit("|", 1)[0] + "|" + date
    result = submit(reader=Reader([partial])).json()
    assert result["qr"]["status"] == "partial"
    assert result["fields"]["issue_date"] == "20/04/2021"
    assert result["field_sources"]["issue_date"] == "ocr_back"
    assert result["fields"]["name"] == "NGUYỄN VĂN THỬ"


def test_partial_qr_personal_fields_are_filled_by_front_ocr():
    partial = PAYLOAD.replace("01021999", "31021999").replace("|Nam|", "||").replace("NGUYỄN VĂN THỬ", "")
    result = submit(reader=Reader([partial])).json()
    assert result["fields"]["date_of_birth"] == FRONT["Date_of_birth"]
    assert result["fields"]["gender"] == FRONT["Gender"]
    assert result["field_sources"]["date_of_birth"] == "ocr_front"
    assert result["field_sources"]["name"] == "ocr_front"
    assert result["fields"]["name"] == FRONT["Name"]
    assert result["fields"]["id_number"] == "001099000001"


@pytest.mark.parametrize("codes,qr_status", [([], "no_qr"), (["https://example.com"], "unsupported_qr"), (["bad|layout"], "unsupported_qr")])
def test_no_usable_qr_falls_back_to_both_ocr_sides(codes, qr_status):
    result = submit(reader=Reader(codes)).json()
    assert result["status"] == "complete"
    assert result["qr"]["status"] == qr_status
    assert result["fields"]["name"] == FRONT["Name"]
    assert result["fields"]["issue_date"] == "20/04/2021"
    assert result["fields"]["old_id_number"] == ""
    assert result["field_sources"]["old_id_number"] is None
    assert result["recognized_fields"] == 10 and result["missing_fields"] == []
    assert result["warnings"]


@pytest.mark.parametrize("error", [QRUnavailable("not installed"), QRBusy("busy"), RuntimeError("private decoder error")])
def test_qr_decoder_errors_do_not_block_ocr_fallback(error):
    response = submit(reader=Reader(error, error))
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "complete" and result["qr"]["status"] == "error"
    assert result["field_sources"]["name"] == "ocr_front"
    assert "private decoder error" not in response.text


def test_second_side_qr_is_used_when_first_side_decoder_fails():
    result = submit(reader=Reader(QRUnavailable("error"), [PAYLOAD])).json()
    assert result["fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert result["field_sources"]["name"] == "qr_back"
    assert result["warnings"]


@pytest.mark.parametrize("error", [OCRUnavailable("missing model"), OCRBusy("busy"), RuntimeError("private OCR details")])
def test_ocr_errors_keep_successful_qr_and_other_side_results(error):
    engine = Engine(back=error)
    response = submit(engine)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "partial"
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert result["fields"]["nationality"] == "Việt Nam"
    assert result["fields"]["issuing_authority"] == ""
    assert result["ocr"]["back"]["status"] == "error"
    assert result["missing_fields"] == ["issuing_authority", "identifying_features"]
    assert "private OCR details" not in response.text
    assert all(not path.parent.exists() for path in engine.paths)


def test_front_ocr_error_still_allows_back_ocr_without_qr():
    result = submit(Engine(front=OCRUnavailable("failure")), Reader([])).json()
    assert result["status"] == "partial"
    assert result["fields"]["name"] == "" and result["fields"]["issue_date"] == "20/04/2021"


@pytest.mark.parametrize("error,status", [(OCRBusy("busy"), 429), (OCRUnavailable("failed"), 503), (RuntimeError("private"), 503)])
def test_all_failed_without_data_returns_service_error(error, status):
    response = submit(Engine(error, error), Reader([]))
    assert response.status_code == status
    assert not response.json()["success"]
    if status == 429:
        assert response.headers["retry-after"] == "5"


def test_no_text_is_a_processed_result_with_empty_fields():
    result = submit(Engine({}, {}), Reader([])).json()
    assert result["status"] == "no_text" and result["recognized_fields"] == 0
    assert all(value == "" for value in result["fields"].values())
    assert all(value is None for value in result["field_sources"].values())


def test_optional_old_cmnd_does_not_prevent_complete_status():
    result = submit(reader=Reader([PAYLOAD.replace("012345678", "")])).json()
    assert result["status"] == "complete" and result["recognized_fields"] == 10
    assert result["fields"]["old_id_number"] == "" and result["missing_fields"] == []


def test_complementary_qr_values_for_same_card_can_be_combined():
    result = submit(reader=Reader([PAYLOAD.replace("15032021", "")], [PAYLOAD.replace("01021999", "")])).json()
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert result["field_sources"]["issue_date"] == "qr_back"
    assert result["field_sources"]["date_of_birth"] == "qr_front"


@pytest.mark.parametrize("second", [PAYLOAD.replace("001099000001", "001099000002"), PAYLOAD.replace("VĂN THỬ", "VĂN KHÁC")])
def test_conflicting_qr_values_are_rejected_before_ocr(second):
    engine = Engine()
    response = submit(engine, Reader([PAYLOAD], [second]))
    assert response.status_code == 422 and "không khớp" in response.json()["error"]
    assert not engine.calls and PAYLOAD not in response.text
    multiple = submit(engine, Reader([PAYLOAD, second]))
    assert multiple.status_code == 422 and not engine.calls


@pytest.mark.parametrize("side", ["front", "back"])
@pytest.mark.parametrize("data,status", [(b"", 422), (b"bad image", 422), (image_data(fmt="GIF"), 422), (b"x" * (MAX_IMAGE_BYTES + 1), 413)])
def test_both_uploads_are_validated_before_any_inference(side, data, status):
    engine, reader = Engine(), Reader()
    response = submit(engine, reader, **{side: data})
    assert response.status_code == status and not response.json()["success"]
    assert not reader.calls and not engine.calls


@pytest.mark.parametrize("present", [None, "front_image", "back_image"])
def test_both_file_fields_are_required(present):
    files = {present: ("image.png", image_data(), "image/png")} if present else {}
    response = TestClient(create_app(Engine(), Reader())).post("/api/cccd/extract", files=files)
    assert response.status_code == 422


def test_swapped_sides_return_actionable_error():
    engine = Engine(front=BACK)
    response = submit(engine)
    assert response.status_code == 422 and "Mặt trước" in response.json()["error"]
    assert "mặt sau" in response.json()["error"]
    assert all(not path.parent.exists() for path in engine.paths)


def test_real_qr_decoder_and_combined_api():
    pytest.importorskip("zxingcpp")
    qrcode = pytest.importorskip("qrcode")
    stream = io.BytesIO()
    qrcode.make(PAYLOAD).save(stream, "PNG")
    class OrderedEngine:
        def __init__(self):
            self.results = iter([FRONT, BACK])

        def describe(self):
            return {}

        def extract(self, path):
            return next(self.results)

    client = TestClient(create_app(OrderedEngine()))
    response = client.post("/api/cccd/extract", files={
        "front_image": ("qr.png", stream.getvalue(), "image/png"),
        "back_image": ("back.png", image_data("back"), "image/png"),
    })
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "complete"
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert result["field_sources"]["name"] == "qr_front"
    assert result["field_sources"]["issuing_authority"] == "ocr_back"
