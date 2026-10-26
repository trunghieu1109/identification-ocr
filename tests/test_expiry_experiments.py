"""Expiry extraction and its isolated HTTP contract, without model weights."""
import httpx
import pytest
from fastapi.testclient import TestClient

from identification_demo.engine import OCRBusy, OCRUnavailable
from identification_demo.expiry_experiments import parse_expiry
from identification_demo.gpu_engine import DocUGPUEngine
from identification_demo.pipeline import CARD_FIELDS
from identification_demo.web import MAX_IMAGE_BYTES, create_app
from test_gpu_engine import READY
from test_pipeline import Engine, Reader, image_data

ENDPOINT = "/api/cccd/expiry/experiment"


def submit(client, quality="medium", front=None, back=None):
    return client.post(ENDPOINT, data={"quality": quality} if quality else {}, files={
        "front_image": ("../../front.png", image_data() if front is None else front, "image/png"),
        "back_image": ("../../back.png", image_data("back") if back is None else back, "image/png"),
    })


@pytest.mark.parametrize("rows,expected", [
    (["Có giá trị đến: 5/2/2036"], "05/02/2036"),
    (["Ngày hết hạn / Date of expiry", "25-10-2036"], "25/10/2036"),
    (["Có giá trị", "đến / Date of expiry:", "25.10.2036"], "25/10/2036"),
    (["VALID UNTIL: 2036-10-25"], "25/10/2036"),
    (["Date of expiration: 29/02/2024"], "29/02/2024"),
    (["Hạn sử dụng: ngày 25 tháng 10 năm 2036"], "25/10/2036"),
    (["Ngày sinh: 25/10/1990 Có giá trị đến: 25/10/2036 Ngày cấp: 25/10/2026"], "25/10/2036"),
])
def test_labelled_dates_are_normalized_and_keep_evidence(rows, expected):
    found = parse_expiry(rows)
    assert found and {item["expiry_date"] for item in found} == {expected}
    assert all(item["expiry_type"] == "date" and item["evidence"] for item in found)


@pytest.mark.parametrize("rows", [
    ["Ngày sinh: 25/10/1990", "Ngày cấp: 25/10/2026"],
    ["25/10/2036"],
    ["Có giá trị đến", "Ngày sinh: 25/10/1990"],
    ["Có giá trị đến", "Date of issue", "25/10/2026"],
    ["Có giá trị đến", "Ngày, tháng, năm", "25/10/2026"],
    ["Có giá trị đến: 31/02/2036"],
    ["Date of expiry: 2036-13-25"],
    ["Số CCCD: 001099000001"],
])
def test_birth_issue_unlabelled_and_invalid_dates_are_not_expiry(rows):
    assert parse_expiry(rows) == []


@pytest.mark.parametrize("text", ["Có giá trị đến: Không thời hạn", "KHÔNG THỜI HẠN", "Valid until: No expiry"])
def test_explicit_unlimited_is_distinct_from_missing(text):
    found = parse_expiry([text])
    assert len(found) == 1 and found[0]["expiry_type"] == "unlimited" and found[0]["expiry_date"] == ""


@pytest.mark.parametrize("side", ["front", "back"])
def test_api_reads_both_sides_and_returns_the_actual_source(side, tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_UPLOAD_TMP_DIR", str(tmp_path))
    raw = {"text_lines": ["Có giá trị đến: 25/10/2036"]}
    engine = Engine(front=raw if side == "front" else {}, back=raw if side == "back" else {})
    qr = Reader()
    response = submit(TestClient(create_app(engine, qr)))
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "complete" and result["experiment"] == "expiry_date"
    assert result["fields"] == {"expiry_date": "25/10/2036"}
    assert result["expiry_type"] == "date" and result["sources"] == [f"ocr_{side}"]
    assert result["field_sources"] == {"expiry_date": f"ocr_{side}"}
    assert engine.calls == ["front", "back"] and not qr.calls
    assert result["ocr"][side]["text_lines"][0]["is_expiry"]
    assert response.headers["cache-control"] == "no-store" and result["request_id"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("front,back,status,value,kind", [
    ("Có giá trị đến: 25/10/2036", "Date of expiry: 25/10/2036", "complete", "25/10/2036", "date"),
    ("Có giá trị đến: 25/10/2036", "Date of expiry: 25/10/2037", "conflict", "", None),
    ("Có giá trị đến: Không thời hạn", "Date of expiry: 25/10/2037", "conflict", "", None),
    ("Không thời hạn", "", "complete", "", "unlimited"),
    ("Ngày sinh: 25/10/1990", "Ngày cấp: 25/10/2026", "not_found", "", None),
])
def test_consensus_conflict_and_absence_have_different_results(front, back, status, value, kind):
    engine = Engine({"text_lines": [front]}, {"text_lines": [back]})
    result = submit(TestClient(create_app(engine, Reader()))).json()
    assert result["status"] == status and result["fields"]["expiry_date"] == value
    assert result["expiry_type"] == kind
    if status == "conflict":
        assert result["field_sources"]["expiry_date"] is None and result["warnings"]
        assert len(result["candidates"]) == 2


def test_disagreement_between_primary_and_backup_ocr_is_visible():
    raw = {"text_lines": ["Có giá trị đến: 25/10/2036"], "paddle_text_lines": ["Có giá trị đến: 25/10/2037"]}
    result = submit(TestClient(create_app(Engine(raw, {}), Reader()))).json()
    assert result["status"] == "conflict"
    assert {candidate["reading"] for candidate in result["candidates"]} == {"primary", "backup"}


def test_a_failed_side_preserves_the_other_side_without_leaking_internal_errors():
    engine = Engine(RuntimeError("secret details"), {"text_lines": ["Có giá trị đến: 25/10/2036"]})
    response = submit(TestClient(create_app(engine, Reader())))
    assert response.status_code == 200 and "secret details" not in response.text
    result = response.json()
    assert result["status"] == "partial" and result["fields"]["expiry_date"] == "25/10/2036"
    assert result["ocr"]["front"]["status"] == "error" and result["sources"] == ["ocr_back"]


@pytest.mark.parametrize("error,status", [(OCRBusy("busy"), 429), (OCRUnavailable("unavailable"), 503)])
def test_all_failed_reads_have_an_actionable_http_status(error, status):
    response = submit(TestClient(create_app(Engine(error, error), Reader())))
    assert response.status_code == status and not response.json()["success"]
    if status == 429:
        assert response.headers["retry-after"] == "5"


@pytest.mark.parametrize("data,status", [(b"", 422), (b"invalid", 422), (b"x" * (MAX_IMAGE_BYTES + 1), 413)])
def test_invalid_second_image_is_rejected_before_any_ocr(data, status):
    engine = Engine()
    response = submit(TestClient(create_app(engine, Reader())), back=data)
    assert response.status_code == status and not engine.calls


def test_gpu_default_uses_both_sides_without_sending_new_production_fields(monkeypatch):
    monkeypatch.setenv("OCR_DEFAULT_QUALITY", "high")
    posts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=READY)
        assert "fields" not in request.url.params
        side = request.url.params["doc_type_hint"]
        posts.append(side)
        return httpx.Response(200, json={"success": True, "doc_type": side, "extracted_fields": {},
                                        "text_lines": ["Có giá trị đến: 25/10/2036"] if side == "cccd_back" else []})

    cpu, qr = Engine(), Reader()
    gpu = DocUGPUEngine("http://gpu.test", transport=httpx.MockTransport(handler))
    result = submit(TestClient(create_app(cpu, qr, gpu)), quality=None).json()
    assert result["fields"]["expiry_date"] == "25/10/2036" and result["quality"] == "high"
    assert posts == ["cccd_front", "cccd_back"] and not cpu.calls and not qr.calls


def test_missing_files_and_unknown_quality_are_rejected():
    engine = Engine()
    client = TestClient(create_app(engine, Reader()))
    assert client.post(ENDPOINT).status_code == 422
    assert submit(client, quality="invalid").status_code == 422
    assert not engine.calls


def test_main_api_also_returns_expiry():
    client = TestClient(create_app(Engine(), Reader()))
    result = client.post("/api/cccd/extract", files={
        "front_image": ("front.png", image_data(), "image/png"),
        "back_image": ("back.png", image_data("back"), "image/png"),
    }).json()
    assert set(result["fields"]) == set(CARD_FIELDS) and len(result["fields"]) == 12
    assert result["fields"]["expiry_date"] == "25/10/2036" and result["total_fields"] == 12
