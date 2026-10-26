import httpx
import pytest
from fastapi.testclient import TestClient

from identification_demo.namlhn_gpu_engine import NamLhnGPUEngine
from identification_demo.web import create_app
from test_pipeline import BACK, FRONT, Engine, Reader, image_data


def submit(client, quality="medium"):
    return client.post("/api/cccd/extract", data={"quality": quality}, files={
        "front_image": ("front.png", image_data(), "image/png"),
        "back_image": ("back.png", image_data("back"), "image/png"),
    })


@pytest.mark.parametrize("quality", ["medium", "high"])
@pytest.mark.parametrize("side", ["front", "back"])
def test_main_api_reads_labelled_expiry_on_either_side_without_another_ocr_pass(quality, side):
    front, back = dict(FRONT), dict(BACK)
    target = front if side == "front" else back
    target["text_lines"] = [*target["text_lines"], "Có giá trị đến: 2036-10-25"]
    selected, unused = Engine(front, back), Engine()
    cpu, gpu = (selected, unused) if quality == "medium" else (unused, selected)
    result = submit(TestClient(create_app(cpu, Reader(), namlhn_gpu_engine=gpu)), quality).json()
    assert result["fields"]["expiry_date"] == "25/10/2036"
    assert result["field_sources"]["expiry_date"] == f"ocr_{side}"
    assert "expiry_date" in result["ocr"][side]["filled_fields"]
    assert result["total_fields"] == 12 and result["status"] == "complete"
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert selected.calls == ["front", "back"] and not unused.calls
    assert "candidates" not in result["expiry"] and "text_lines" not in str(result)


@pytest.mark.parametrize("quality", ["medium", "high"])
@pytest.mark.parametrize("rows,backup,status,expected,kind", [
    (["Ngày sinh: 01/02/1999", "25/10/2036"], [], "not_found", "", None),
    (["Có giá trị đến: 31/02/2036"], [], "not_found", "", None),
    (["Có giá trị đến: Không thời hạn"], [], "found", "", "unlimited"),
    (["Có giá trị đến: 25/10/2036"], ["Có giá trị đến: 25/10/2037"], "conflict", "", None),
])
def test_main_expiry_handles_missing_invalid_unlimited_and_conflicting_readings(quality, rows, backup, status, expected, kind):
    engine = Engine({**FRONT, "text_lines": [*FRONT["text_lines"], *rows], "paddle_text_lines": backup}, BACK)
    result = submit(TestClient(create_app(engine, Reader(), namlhn_gpu_engine=engine)), quality).json()
    assert result["fields"]["expiry_date"] == expected
    assert result["expiry"]["status"] == status and result["expiry"]["expiry_type"] == kind
    assert ("expiry_date" not in result["missing_fields"]) == (kind == "unlimited")
    assert result["field_sources"]["expiry_date"] is None
    assert result["status"] == ("complete" if kind == "unlimited" else "partial")


def test_main_expiry_rejects_conflicting_dates_between_sides():
    engine = Engine({**FRONT, "text_lines": [*FRONT["text_lines"], "Có giá trị đến: 25/10/2036"]},
                    {"text_lines": [*BACK["text_lines"], "Date of expiry: 25/10/2040"]})
    result = submit(TestClient(create_app(engine, Reader()))).json()
    assert result["fields"]["expiry_date"] == "" and result["field_sources"]["expiry_date"] is None
    assert result["expiry"]["status"] == "conflict" and result["status"] == "partial"
    assert all("expiry_date" not in stage["filled_fields"] for stage in result["ocr"].values())


def test_main_gpu_defaults_to_namlhn_worker_and_reports_its_actual_model(monkeypatch):
    monkeypatch.setenv("OCR_DEFAULT_QUALITY", "high")
    posts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"pipeline_ready": True, "cuda_available": True, "device": "cuda:0",
                                            "recognition_model": "vgg_seq2seq", "strategy": "namlhn_docu_v1"})
        side = request.url.params["doc_type_hint"].removeprefix("cccd_")
        posts.append((side, request.url.params["fields"]))
        fields = ({"nationality": "Việt Nam", "place_of_origin": "Hà Nội"} if side == "front"
                  else {"issuing_authority": "CÔNG AN TỈNH HÀ NAM"})
        return httpx.Response(200, json={"success": True, "document_side": side, "structured_fields": fields,
                                        "text_lines": ["Có giá trị đến: 25/10/2036"] if side == "front" else []})

    cpu, docu = Engine(), Engine()
    gpu = NamLhnGPUEngine("http://namlhn.test", httpx.MockTransport(handler))
    app = create_app(cpu, Reader(), namlhn_gpu_engine=gpu, docu_gpu_engine=docu)
    with TestClient(app) as client:
        response = client.post("/api/cccd/extract", files={
            "front_image": ("front.png", image_data(), "image/png"),
            "back_image": ("back.png", image_data("back"), "image/png"),
        })
        assert client.get("/api/health").json()["ocr_profiles"]["high"]["provider"] == "namlhn/vietnam-cccd-ocr"
    result = response.json()
    assert response.status_code == 200 and result["status"] == "complete"
    assert result["ocr_engine"]["provider"] == "namlhn/vietnam-cccd-ocr"
    assert result["ocr_engine"]["model"] == "vgg_seq2seq" and result["ocr_engine"]["strategy"] == "namlhn_docu_v1"
    assert result["fields"]["expiry_date"] == "25/10/2036"
    assert len(posts) == 2 and all("expiry_date" in fields for _, fields in posts)
    assert not cpu.calls and not docu.calls
