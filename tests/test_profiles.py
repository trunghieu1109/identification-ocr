import httpx
import pytest
from fastapi.testclient import TestClient

from identification_demo.gpu_engine import DocUGPUEngine
from identification_demo.web import create_app
from test_pipeline import Engine, Reader, PAYLOAD, image_data
from test_gpu_engine import READY, FRONT


def submit(client, quality=None):
    data = {"quality": quality} if quality is not None else {}
    return client.post("/api/cccd/extract", data=data, files={
        "front_image": ("front.png", image_data(), "image/png"),
        "back_image": ("back.png", image_data("back"), "image/png"),
    })


def test_default_and_explicit_medium_only_use_cpu():
    class NoGPU:
        def extract_document(self, *_):
            raise AssertionError("Medium must not select GPU")
    for quality in (None, "medium"):
        cpu = Engine()
        result = submit(TestClient(create_app(cpu, Reader(), NoGPU())), quality).json()
        assert result["quality"] == "medium" and cpu.calls == ["front", "back"]
        assert result["ocr_engine"]["provider"] == "namlhn/vietnam-cccd-ocr"


def test_gpu_default_never_warms_or_calls_cpu_and_still_prioritizes_qr(monkeypatch):
    monkeypatch.setenv("OCR_DEFAULT_QUALITY", "high")

    class NoCPU(Engine):
        def initialize(self):
            pytest.fail("GPU default must not warm the CPU model")

        def extract(self, _path):
            pytest.fail("All requested OCR fields must go to the GPU")

    gpu = Engine()
    with TestClient(create_app(NoCPU(), Reader(), gpu)) as client:
        result = submit(client).json()
        assert client.get("/api/health").json()["default_quality"] == "high"
    assert gpu.calls == ["front", "back"]
    assert result["quality"] == "high" and result["status"] == "complete"
    assert result["fields"]["issuing_authority"] and result["field_sources"]["issuing_authority"] == "ocr_back"
    assert result["fields"]["issue_date"] == "15/03/2021" and result["field_sources"]["issue_date"] == "qr_front"


def test_gpu_default_unavailable_preserves_qr_without_falling_back_to_cpu(monkeypatch):
    monkeypatch.setenv("OCR_DEFAULT_QUALITY", "high")
    cpu = Engine()
    result = submit(TestClient(create_app(cpu, Reader(), DocUGPUEngine("")))).json()
    assert result["quality"] == "high" and result["status"] == "partial"
    assert result["field_sources"]["issue_date"] == "qr_front"
    assert not cpu.calls


def test_high_selects_gpu_metadata_and_retains_qr_priority():
    posts = []
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=READY)
        side = request.url.params["doc_type_hint"]
        posts.append(side)
        payload = FRONT if side == "cccd_front" else {"success": True, "doc_type": "cccd_back", "extracted_fields": {
            "issue_date": "2021-04-20", "issue_authority": "WRONG DEFAULT", "characteristics": "Nốt ruồi má trái",
        }, "text_lines": [{"text": "Ngày cấp: 20/04/2021"}, {"text": "Cơ quan cấp: Công an tỉnh Hà Nam"}]}
        return httpx.Response(200, json=payload)
    cpu = Engine()
    gpu = DocUGPUEngine("http://gpu.test", transport=httpx.MockTransport(handler))
    client = TestClient(create_app(cpu, Reader(), gpu))
    response = submit(client, "high")
    assert response.status_code == 200
    result = response.json()
    assert result["quality"] == "high" and result["status"] == "complete"
    assert result["ocr_engine"] == {"provider": "CaoAnhNato/vn-identity-card-ocr", "device": "cuda", "used": True}
    assert result["fields"]["issue_date"] == "15/03/2021" and result["field_sources"]["issue_date"] == "qr_front"
    assert result["fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert result["fields"]["issuing_authority"] == "Công an tỉnh Hà Nam"
    assert result["field_sources"]["issuing_authority"] == "ocr_back"
    assert result["field_sources"]["nationality"] == "ocr_front"
    assert not cpu.calls and posts == ["cccd_front", "cccd_back"]
    assert {"raw_text", "raw_qr", "qr_codes"}.isdisjoint(result)
    assert "WRONG DEFAULT" not in response.text and PAYLOAD not in response.text
    assert client.get("/api/health").json()["ocr_profiles"]["high"]["available"]


def test_high_without_qr_fills_all_fields_from_gpu_and_never_cpu():
    cpu, gpu = Engine(), Engine()
    result = submit(TestClient(create_app(cpu, Reader([]), gpu)), "high").json()
    assert result["quality"] == "high" and result["status"] == "complete"
    assert result["field_sources"]["name"] == "ocr_front"
    assert result["qr"]["status"] == "no_qr"
    assert gpu.calls == ["front", "back"] and not cpu.calls


def test_partial_gpu_institution_stays_visible_in_common_api_with_qr_priority():
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=READY)
        payload = FRONT if request.url.params["doc_type_hint"] == "cccd_front" else {
            "success": True, "doc_type": "cccd_back", "extracted_fields": {"issue_authority": "WRONG DEFAULT"},
            "text_lines": [{"text": "Ngày cấp: 20/04/2021"}, {"text": "CỤC TRƯỞNG CỤC CẢNH SÁT"}],
        }
        return httpx.Response(200, json=payload)

    cpu = Engine()
    gpu = DocUGPUEngine("http://gpu.test", transport=httpx.MockTransport(handler))
    response = submit(TestClient(create_app(cpu, Reader(), gpu)), "high")
    assert response.status_code == 200
    result = response.json()
    assert result["fields"]["issuing_authority"] == "CỤC CẢNH SÁT"
    assert result["field_sources"]["issuing_authority"] == "ocr_back"
    assert result["ocr"]["back"]["authority_region"]["status"] == result["status"] == "partial"
    assert not result["missing_fields"]
    assert any("Cơ quan cấp mới đọc được một phần" in warning for warning in result["warnings"])
    assert result["fields"]["issue_date"] == "15/03/2021" and result["field_sources"]["issue_date"] == "qr_front"
    assert not cpu.calls and "WRONG DEFAULT" not in response.text and "raw_text" not in response.text


def test_gpu_unavailable_keeps_qr_with_explicit_errors_and_does_not_use_cpu():
    cpu = Engine()
    result = submit(TestClient(create_app(cpu, Reader(), DocUGPUEngine(""))), "high").json()
    assert result["status"] == "partial" and result["quality"] == "high"
    assert result["ocr_engine"]["used"] is False
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert result["fields"]["issuing_authority"] == ""
    assert "GPU" in result["ocr"]["back"]["error"]
    assert result["warnings"] and not cpu.calls


def test_gpu_unavailable_without_any_qr_returns_503_and_does_not_use_cpu():
    cpu = Engine()
    response = submit(TestClient(create_app(cpu, Reader([]), DocUGPUEngine(""))), "high")
    assert response.status_code == 503 and not cpu.calls


@pytest.mark.parametrize("quality", ["low", "gpu", "HIGH", "cpu", "invalid"])
def test_unknown_quality_is_rejected_before_reading_images(quality):
    cpu, gpu, qr = Engine(), Engine(), Reader()
    response = submit(TestClient(create_app(cpu, qr, gpu)), quality)
    assert response.status_code == 422
    assert not cpu.calls and not gpu.calls and not qr.calls


def test_swapped_gpu_sides_are_rejected():
    def handler(request):
        return httpx.Response(200, json=READY if request.method == "GET" else {"success": True, "doc_type": "cccd_back", "extracted_fields": {}})
    gpu = DocUGPUEngine("http://gpu.test", transport=httpx.MockTransport(handler))
    response = submit(TestClient(create_app(Engine(), Reader(), gpu)), "high")
    assert response.status_code == 422 and "mặt sau" in response.json()["error"]


def test_ambiguous_back_regions_keep_qr_and_successful_front_ocr():
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=READY)
        if request.url.params["doc_type_hint"] == "cccd_front":
            return httpx.Response(200, json=FRONT)
        return httpx.Response(422, json={"error_code": "multiple_documents", "detected_regions": 2})
    gpu = DocUGPUEngine("http://gpu.test", transport=httpx.MockTransport(handler))
    cpu = Engine()
    response = submit(TestClient(create_app(cpu, Reader(), gpu)), "high")
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "partial"
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert result["field_sources"]["issue_date"] == "qr_front"
    assert result["fields"]["nationality"] == "Việt Nam"
    assert result["field_sources"]["nationality"] == "ocr_front"
    assert result["fields"]["issuing_authority"] == ""
    assert result["ocr"]["back"]["error_code"] == "multiple_documents"
    assert result["ocr"]["back"]["detected_regions"] == 2
    assert "nghi là thẻ" in result["ocr"]["back"]["error"]
    assert not cpu.calls


def test_ambiguous_regions_on_both_sides_without_any_data_return_actionable_422():
    def handler(request):
        return httpx.Response(200, json=READY) if request.method == "GET" else httpx.Response(422, json={
            "error_code": "multiple_documents", "detected_regions": 2,
        })
    gpu = DocUGPUEngine("http://gpu.test", transport=httpx.MockTransport(handler))
    response = submit(TestClient(create_app(Engine(), Reader([]), gpu)), "high")
    assert response.status_code == 422
    assert "vùng thẻ duy nhất" in response.json()["error"]
