import httpx
import pytest
from PIL import Image

from identification_demo.engine import OCRBusy, OCRUnavailable, OCRMultipleDocuments
from identification_demo.namlhn_gpu_engine import NamLhnGPUEngine
from identification_demo.processing import InvalidImage


READY = {"pipeline_ready": True, "device": "cuda:0", "cuda_available": True,
         "strategy": "namlhn_docu_v1", "recognition_model": "vgg_seq2seq"}
FRONT = {"success": True, "document_side": "front", "structured_fields": {
    "id_number": "001099000001", "name": "NGUYỄN VĂN THỬ", "date_of_birth": "01/02/1999",
    "gender": "Nam", "nationality": "Việt Nam", "place_of_origin": "Hà Nội",
    "place_of_residence": "Địa chỉ thử",
}, "text_lines": [{"text": "Họ và tên"}, {"text": "Ngày sinh"},
                   {"text": "Giới tính: Nam"}, {"text": "Quốc tịch: Việt Nam"}]}


def engine(handler):
    return NamLhnGPUEngine("http://gpu.test", transport=httpx.MockTransport(handler))


def test_http_adapter_preserves_worker_fields_and_layout_with_authenticated_multipart(monkeypatch):
    monkeypatch.setenv("NAMLHN_GPU_OCR_API_KEY", "test-token")
    requests = []
    def handler(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(200, json=READY if request.method == "GET" else FRONT)
    reader = engine(handler)
    with Image.new("RGB", (20, 20)) as image:
        result = reader.extract_image(image, "front", ["name", "expiry_date"], b"image bytes")
    assert result["structured_fields"] == FRONT["structured_fields"]
    assert result["text_lines"] == FRONT["text_lines"]
    assert [request.url.path for request in requests] == ["/api/v1/health/ready", "/api/v1/extract"]
    assert requests[1].url.params["doc_type_hint"] == "cccd_front"
    assert requests[1].url.params["fields"] == "name,expiry_date"
    assert b'name="file"' in requests[1].content and b"image bytes" in requests[1].content
    assert reader.describe()["available"]
    reader.close()


@pytest.mark.parametrize("ready", [
    {**READY, "pipeline_ready": False}, {**READY, "device": "cpu"},
    {**READY, "cuda_available": False}, {**READY, "strategy": "docu_regions_v2"},
    {**READY, "recognition_model": "vgg_transformer"}, {},
])
def test_high_mode_refuses_cpu_or_retired_workers_before_sending_an_image(ready):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=ready)
    reader = engine(handler)
    with pytest.raises(OCRUnavailable), Image.new("RGB", (20, 20)) as image:
        reader.extract_image(image, "front")
    assert all(request.method == "GET" for request in requests)
    assert reader.describe()["available"] is False
    reader.close()


@pytest.mark.parametrize("payload", [None, [], {"success": False}, {"success": True},
    {"success": True, "document_side": "front", "structured_fields": []},
    {"success": True, "document_side": "front", "structured_fields": {"name": ["bad"]}},
    {"success": True, "document_side": "gplx", "structured_fields": {}},
])
def test_invalid_worker_payloads_never_become_valid_ocr(payload):
    reader = engine(lambda request: httpx.Response(200, json=READY if request.method == "GET" else payload))
    with pytest.raises(OCRUnavailable), Image.new("RGB", (20, 20)) as image:
        reader.extract_image(image, "front")
    reader.close()


@pytest.mark.parametrize("status,error", [(429, OCRBusy), (500, OCRUnavailable), (401, OCRUnavailable), (422, InvalidImage)])
def test_remote_http_errors_are_safe_and_actionable(status, error):
    reader = engine(lambda request: httpx.Response(200, json=READY) if request.method == "GET"
                    else httpx.Response(status, json={"error": "private details"}))
    with pytest.raises(error) as caught, Image.new("RGB", (20, 20)) as image:
        reader.extract_image(image, "back")
    assert "private details" not in str(caught.value)
    reader.close()


def test_missing_url_and_timeout_do_not_activate_cpu_fallback():
    reader = NamLhnGPUEngine("")
    with pytest.raises(OCRUnavailable), Image.new("RGB", (20, 20)) as image:
        reader.extract_image(image, "front")
    reader.close()
    reader = engine(lambda request: (_ for _ in ()).throw(httpx.ConnectTimeout("private")))
    with pytest.raises(OCRUnavailable), Image.new("RGB", (20, 20)) as image:
        reader.extract_image(image, "front")
    assert not reader.describe()["available"]
    reader.close()


def test_multiple_documents_remain_a_distinct_error():
    reader = engine(lambda request: httpx.Response(200, json=READY) if request.method == "GET" else
                    httpx.Response(422, json={"error_code": "multiple_documents", "detected_regions": 2}))
    with pytest.raises(OCRMultipleDocuments) as caught, Image.new("RGB", (20, 20)) as image:
        reader.extract_image(image, "back")
    assert caught.value.detected_regions == 2
    reader.close()
