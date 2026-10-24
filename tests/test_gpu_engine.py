import httpx
import pytest

from identification_demo.engine import OCRBusy, OCRUnavailable, OCRMultipleDocuments
from identification_demo.gpu_engine import DocUGPUEngine, normalize_result
from identification_demo.processing import InvalidImage


READY = {"pipeline_ready": True, "device": "cuda:0", "cuda_available": True}
FRONT = {"success": True, "doc_type": "cccd_front", "extracted_fields": {
    "id_number": "001099000001", "full_name": "NGUYỄN VĂN THỬ", "dob": "1999-02-01",
    "gender": "Nam", "nationality": "Việt Nam", "origin_location": "Hà Nội",
    "residence_location": "Địa chỉ thử",
}, "text_lines": [{"text": "Họ và tên"}, {"text": "Ngày sinh"},
                   {"text": "Giới tính: Nam"}, {"text": "Quốc tịch: Việt Nam"}]}


def engine(handler):
    return DocUGPUEngine("http://gpu.test", transport=httpx.MockTransport(handler))


def test_maps_actual_docu_metadata_and_normalizes_dates():
    raw = normalize_result(FRONT)
    assert raw["structured_fields"]["id_number"] == "001099000001"
    assert raw["structured_fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert raw["structured_fields"]["date_of_birth"] == "01/02/1999"
    assert raw["structured_fields"]["place_of_origin"] == "Hà Nội"
    assert raw["document_side"] == "front"


def test_authority_comes_from_printed_text_instead_of_upstream_constant():
    raw = normalize_result({"success": True, "doc_type": "cccd_back", "extracted_fields": {
        "issue_date": "2021-03-15", "issue_authority": "CỤC TRƯỞNG CỤC CẢNH SÁT", "characteristics": "Nốt ruồi má trái",
    }, "text_lines": [{"text": "Cơ quan cấp: Công an tỉnh Hà Nam"}]})
    assert raw["structured_fields"]["issuing_authority"] == "Công an tỉnh Hà Nam"
    assert raw["structured_fields"]["issue_date"] == "15/03/2021"
    assert raw["structured_fields"]["identifying_features"] == "Nốt ruồi má trái"
    absent = normalize_result({"success": True, "doc_type": "cccd_back", "extracted_fields": {"issue_authority": "FIXED GUESS"}})
    assert absent["structured_fields"]["issuing_authority"] == ""


def test_invalid_fields_dates_and_raw_content_are_not_exposed():
    raw = normalize_result({"success": True, "doc_type": "cccd_front", "extracted_fields": {"dob": "1999-02-31", "full_name": "INVALID NAME"},
                            "field_details": {"full_name": {"is_valid": False}}})
    assert raw["structured_fields"]["date_of_birth"] == ""
    assert raw["structured_fields"]["name"] == ""


def test_upstream_defaults_need_recognized_evidence():
    payload = {**FRONT, "text_lines": [{"text": "Họ và tên"}, {"text": "GIỚI TÍNH"},
                                     {"text": "QUỐC TỊCH"}, {"text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"}]}
    fields = normalize_result(payload)["structured_fields"]
    assert fields["gender"] == fields["nationality"] == ""
    valid = normalize_result(FRONT)["structured_fields"]
    assert valid["gender"] == "Nam" and valid["nationality"] == "Việt Nam"


def test_identifying_features_use_actual_text_instead_of_the_caption():
    caption = "Đặc điểm nhân dạng / Personal identification"
    payload = {"success": True, "doc_type": "cccd_back", "extracted_fields": {"characteristics": caption},
               "text_lines": [{"text": caption}, {"text": "Nốt ruồi má trái"}, {"text": "Ngày cấp: 15/03/2021"}]}
    assert normalize_result(payload)["structured_fields"]["identifying_features"] == "Nốt ruồi má trái"
    payload["text_lines"] = [{"text": caption}, {"text": "Ngày cấp: 15/03/2021"}]
    assert normalize_result(payload)["structured_fields"]["identifying_features"] == ""


@pytest.mark.parametrize("payload", [None, [], {"success": False, "error_message": "private error"},
    {"success": True}, {"success": True, "extracted_fields": []},
    {"success": True, "extracted_fields": {"full_name": ["bad"]}},
    {"success": True, "doc_type": "gplx", "extracted_fields": {}},
])
def test_invalid_gpu_payloads_never_become_valid_ocr(payload):
    with pytest.raises(OCRUnavailable):
        normalize_result(payload)


def test_http_adapter_uses_multipart_and_correct_side_without_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_OCR_API_KEY", "test-token")
    requests = []
    def handler(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(200, json=READY if request.method == "GET" else FRONT)
    reader = engine(handler)
    path = tmp_path / "card.png"
    path.write_bytes(b"image bytes")
    result = reader.extract_document(path, "front")
    assert result["structured_fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert requests[0].url.path == "/api/v1/health/ready"
    assert requests[1].url.path == "/api/v1/extract"
    assert requests[1].url.params["doc_type_hint"] == "cccd_front"
    assert b'name="file"' in requests[1].content and b"image bytes" in requests[1].content
    assert reader.describe()["available"]


@pytest.mark.parametrize("ready", [
    {**READY, "pipeline_ready": False}, {**READY, "device": "cpu"},
    {**READY, "cuda_available": False}, {},
])
def test_high_mode_refuses_a_worker_falling_back_to_cpu(ready, tmp_path):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=ready)
    reader = engine(handler)
    with pytest.raises(OCRUnavailable):
        reader.extract_document(tmp_path / "not-opened.png", "front")
    assert all(request.method == "GET" for request in requests)
    assert reader.describe()["available"] is False


@pytest.mark.parametrize("status,error", [(429, OCRBusy), (503, OCRUnavailable), (500, OCRUnavailable), (401, OCRUnavailable), (422, InvalidImage)])
def test_remote_http_errors_are_safe_and_actionable(status, error, tmp_path):
    def handler(request):
        return httpx.Response(200, json=READY) if request.method == "GET" else httpx.Response(status, text="private backend error")
    path = tmp_path / "card.png"
    path.write_bytes(b"image")
    with pytest.raises(error) as caught:
        engine(handler).extract_document(path, "back")
    assert "private backend error" not in str(caught.value)


def test_missing_url_and_timeout_do_not_activate_cpu_fallback(tmp_path):
    reader = DocUGPUEngine("")
    assert reader.describe()["configured"] is False
    with pytest.raises(OCRUnavailable):
        reader.extract_document(tmp_path / "not-opened.png", "front")
    def timeout(request):
        raise httpx.ConnectTimeout("private address")
    assert engine(timeout).describe()["available"] is False
    with pytest.raises(OCRUnavailable):
        engine(timeout).extract_document(tmp_path / "not-opened.png", "front")


def test_ambiguous_regions_remain_distinct_from_invalid_image(tmp_path):
    def handler(request):
        return httpx.Response(200, json=READY) if request.method == "GET" else httpx.Response(422, json={
            "error_code": "multiple_documents", "detected_regions": 2, "error": "private backend details",
        })
    path = tmp_path / "card.png"
    path.write_bytes(b"image")
    with pytest.raises(OCRMultipleDocuments) as caught:
        engine(handler).extract_document(path, "back")
    assert caught.value.detected_regions == 2
    assert "private backend" not in str(caught.value)


def test_enhancement_metadata_excludes_raw_text_and_unknown_properties():
    result = normalize_result({**FRONT, "enhancement": {
        "target": "issuing_authority", "reread_lines": 2, "changed_lines": 1,
        "contrast_enhancement": True, "expanded_crop": True, "raw_text": "private debug text",
    }})
    assert result["enhancement"]["changed_lines"] == 1
    assert result["enhancement"]["reread_lines"] == 2
    assert "raw_text" not in result["enhancement"]
    assert "private debug text" not in str(result)
