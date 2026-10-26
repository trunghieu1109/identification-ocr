import httpx
import pytest
from fastapi.testclient import TestClient

from identification_demo.engine import OCRBusy, OCRUnavailable
from identification_demo.gpu_engine import DocUGPUEngine
from identification_demo.namlhn_gpu_engine import NamLhnGPUEngine
from identification_demo.ocr_experiments import EXPERIMENT_FIELDS
from identification_demo.web import MAX_IMAGE_BYTES, create_app
from test_pipeline import Engine, Reader, FRONT, BACK, PAYLOAD, image_data
from test_gpu_engine import READY

ENDPOINT = "/api/cccd/experiment/extract"


def full_engine():
    return Engine({**FRONT, "text_lines": [*FRONT["text_lines"], "Có giá trị đến: 25/10/2036"]}, BACK)


@pytest.mark.parametrize("text", ["Số 1001205025331", "Số: O01205025331"])
def test_empty_provider_id_is_recovered_from_observed_text_without_requiring_valid_format(text):
    from identification_demo.processing import process_image

    backend = Engine({"structured_fields": {"id_number": ""}, "text_lines": [text], "document_side": "front"}, BACK)
    result = process_image(image_data(), backend, "front")
    assert result["fields"]["id_number"] == text.split()[-1]
    assert result["id_number_validation"]["status"] == "unverified"
    assert not result["id_number_validation"]["format_valid"]


@pytest.mark.parametrize("use_qr", [False, True])
def test_imperfect_ocr_id_is_retained_with_validation_but_cannot_replace_qr(use_qr):
    from test_namlhn_identity import line
    from identification_demo.cpu_pipeline import _parse

    raw, fields = _parse([line("Số: 1001205025331")], "front")
    front = {"structured_fields": fields, "document_side": "front", "text_lines": [line("Số: 1001205025331")],
             "id_number_validation": raw["_id_number_validation"],
             "image_preprocessing": {"crop": {"status": "cropped"}, "enhancement": {"brightness": True}}}
    backend = Engine(front, BACK)
    reader = Reader() if use_qr else Reader([])
    with TestClient(create_app(full_engine(), reader, namlhn_gpu_engine=backend)) as client:
        result = submit(client, "namlhn_gpu", use_qr=str(use_qr).lower()).json()
        assert result["success"]
        assert result["fields"]["id_number"] == ("001099000001" if use_qr else "1001205025331")
        assert result["field_sources"]["id_number"] == ("qr_front" if use_qr else "ocr_front")
        assert not result["ocr"]["front"]["id_number_validation"]["format_valid"]
        assert result["ocr"]["front"]["image_preprocessing"]["crop"]["status"] == "cropped"
        assert any("Số căn cước OCR" in warning for warning in result["warnings"]) is not use_qr
        # Main API uses the same metadata and retains imperfect OCR values as well.
        response = client.post("/api/cccd/extract", data={"quality": "high"}, files={
            "front_image": ("front.png", image_data(), "image/png"),
            "back_image": ("back.png", image_data("back"), "image/png")})
        main = response.json()
        assert main["success"] and main["fields"]["id_number"] == result["fields"]["id_number"]
        assert main["ocr"]["front"]["id_number_validation"]["status"] == "unverified"


def submit(client, engine="docu_gpu", front=None, back=None, **data):
    return client.post(ENDPOINT, data={"engine": engine, **data}, files={
        "front_image": ("../../front.png", image_data() if front is None else front, "image/png"),
        "back_image": ("../../back.png", image_data("back") if back is None else back, "image/png"),
    })


@pytest.mark.parametrize("name", ["namlhn_cpu", "namlhn_gpu", "docu_gpu", "docu_gpu_regions", "docu_gpu_v3"])
def test_each_variant_selects_only_its_engine_and_returns_all_fields(name, tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_UPLOAD_TMP_DIR", str(tmp_path))
    cpu, gpu, docu, regions, multi = full_engine(), full_engine(), full_engine(), full_engine(), full_engine()
    engines = {"namlhn_cpu": cpu, "namlhn_gpu": gpu, "docu_gpu": docu, "docu_gpu_regions": regions, "docu_gpu_v3": multi}
    qr = Reader()
    client = TestClient(create_app(cpu, qr, namlhn_gpu_engine=gpu, docu_gpu_engine=docu,
                                   docu_region_engine=regions, docu_expanded_engine=multi))
    response = submit(client, name)
    assert response.status_code == 200
    result = response.json()
    assert result["engine"] == name and result["experiment"] == "full_ocr" and result["status"] == "complete"
    assert tuple(result["fields"]) == EXPERIMENT_FIELDS and result["total_fields"] == 12
    assert result["fields"]["name"] == "NGUYỄN VĂN THỬ" and result["field_sources"]["name"] == "qr_front"
    assert result["fields"]["expiry_date"] == "25/10/2036" and result["field_sources"]["expiry_date"] == "ocr_front"
    assert result["fields"]["identifying_features"] == "Nốt ruồi má trái"
    assert result["field_sources"]["identifying_features"] == "ocr_back"
    assert result["recognized_fields"] == 12 and result["missing_fields"] == []
    assert not result["expiry"].get("candidates") and "debug" not in result["ocr"]["front"]
    assert "text_lines" not in response.text and PAYLOAD not in response.text
    assert result["request_timings_ms"]["pipeline"] >= 0 and "server-timing" in response.headers
    assert response.headers["cache-control"] == "no-store"
    for key, engine in engines.items():
        assert engine.calls == (["front", "back"] if key == name else [])
    assert not list(tmp_path.iterdir())


def test_expiry_is_only_taken_from_front_even_when_back_has_a_different_date():
    engine = Engine({**FRONT, "text_lines": [*FRONT["text_lines"], "Có giá trị đến: 25/10/2036"]},
                    {"text_lines": [*BACK["text_lines"], "Có giá trị đến: 25/10/2040"]})
    client = TestClient(create_app(engine, Reader()))
    result = submit(client, "namlhn_cpu", include_ocr_text="true").json()
    assert result["fields"]["expiry_date"] == "25/10/2036"
    assert result["expiry"]["side"] == "front" and result["expiry"]["status"] == "found"
    assert len(result["expiry"]["candidates"]) == 1
    assert result["ocr"]["front"]["debug"]["text_lines"][-1]["is_expiry"]


def test_spatial_evidence_uses_visible_row_numbers_and_requires_debug_opt_in():
    front = {**FRONT, "text_lines": ["", "Họ và tên:", "HOÀNG THÀNH VINH"],
             "field_mapping": {"name": {"value": "HOÀNG THÀNH VINH", "method": "spatial", "status": "found",
                                         "box": [], "label_indices": [1], "value_indices": [2]}}}
    client = TestClient(create_app(Engine(front, BACK), Reader([])))
    response = submit(client, "namlhn_cpu", include_ocr_text="true", use_qr="false")
    assert response.status_code == 200
    debug = response.json()["ocr"]["front"]["debug"]
    assert debug["field_mapping"]["name"]["label_rows"] == [1]
    assert debug["field_mapping"]["name"]["value_rows"] == [2]
    response = submit(client, "namlhn_cpu", use_qr="false")
    assert "field_mapping" not in response.text and "debug" not in response.json()["ocr"]["front"]


def test_a_date_only_on_the_back_does_not_fill_front_expiry():
    engine = Engine(FRONT, {"text_lines": [*BACK["text_lines"], "Có giá trị đến: 25/10/2040"]})
    result = submit(TestClient(create_app(engine, Reader())), "namlhn_cpu").json()
    assert result["fields"]["expiry_date"] == "" and result["field_sources"]["expiry_date"] is None
    assert result["missing_fields"] == ["expiry_date"] and result["status"] == "partial"


def test_ocr_only_mode_skips_qr_and_reads_every_field_including_expiry_and_features():
    qr, engine = Reader(), full_engine()
    result = submit(TestClient(create_app(engine, qr)), "namlhn_cpu", use_qr="false").json()
    assert not qr.calls and result["source"] == "ocr" and result["qr"]["status"] == "skipped"
    assert result["fields"]["name"] == FRONT["Name"] and result["field_sources"]["name"] == "ocr_front"
    assert result["fields"]["old_id_number"] == "" and result["fields"]["identifying_features"]


def test_unlimited_front_has_explicit_metadata_and_is_not_a_missing_expiry():
    engine = Engine({**FRONT, "text_lines": [*FRONT["text_lines"], "Có giá trị đến: Không thời hạn"]}, BACK)
    result = submit(TestClient(create_app(engine, Reader())), "namlhn_cpu").json()
    assert result["fields"]["expiry_date"] == "" and result["expiry"]["expiry_type"] == "unlimited"
    assert "expiry_date" not in result["missing_fields"] and result["status"] == "complete"


def test_front_conflicting_expiry_values_are_not_arbitrarily_chosen():
    engine = Engine({**FRONT, "text_lines": [*FRONT["text_lines"], "Có giá trị đến: 25/10/2036", "Date of expiry: 25/10/2037"]}, BACK)
    result = submit(TestClient(create_app(engine, Reader())), "namlhn_cpu").json()
    assert result["fields"]["expiry_date"] == "" and result["expiry"]["status"] == "conflict"
    assert result["status"] == "partial" and result["warnings"]


@pytest.mark.parametrize("data,status", [(b"", 422), (b"invalid", 422), (b"x" * (MAX_IMAGE_BYTES + 1), 413)])
def test_invalid_second_image_never_reaches_any_reader(data, status):
    engine, qr = full_engine(), Reader()
    response = submit(TestClient(create_app(engine, qr)), "namlhn_cpu", back=data)
    assert response.status_code == status and not engine.calls and not qr.calls


def test_missing_files_unknown_engine_and_conflicting_qr_are_rejected():
    engine, qr = full_engine(), Reader()
    client = TestClient(create_app(engine, qr))
    assert client.post(ENDPOINT).status_code == 422
    assert submit(client, "unknown").status_code == 422 and not engine.calls and not qr.calls
    qr.results["back"] = [PAYLOAD.replace("001099000001", "001099000002")]
    assert submit(client, "namlhn_cpu").status_code == 422 and not engine.calls


@pytest.mark.parametrize("exception,status", [(OCRBusy("busy"), 429), (OCRUnavailable("missing"), 503)])
def test_no_fallback_when_gpu_fails_without_any_data(exception, status):
    cpu, gpu = full_engine(), Engine(exception, exception)
    response = submit(TestClient(create_app(cpu, Reader([]), namlhn_gpu_engine=gpu)), "namlhn_gpu")
    assert response.status_code == status and not cpu.calls
    if status == 429:
        assert response.headers["retry-after"] == "5"


def test_gpu_failure_keeps_qr_and_never_uses_the_cpu_engine():
    cpu, gpu = full_engine(), Engine(OCRUnavailable("unavailable"), OCRUnavailable("unavailable"))
    result = submit(TestClient(create_app(cpu, Reader(), namlhn_gpu_engine=gpu)), "namlhn_gpu").json()
    assert result["status"] == "partial" and result["field_sources"]["name"] == "qr_front"
    assert not cpu.calls and result["ocr"]["front"]["status"] == "error"


def test_main_cpu_api_returns_expiry_without_selecting_either_gpu():
    cpu, gpu, docu = full_engine(), full_engine(), full_engine()
    client = TestClient(create_app(cpu, Reader(), docu, namlhn_gpu_engine=gpu))
    result = client.post("/api/cccd/extract", data={"quality": "medium"}, files={
        "front_image": ("front.png", image_data(), "image/png"),
        "back_image": ("back.png", image_data("back"), "image/png"),
    }).json()
    assert result["total_fields"] == 12 and result["fields"]["expiry_date"] == "25/10/2036"
    assert result["fields"]["identifying_features"] == "" and not gpu.calls and not docu.calls


def test_namlhn_gpu_http_protocol_and_cpu_shaped_fields_are_supported():
    seen = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={**READY, "recognition_model": "vgg_seq2seq", "strategy": "namlhn_docu_v1", "text_enhancement": "adaptive"})
        side = request.url.params["doc_type_hint"].removeprefix("cccd_")
        seen.append((side, request.url.params["fields"]))
        structured = ({"id_number": "001099000001", "name": "TEST", "nationality": "Việt Nam", "place_of_origin": "Hà Nội"}
                      if side == "front" else {"issue_date": "15/03/2021", "issuing_authority": "CÔNG AN TỈNH HÀ NAM", "identifying_features": "Nốt ruồi"})
        return httpx.Response(200, json={"success": True, "document_side": side, "structured_fields": structured,
                                        "text_lines": ["Có giá trị đến: 25/10/2036"] if side == "front" else ["Ngày cấp: 15/03/2021"]})

    gpu = NamLhnGPUEngine("http://gpu.test", httpx.MockTransport(handler))
    result = submit(TestClient(create_app(full_engine(), Reader(), namlhn_gpu_engine=gpu)), "namlhn_gpu").json()
    assert result["status"] == "complete" and result["fields"]["expiry_date"] == "25/10/2036"
    assert "expiry_date" in seen[0][1] and "identifying_features" in seen[1][1]
    assert result["ocr_engine"]["provider"] == "namlhn/vietnam-cccd-ocr" and result["ocr_engine"]["model"] == "vgg_seq2seq"
    assert result["ocr_engine"]["strategy"] == "namlhn_docu_v1"
    assert result["ocr_engine"]["text_enhancement"] == "adaptive"


def test_docu_experiment_does_not_send_expiry_to_its_existing_worker_whitelist():
    seen = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=READY)
        seen.append(request.url.params["fields"])
        side = request.url.params["doc_type_hint"]
        return httpx.Response(200, json={"success": True, "doc_type": side, "extracted_fields": {},
                                        "text_lines": ["Có giá trị đến: 25/10/2036"] if side == "cccd_front" else []})

    gpu = DocUGPUEngine("http://gpu.test", httpx.MockTransport(handler))
    result = submit(TestClient(create_app(full_engine(), Reader(), docu_gpu_engine=gpu))).json()
    assert result["fields"]["expiry_date"] == "25/10/2036" and all("expiry_date" not in fields for fields in seen)


def test_namlhn_health_reports_the_workers_actual_model_not_the_client_default(monkeypatch):
    monkeypatch.setenv("NAMLHN_GPU_PIPELINE", "legacy")
    monkeypatch.delenv("NAMLHN_GPU_MODEL", raising=False)
    monkeypatch.delenv("OCR_MODEL", raising=False)
    gpu = NamLhnGPUEngine("http://gpu.test", httpx.MockTransport(
        lambda _request: httpx.Response(200, json={**READY, "recognition_model": "vgg_seq2seq"})))
    assert gpu.model == "vgg_transformer"
    assert gpu.describe()["recognition_model"] == "vgg_seq2seq"
