"""Promotion must preserve an immediate classic switch and the public contract."""
import copy

import httpx
import pytest
from fastapi.testclient import TestClient

from identification_demo.engine import OCRUnavailable
from identification_demo.experiment_pipeline import process_experiment
from identification_demo.gpu_parallel import main_result
from identification_demo.pipeline import CARD_FIELDS
from identification_demo.processing import InvalidImage
from identification_demo.web import create_app
from test_cccd_experiment import Aligner, MRZ
from test_pipeline import Engine, Reader, image_data


def raw_result():
    return process_experiment({"front": image_data(), "back": image_data("back")},
                              Engine(), Reader(), Aligner(), MRZ(), align=False)


class Backend:
    def __init__(self, payload=None, status=200):
        self.payload = main_result(raw_result()) if payload is None else payload
        self.status, self.calls, self.health_calls = status, [], 0

    def describe(self):
        self.health_calls += 1
        return {"available": True, "experimental": True}

    def extract(self, images, **options):
        self.calls.append(options)
        return copy.deepcopy(self.payload), self.status


def submit(client, **options):
    return client.post("/api/cccd/extract", data={"quality": "high", **options}, files={
        "front_image": ("front.png", image_data(), "image/png"),
        "back_image": ("back.png", image_data("back"), "image/png"),
    })


def test_server_switch_off_never_calls_parallel_backend(monkeypatch):
    monkeypatch.setenv("GPU_MRZ_ENABLED", "0")
    backend, gpu = Backend(), Engine()
    with TestClient(create_app(Engine(), Reader(), gpu, experiment_backend=backend)) as client:
        health = client.get("/api/health").json()
        result = submit(client).json()
    assert health["gpu_mrz"] == {"default_enabled": False}
    assert backend.health_calls == 0 and backend.calls == []
    assert gpu.calls == ["front", "back"]
    assert result["gpu_pipeline"] == {"requested": "classic", "used": "classic", "fallback": False}
    assert result["field_sources"]["name"] == "qr_front"


def test_server_switch_on_defaults_to_parallel_and_keeps_twelve_field_contract(monkeypatch):
    monkeypatch.setenv("GPU_MRZ_ENABLED", "1")
    cpu, gpu, reader, backend = Engine(), Engine(), Reader(), Backend()
    with TestClient(create_app(cpu, reader, gpu, experiment_backend=backend)) as client:
        health = client.get("/api/health").json()
        response = submit(client)
    result = response.json()
    assert health["gpu_mrz"]["default_enabled"] and health["gpu_mrz"]["worker"]["available"]
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert result["source"] == "qr_ocr" and list(result["fields"]) == list(CARD_FIELDS)
    assert result["fields"]["identifying_features"] == "" and result["total_fields"] == 12
    assert result["gpu_pipeline"]["used"] == "parallel" and result["request_timings_ms"]
    assert backend.calls == [{"diagnostics": False, "include_back_ocr": False}]
    assert not cpu.calls and not gpu.calls and not reader.calls


@pytest.mark.parametrize("default,override,used", [("1", "false", "classic"), ("0", "true", "parallel")])
def test_per_request_switch_overrides_server_default(monkeypatch, default, override, used):
    monkeypatch.setenv("GPU_MRZ_ENABLED", default)
    backend, gpu = Backend(), Engine()
    with TestClient(create_app(Engine(), Reader(), gpu, experiment_backend=backend)) as client:
        result = submit(client, use_mrz=override).json()
    assert result["gpu_pipeline"]["used"] == used
    assert bool(backend.calls) == (used == "parallel")
    assert bool(gpu.calls) == (used == "classic")


def test_cpu_ignores_mrz_switch_and_does_not_contact_parallel_worker(monkeypatch):
    monkeypatch.setenv("GPU_MRZ_ENABLED", "1")
    cpu, gpu, backend = Engine(), Engine(), Backend()
    with TestClient(create_app(cpu, Reader(), gpu, experiment_backend=backend)) as client:
        result = submit(client, quality="medium", use_mrz="true").json()
    assert result["quality"] == "medium" and "gpu_pipeline" not in result
    assert cpu.calls == ["front", "back"] and not gpu.calls and not backend.calls


@pytest.mark.parametrize("status", [401, 502, 503, 504])
def test_unavailable_parallel_worker_uses_classic_gpu_with_explicit_warning(monkeypatch, status):
    monkeypatch.setenv("GPU_MRZ_ENABLED", "1")
    gpu, backend = Engine(), Backend({"success": False, "error": "unavailable"}, status)
    with TestClient(create_app(Engine(), Reader(), gpu, experiment_backend=backend)) as client:
        response = submit(client)
    result = response.json()
    assert response.status_code == 200 and gpu.calls == ["front", "back"]
    assert result["gpu_pipeline"] == {"requested": "parallel", "used": "classic", "fallback": True}
    assert any("MRZ chưa sẵn sàng" in message for message in result["warnings"])


@pytest.mark.parametrize("status", [422, 429])
def test_identity_or_busy_error_does_not_start_a_second_pipeline(monkeypatch, status):
    monkeypatch.setenv("GPU_MRZ_ENABLED", "1")
    gpu, backend = Engine(), Backend({"success": False, "error": "different cards"}, status)
    with TestClient(create_app(Engine(), Reader(), gpu, experiment_backend=backend)) as client:
        response = submit(client)
    assert response.status_code == status and not gpu.calls
    if status == 429:
        assert response.headers["retry-after"] == "5"


def test_invalid_parallel_payload_falls_back_without_leaking_debug(monkeypatch):
    monkeypatch.setenv("GPU_MRZ_ENABLED", "1")
    backend = Backend({"success": True, "fields": {}, "raw_text": "PRIVATE DEBUG"})
    with TestClient(create_app(Engine(), Reader(), Engine(), experiment_backend=backend)) as client:
        response = submit(client)
    assert response.json()["gpu_pipeline"]["fallback"] and "PRIVATE DEBUG" not in response.text


def test_main_adapter_strips_previews_raw_mrz_and_front_ocr_but_allows_opt_in_back_text():
    raw = raw_result()
    raw["sides"]["front"]["preview"] = "PRIVATE IMAGE"
    raw["sides"]["front"]["ocr"][0]["raw_text"] = "PRIVATE FRONT OCR"
    raw["sides"]["back"]["mrz"]["raw_text"] = "PRIVATE MRZ"
    plain = main_result(raw)
    assert all(text not in str(plain) for text in ("PRIVATE IMAGE", "PRIVATE FRONT OCR", "PRIVATE MRZ"))
    assert "debug" not in plain["ocr"]["back"]
    assert main_result(raw, True)["ocr"]["back"]["debug"]["final_lines"]


def test_main_adapter_preserves_unlimited_expiry_without_marking_it_missing():
    raw = raw_result()
    raw["fields"]["expiry_date"] = ""
    raw["field_sources"]["expiry_date"] = None
    raw["sides"]["front"]["ocr"][0]["expiry"] = {"expiry_type": "unlimited", "label_found": True}
    raw["sides"]["back"]["ocr"][0].pop("expiry", None)
    result = main_result(raw)
    assert result["expiry"]["expiry_type"] == "unlimited" and "expiry_date" not in result["missing_fields"]


def test_main_adapter_keeps_conflicting_ocr_expiry_empty_and_uses_verified_mrz_when_present():
    raw = raw_result()
    raw["field_sources"]["expiry_date"] = "ocr_front_original"
    for side, value in (("front", "25/10/2036"), ("back", "25/10/2040")):
        raw["sides"][side]["ocr"][0]["expiry"] = {"label_found": True,
            "candidates": [{"expiry_type": "date", "expiry_date": value}]}
    result = main_result(raw)
    assert result["expiry"]["status"] == "conflict" and result["fields"]["expiry_date"] == ""
    assert result["field_sources"]["expiry_date"] is None and "expiry_date" in result["missing_fields"]
    raw["field_sources"]["expiry_date"] = "mrz_back"
    result = main_result(raw)
    assert result["expiry"]["status"] == "found" and result["fields"]["expiry_date"] == "25/10/2036"


def test_complete_qr_on_front_is_reported_complete_for_that_side():
    result = main_result(raw_result())
    assert result["qr"]["status"] == "complete" and result["qr"]["sides"]["front"] == "complete"


def test_main_adapter_rejects_conflicting_cards_and_all_models_failing():
    raw = raw_result()
    raw["status"] = "identity_conflict"
    with pytest.raises(InvalidImage):
        main_result(raw)
    raw = raw_result()
    raw["fields"] = dict.fromkeys(CARD_FIELDS, "")
    for side in raw["sides"].values():
        side["ocr"] = [{"status": "unavailable", "fields": {}}]
    with pytest.raises(OCRUnavailable):
        main_result(raw)


def test_worker_returns_compact_main_response_when_diagnostics_disabled():
    from identification_demo.experiment_worker import create_app as worker_app
    class Worker:
        def initialize(self): pass
        def close(self): pass
        def describe(self): return {"experimental": True, "available": True}
        def extract(self, images, **options): return raw_result()
    with TestClient(worker_app(Worker())) as client:
        response = client.post("/extract", data={"diagnostics": "false"}, files={
            "front_image": ("front.png", image_data()), "back_image": ("back.png", image_data("back"))})
    result = response.json()
    assert response.status_code == 200 and result["experimental"]
    assert result["gpu_pipeline"]["used"] == "parallel"
    assert all(key not in result for key in ("sides", "candidates", "raw_text"))
    assert "data:image" not in response.text and "raw_text" not in response.text


def test_invalid_server_toggle_is_rejected(monkeypatch):
    monkeypatch.setenv("GPU_MRZ_ENABLED", "maybe")
    with pytest.raises(ValueError, match="GPU_MRZ_ENABLED"):
        create_app(Engine(), Reader(), Engine())
