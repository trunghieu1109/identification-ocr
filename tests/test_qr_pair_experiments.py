import threading

import pytest
from fastapi.testclient import TestClient

from identification_demo.pipeline import CARD_FIELDS
from identification_demo.processing import InvalidImage
from identification_demo.qr import QRBusy, QRUnavailable
from identification_demo.qr_experiments import METHODS, QRPairExperiments, process_qr_pair_experiment
from identification_demo.web import MAX_IMAGE_BYTES, create_app
from test_pipeline import Engine, Reader, image_data
from test_qr_experiments import FakeExperiments, NoOCR, PAYLOAD


ENDPOINT = "/api/qr/experiment/pair"


def pair_data(back=None):
    return {"front": image_data(), "back": image_data("back") if back is None else back}


def submit(client, method="wechat", execution="parallel", back=None):
    return client.post(ENDPOINT, data={"method": method, "execution": execution}, files={
        f"{side}_image": (f"{side}.png", data, "image/png") for side, data in pair_data(back).items()
    })


def make_lab(front=None, back=None):
    return QRPairExperiments({"front": front or FakeExperiments(), "back": back or FakeExperiments()})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("execution", ["parallel", "sequential"])
def test_both_sides_use_the_selected_method_and_keep_individual_results(method, execution):
    front, back = FakeExperiments(texts=[]), FakeExperiments()
    lab = make_lab(front, back)
    result = process_qr_pair_experiment(pair_data(), lab, method, execution)
    assert front.calls == [(method, (300, 190), "RGB")]
    assert back.calls == [(method, (320, 190), "RGB")]
    assert result["source"] == "qr_pair_experiment" and result["execution"] == execution
    assert result["status"] == "complete" and result["decoded_count"] == 1
    assert result["sides"]["front"]["status"] == "no_qr"
    assert result["sides"]["back"]["fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert result["sides"]["back"]["timings_ms"]["detect_decode"] == 3.4
    for side in result["sides"].values():
        assert 0 <= side["schedule_ms"]["started_at_ms"] <= side["schedule_ms"]["finished_at_ms"]
        assert side["processing_time_ms"] >= side["timings_ms"]["image_decode"]
    assert result["timings_ms"]["total"] == result["processing_time_ms"]
    assert result["timings_ms"]["execution"] <= result["processing_time_ms"]
    assert PAYLOAD not in str(result)


def test_parallel_readers_really_overlap_instead_of_waiting_for_a_shared_lock():
    barrier = threading.Barrier(2)

    class Concurrent(FakeExperiments):
        def read(self, image, method):
            barrier.wait(timeout=3)
            return super().read(image, method)

    result = process_qr_pair_experiment(pair_data(), make_lab(Concurrent(), Concurrent()), "wechat")
    assert all(side["success"] for side in result["sides"].values())
    front, back = (result["sides"][side]["schedule_ms"] for side in ("front", "back"))
    assert max(front["started_at_ms"], back["started_at_ms"]) <= min(front["finished_at_ms"], back["finished_at_ms"])


def test_sequential_mode_finishes_front_before_starting_back():
    result = process_qr_pair_experiment(pair_data(), make_lab(), "wechat", "sequential")
    assert result["sides"]["front"]["schedule_ms"]["finished_at_ms"] <= result["sides"]["back"]["schedule_ms"]["started_at_ms"]


def test_pair_models_are_independent_reused_and_not_the_main_reader(monkeypatch):
    lab = QRPairExperiments()
    initialized = []
    for side, reader in lab._readers.items():
        def initialize(method, side=side, reader=reader):
            if method not in reader._backends:
                initialized.append(side)
                reader._backends[method] = object()
            return reader._backends[method]
        monkeypatch.setattr(reader, "_initialize", initialize)
        monkeypatch.setattr(reader, "_read_wechat", lambda backend, image: ([PAYLOAD], []))
    first = process_qr_pair_experiment(pair_data(), lab, "wechat")
    second = process_qr_pair_experiment(pair_data(), lab, "wechat", "sequential")
    assert sorted(initialized) == ["back", "front"]
    assert lab._readers["front"]._backends["wechat"] is not lab._readers["back"]._backends["wechat"]
    assert all(side["cold_start"] for side in first["sides"].values())
    assert not any(side["cold_start"] for side in second["sides"].values())


@pytest.mark.parametrize("second,status", [(PAYLOAD, "complete"), (PAYLOAD.replace("001099000001", "001099000002"), "multiple_qr")])
def test_duplicate_payloads_are_deduplicated_and_conflicts_remain_visible(second, status):
    result = process_qr_pair_experiment(pair_data(), make_lab(back=FakeExperiments(texts=[second])), "wechat")
    assert result["status"] == status
    if status == "multiple_qr":
        assert not any(result["fields"].values())
        assert result["sides"]["front"]["fields"]["id_number"] != result["sides"]["back"]["fields"]["id_number"]
    else:
        assert result["decoded_count"] == 1


def test_invalid_back_is_rejected_before_either_reader_runs():
    front, back = FakeExperiments(), FakeExperiments()
    with pytest.raises(InvalidImage, match="Mặt sau"):
        process_qr_pair_experiment(pair_data(b"bad image"), make_lab(front, back), "wechat")
    assert not front.calls and not back.calls


def test_one_reader_failure_preserves_the_other_side_and_hides_internal_errors():
    lab = make_lab(front=FakeExperiments(error=RuntimeError("private error")))
    result = process_qr_pair_experiment(pair_data(), lab, "wechat")
    assert result["status"] == "partial_error"
    assert not result["sides"]["front"]["success"]
    assert result["sides"]["back"]["fields"]["id_number"] == "001099000001"
    assert "private error" not in str(result)
    assert not lab._lock.locked()


@pytest.mark.parametrize("error,status", [(QRBusy("busy"), 429), (QRUnavailable("unavailable"), 503), (RuntimeError("private"), 503)])
def test_both_readers_failing_return_an_actionable_http_error(error, status):
    lab = make_lab(FakeExperiments(error=error), FakeExperiments(error=error))
    client = TestClient(create_app(NoOCR(), qr_pair_experiments=lab))
    response = submit(client)
    assert response.status_code == status and not response.json()["success"]
    assert "private" not in response.text
    assert not lab._lock.locked()
    if status == 429:
        assert response.headers["retry-after"] == "5"


def test_api_is_experimental_and_never_invokes_main_qr_or_ocr():
    reader = Reader()
    client = TestClient(create_app(NoOCR(), reader, gpu_engine=NoOCR(), qr_pair_experiments=make_lab()))
    response = submit(client)
    assert response.status_code == 200 and response.json()["request_id"]
    assert response.headers["cache-control"] == "no-store"
    assert not reader.calls
    assert PAYLOAD not in response.text


@pytest.mark.parametrize("data,status", [(b"", 422), (b"bad pixels", 422), (b"x" * (MAX_IMAGE_BYTES + 1), 413)])
def test_invalid_second_upload_never_invokes_either_method(data, status):
    front, back = FakeExperiments(), FakeExperiments()
    client = TestClient(create_app(NoOCR(), qr_pair_experiments=make_lab(front, back)))
    response = submit(client, back=data)
    assert response.status_code == status
    assert not front.calls and not back.calls


def test_invalid_method_mode_and_missing_files_are_rejected():
    front, back = FakeExperiments(), FakeExperiments()
    client = TestClient(create_app(NoOCR(), qr_pair_experiments=make_lab(front, back)))
    assert submit(client, method="unknown").status_code == 422
    assert submit(client, execution="unknown").status_code == 422
    assert client.post(ENDPOINT, data={"method": "wechat"}).status_code == 422
    assert not front.calls and not back.calls


def test_main_pipeline_keeps_its_existing_schema_and_does_not_use_pair_lab(monkeypatch):
    lab = make_lab()
    monkeypatch.setattr(lab, "read", lambda *args: pytest.fail("Main flow invoked pair experiment"))
    engine, reader = Engine(), Reader()
    client = TestClient(create_app(engine, reader, qr_pair_experiments=lab))
    response = client.post("/api/cccd/extract", files={
        f"{side}_image": (f"{side}.png", data, "image/png") for side, data in pair_data().items()
    })
    assert response.status_code == 200
    assert set(response.json()["fields"]) == set(CARD_FIELDS)
    assert reader.calls == ["front", "back"]
