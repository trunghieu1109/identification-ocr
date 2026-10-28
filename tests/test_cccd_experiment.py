"""Checks for MRZ trust boundaries, crop safety and production isolation."""
from datetime import date
import io

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from identification_demo.experiment_alignment import rectify
from identification_demo.experiment_api import ExperimentClient
from identification_demo.experiment_mrz import check_digit, parse_vietnam_td1, usable_mrz_fields, mrz_rank
from identification_demo.experiment_pipeline import process_experiment, reconcile
from identification_demo.processing import InvalidImage
from identification_demo.web import create_app
from test_pipeline import Engine, Reader, image_data, FRONT


def mrz_text(number="001099000001", birth="990201", expiry="361025", name="NGUYEN<<VAN<THU"):
    doc = number[-9:]
    first = "IDVNM" + doc + check_digit(doc) + number + "<<" + check_digit(number)
    second = birth + check_digit(birth) + "M" + expiry + check_digit(expiry) + "VNM" + "<" * 11
    second += check_digit(first[5:] + second[:7] + second[8:15] + second[18:29])
    return "\n".join((first, second, name.ljust(30, "<")))


def test_verified_vietnam_mrz_maps_full_id_and_never_invents_old_cmnd():
    result = parse_vietnam_td1(mrz_text(), today=date(2026, 10, 28))
    assert result["status"] == "verified"
    assert result["fields"]["id_number"] == "001099000001"
    assert "old_id_number" not in result["fields"]
    assert result["fields"]["name"] == "NGUYEN VAN THU"
    assert result["field_checks"]["name"] == "not_covered_by_checksum"
    assert result["fields"]["expiry_date"] == "25/10/2036"
    assert all(result["checks"].values())


@pytest.mark.parametrize("position", [6, 15, 28, 31, 43, 59])
def test_corrupt_protected_digits_are_not_promoted(position):
    raw = mrz_text()
    replacement = "1" if raw[position] != "1" else "2"
    result = parse_vietnam_td1(raw[:position] + replacement + raw[position + 1:])
    assert result["status"] != "verified"


def test_bad_lengths_and_unrecognized_national_layout_are_rejected():
    assert parse_vietnam_td1(mrz_text() + "<")["status"] == "invalid"
    assert parse_vietnam_td1(mrz_text().replace("IDVNM", "IDUTO"))["status"] == "unsupported"
    assert parse_vietnam_td1(mrz_text().replace("001099000001", "OO1099000001"))["status"] != "verified"


def test_ambiguous_birth_century_is_retained_as_evidence_not_guessed():
    result = parse_vietnam_td1(mrz_text(birth="200201"), today=date(2026, 10, 28))
    assert result["status"] == "verified"
    assert "date_of_birth" not in result["fields"]
    assert result["birth_date_yymmdd"] == "200201"


def test_mrz_does_not_fabricate_a_calendar_date_or_clean_name_letters():
    result = parse_vietnam_td1(mrz_text(birth="991332", name="NGUYEN<<ANNA"))
    assert "date_of_birth" not in result["fields"]
    assert result["status"] != "verified" and not result["checks"]["date_of_birth"]
    assert result["fields"]["name"] == "NGUYEN ANNA"


def test_partial_mrz_can_supply_verified_id_without_accepting_bad_expiry_or_unchecked_name():
    # G vs 6 can collide under the modulo-10 checksum; numeric/calendar checks
    # still reject it. Keep the independently verified document/ID evidence.
    raw = mrz_text().replace("M361025", "M3G1025")
    result = parse_vietnam_td1(raw)
    assert result["status"] == "partial"
    assert usable_mrz_fields(result) == {"id_number": "001099000001"}
    assert mrz_rank(result) > mrz_rank(parse_vietnam_td1("bad"))
    result["checks"]["composite"] = False
    assert usable_mrz_fields(result) == {}


def candidate(key, value, source, priority, **extra):
    return {"field": key, "value": value, "source": source, "priority": priority, **extra}


def test_qr_priority_accented_name_preservation_and_conflict_visibility():
    result = reconcile([
        candidate("name", "NGUYEN VAN THU", "mrz_back", 1, check="not_covered_by_checksum"),
        candidate("name", "NGUYỄN VĂN THỬ", "ocr_front_aligned", 2),
        candidate("date_of_birth", "01/02/1999", "qr_front_original", 0),
        candidate("date_of_birth", "02/02/1999", "ocr_front_aligned", 2),
    ])
    fields, sources, conflicts, review, identity_conflict = result
    assert fields["name"] == "NGUYỄN VĂN THỬ" and sources["name"] == "ocr_front_aligned"
    assert fields["date_of_birth"] == "01/02/1999"
    assert [item["field"] for item in conflicts] == ["date_of_birth"]
    assert review == ["date_of_birth"] and not identity_conflict


def test_different_qr_and_mrz_identities_block_automatic_merge():
    fields, sources, conflicts, review, blocked = reconcile([
        candidate("id_number", "001099000001", "qr_front_original", 0),
        candidate("id_number", "001099000002", "mrz_back", 1),
        candidate("name", "NGUYỄN VĂN THỬ", "qr_front_original", 0),
    ])
    assert blocked and not any(fields.values()) and not any(sources.values())
    assert conflicts[0]["field"] == "id_number"


@pytest.mark.parametrize("points,confidence", [
    ([[0, 0], [0, 0], [100, 100], [200, 100]], .99),
    ([[1, 1], [299, 1], [299, 189], [1, 189]], .3),
    ([[-30, 1], [299, 1], [299, 189], [1, 189]], .99),
    ([[float("nan"), 1], [299, 1], [299, 189], [1, 189]], .99),
    ([[2, 2], [15, 2], [15, 9], [2, 9]], .99),
])
def test_unsafe_crop_keeps_original_pixels(points, confidence):
    original = Image.new("RGB", (300, 190), "red")
    selected, info = rectify(original, points, confidence)
    assert info["status"] == "original" and selected.tobytes() == original.tobytes()
    selected.close()


def test_perspective_rectification_preserves_corner_colors():
    pixels = np.zeros((500, 800, 3), np.uint8)
    points = np.float32([[100, 100], [690, 60], [740, 400], [70, 440]])
    import cv2
    cv2.fillConvexPoly(pixels, points.astype("int32"), (255, 255, 255))
    for point, color in zip(points, [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        cv2.circle(pixels, tuple(point.astype(int)), 25, color, -1)
    selected, info = rectify(Image.fromarray(pixels), points[[2, 0, 3, 1]], .99)
    assert info["status"] == "aligned" and selected.width > selected.height
    actual = np.asarray(selected)
    for pixel, color in zip([actual[2, 2], actual[2, -3], actual[-3, -3], actual[-3, 2]],
                            [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        assert np.max(np.abs(pixel.astype(int) - color)) < 10


class Aligner:
    def align(self, image):
        return image.copy(), {"status": "aligned", "corners": [[1, 1], [299, 1], [299, 189], [1, 189]]}


class MRZ:
    def __init__(self, raw=None):
        self.raw = raw or mrz_text()

    def read(self, _image, **_options):
        return parse_vietnam_td1(self.raw)


def test_pipeline_reads_mrz_with_unreadable_qr_and_keeps_evidence():
    engine = Engine()
    result = process_experiment({"front": image_data(), "back": image_data("back")}, engine,
                                Reader([], []), Aligner(), MRZ(), align=True)
    assert result["fields"]["id_number"] == "001099000001"
    assert result["field_sources"]["id_number"] == "mrz_back"
    assert "id_number" in result["review_fields"]  # OCR disagrees; MRZ remains visible.
    assert result["sides"]["back"]["mrz"]["status"] == "verified"
    assert result["sides"]["front"]["preview"].startswith("data:image/jpeg;base64,")
    assert all(not path.parent.exists() for path in engine.paths)


def test_corrupt_mrz_never_becomes_an_automatic_field_source():
    bad = mrz_text().replace("IDVNM099000001", "IDVNM099000002")
    result = process_experiment({"front": image_data(), "back": image_data("back")}, Engine(),
                                Reader([], []), Aligner(), MRZ(bad), align=False)
    assert not any(source and source.startswith("mrz") for source in result["field_sources"].values())


def test_partial_mrz_still_flags_disagreement_with_checksum_valid_birth_digits():
    result = process_experiment({"front": image_data(), "back": image_data("back")},
        Engine(front={**FRONT, "Date_of_birth": "02/02/1998"}), Reader([], []), Aligner(),
        MRZ(mrz_text().replace("M361025", "M3G1025")), align=False)
    assert result["sides"]["back"]["mrz"]["birth_date_conflict"]
    assert "date_of_birth" in result["review_fields"]


def test_both_uploads_are_validated_before_any_model_is_called():
    class NoCalls:
        def __getattr__(self, name):
            raise AssertionError("No backend calls permitted")
    with pytest.raises(InvalidImage):
        process_experiment({"front": image_data(), "back": b"invalid"}, NoCalls(), NoCalls(), NoCalls(), NoCalls())


class IsolatedBackend:
    def __init__(self):
        self.calls = []

    def describe(self):
        self.calls.append("health")
        return {"experimental": True, "available": True}

    def extract(self, images, **options):
        self.calls.append((images, options))
        return {"success": True, "experimental": True, "fields": {}}, 200


def test_main_health_and_extraction_never_contact_experiment_worker():
    lab = IsolatedBackend()
    engine, reader = Engine(), Reader()
    with TestClient(create_app(engine=engine, qr_reader=reader, gpu_engine=engine, experiment_backend=lab)) as client:
        health = client.get("/api/health").json()
        assert "experimental" not in health and lab.calls == []
        response = client.post("/api/cccd/extract", files={
            "front_image": ("front.png", image_data()), "back_image": ("back.png", image_data("back"))}, data={"quality": "high"})
        assert response.status_code == 200 and response.json()["source"] == "qr_ocr"
        assert lab.calls == []
        assert client.get("/api/cccd/experiment/health").json()["experimental"]
        assert lab.calls == ["health"]


def test_experiment_uploads_and_options_are_isolated_and_response_is_not_cached():
    lab = IsolatedBackend()
    with TestClient(create_app(Engine(), Reader(), experiment_backend=lab)) as client:
        response = client.post("/api/cccd/experiment/extract", files={
            "front_image": ("front.png", image_data()), "back_image": ("back.png", image_data("back"))},
            data={"align": "false", "front_rotation": "90"})
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert lab.calls[0][1] == {"align": False, "front_rotation": 90, "back_rotation": 0}
        response = client.post("/api/cccd/experiment/extract", files={
            "front_image": ("front.png", image_data()), "back_image": ("back.png", image_data("back"))},
            data={"front_rotation": "45"})
        assert response.status_code == 422 and len(lab.calls) == 1


def test_experiment_client_handles_worker_absence_and_busy_status():
    assert ExperimentClient(base_url="").extract({})[1] == 503
    def handle(request):
        return httpx.Response(429, json={"success": False, "error": "busy"})
    client = ExperimentClient(base_url="http://worker", transport=httpx.MockTransport(handle))
    payload, status = client.extract({"front": image_data(), "back": image_data("back")})
    assert status == 429 and payload["error"] == "busy"
    client.close()


def test_internal_worker_validates_uploads_options_and_auth(monkeypatch):
    from identification_demo.experiment_worker import create_app as worker_app
    class Backend:
        def __init__(self):
            self.calls = []
        def initialize(self):
            pass
        def close(self):
            pass
        def describe(self):
            return {"experimental": True, "available": True}
        def extract(self, images, **options):
            self.calls.append(options)
            return {"success": True, "experimental": True, "fields": {}}
    backend = Backend()
    monkeypatch.setenv("CCCD_EXPERIMENT_API_KEY", "lab-test-key")
    with TestClient(worker_app(backend)) as client:
        assert client.get("/health").status_code == 401
        assert client.get("/health", headers={"Authorization": "Bearer lab-test-key"}).status_code == 200
        files = {"front_image": ("front.png", image_data()), "back_image": ("back.png", image_data("back"))}
        assert client.post("/extract", files=files).status_code == 401
        headers = {"Authorization": "Bearer lab-test-key"}
        response = client.post("/extract", files=files, data={"back_rotation": "270", "align": "false"}, headers=headers)
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert backend.calls == [{"align": False, "rotations": {"front": 0, "back": 270}}]
        assert client.post("/extract", files=files, data={"front_rotation": "45"}, headers=headers).status_code == 422
        assert len(backend.calls) == 1
