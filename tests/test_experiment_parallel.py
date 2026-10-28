"""Parallel scheduling must preserve complete OCR fallback and field priority."""
from threading import Event

from PIL import Image

from identification_demo.documents import FRONT_FIELDS, BACK_FIELDS
from identification_demo.experiment_mrz import parse_vietnam_td1
from identification_demo.experiment_mrz_reader import MRZReader
from identification_demo.experiment_pipeline import process_experiment
from test_cccd_experiment import Aligner, MRZ, mrz_text
from test_pipeline import Engine, Reader, image_data, PAYLOAD, FRONT


def inputs():
    return {"front": image_data(), "back": image_data("back")}


class RecordingEngine(Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requested = {}

    def extract_image(self, image, side, needed_fields=None, source_data=None):
        self.calls.append(side)
        self.requested[side] = list(needed_fields)
        return self.results[side]


def test_no_qr_no_mrz_still_reads_all_fields_on_both_sides():
    engine = RecordingEngine()
    result = process_experiment(inputs(), engine, Reader([], []), Aligner(), MRZ("invalid MRZ"), align=False)
    assert engine.calls == ["front", "back"]
    assert set(engine.requested["front"]) == {key for key, *_ in FRONT_FIELDS} | {"expiry_date"}
    assert {key for key, *_ in BACK_FIELDS} <= set(engine.requested["back"])
    assert result["fields"]["name"] == FRONT["Name"]
    assert all(result["fields"][key] for key, *_ in FRONT_FIELDS)
    assert all(result["fields"][key] for key, *_ in BACK_FIELDS)
    assert result["fields"]["expiry_date"] == "25/10/2036"
    assert all(source and source.startswith("ocr") for key, source in result["field_sources"].items() if result["fields"][key])


def test_qr_and_mrz_priority_is_independent_of_completion_order():
    result = process_experiment(inputs(), RecordingEngine(), Reader([PAYLOAD], []), Aligner(), MRZ(), align=False)
    assert result["field_sources"]["name"] == "qr_front_original"
    assert result["fields"]["name"] == "NGUYỄN VĂN THỬ"
    assert result["field_sources"]["expiry_date"] == "mrz_back"
    assert result["field_sources"]["place_of_origin"].startswith("ocr_front")
    assert result["field_sources"]["issuing_authority"].startswith("ocr_back")
    # OCR returned a different DOB/name; reliable QR values are not mixed in.
    assert not any(item["field"] in {"name", "date_of_birth"} and item["source"].startswith("ocr") for item in result["candidates"])


def test_qr_mrz_and_front_ocr_actually_overlap_without_detached_inference():
    mrz_started, ocr_started, mrz_finished = Event(), Event(), Event()

    class ConcurrentQR(Reader):
        def read(self, image):
            assert mrz_started.wait(3), "MRZ was not started alongside QR"
            assert ocr_started.wait(3), "OCR waited for QR to finish"
            return []

    class ConcurrentMRZ:
        def read(self, image, **options):
            mrz_started.set()
            assert ocr_started.wait(3), "OCR waited for MRZ to finish"
            mrz_finished.set()
            return parse_vietnam_td1(mrz_text())

    class ConcurrentOCR(RecordingEngine):
        def extract_image(self, image, side, **options):
            if side == "front":
                assert mrz_started.wait(3) and not mrz_finished.is_set()
                ocr_started.set()
            return super().extract_image(image, side, **options)

    result = process_experiment(inputs(), ConcurrentOCR(), ConcurrentQR([], []), Aligner(), ConcurrentMRZ(), align=False)
    assert result["execution"]["mode"] == "parallel_qr_mrz_ocr"
    assert mrz_finished.is_set()
    timeline = result["stage_intervals_ms"]
    front = timeline["ocr_front_original"]
    mrz = timeline["mrz_back"]
    assert max(mrz["start_ms"], front["start_ms"]) <= min(mrz["end_ms"], front["end_ms"])
    qr = timeline["qr_front_original"]
    assert max(qr["start_ms"], front["start_ms"]) <= min(qr["end_ms"], front["end_ms"])


def test_unreadable_mrz_failure_does_not_suppress_ocr():
    class BrokenMRZ:
        def read(self, *args, **kwargs):
            raise RuntimeError("model unavailable")
    engine = RecordingEngine()
    result = process_experiment(inputs(), engine, Reader([], []), Aligner(), BrokenMRZ(), align=False)
    assert engine.calls == ["front", "back"]
    assert result["sides"]["back"]["mrz"]["status"] == "error"
    assert result["fields"]["id_number"] == FRONT["ID_number"]
    assert result["fields"]["name"] == FRONT["Name"]


def test_full_ocr_coverage_cancels_only_additional_mrz_attempts():
    seen = Event()
    # Make the printed front result complete for all possible MRZ fields.
    engine = RecordingEngine()
    class CancellableMRZ:
        def read(self, image, *, stop_event, **options):
            assert stop_event.wait(3), "Complete printed fields did not stop MRZ retries"
            seen.set()
            return parse_vietnam_td1("")
    result = process_experiment(inputs(), engine, Reader([PAYLOAD], []), Aligner(), CancellableMRZ(), align=False)
    assert seen.is_set() and engine.calls == ["front", "back"]
    assert result["fields"]["expiry_date"] == "25/10/2036"


def test_ocr_values_alone_do_not_cancel_mrz_identity_recovery():
    engine = RecordingEngine()
    class UncertainMRZ:
        def read(self, image, *, stop_event, **options):
            # Wait for both printed passes without trusting their ID merely
            # because it is nonempty. No QR means MRZ recovery keeps its budget.
            import time
            deadline = time.monotonic() + 3
            while len(engine.calls) < 2 and time.monotonic() < deadline:
                time.sleep(.005)
            assert len(engine.calls) == 2 and not stop_event.is_set()
            return parse_vietnam_td1(mrz_text())
    result = process_experiment(inputs(), engine, Reader([], []), Aligner(), UncertainMRZ(), align=False)
    assert result["fields"]["id_number"] == "001099000001"
    assert result["field_sources"]["id_number"] == "mrz_back"


def fake_mrz_reader(first="", bottom=""):
    reader = MRZReader.__new__(MRZReader)
    reader.calls = []
    def fast(pixels):
        reader.calls.append("fastmrz")
        return first
    def last(pixels):
        reader.calls.append("bottom_region")
        return bottom
    reader._fast, reader._bottom = fast, last
    return reader


def test_partial_mrz_stops_once_protected_fields_are_available():
    reader = fake_mrz_reader(mrz_text().replace("M361025", "M3G1025"))
    result = reader.read(Image.new("RGB", (300, 190)))
    assert reader.calls == ["fastmrz"]
    assert result["stop_reason"] == "usable_fields" and result["status"] == "partial"
    assert result["fields"]["id_number"] == "001099000001"


def test_cancelled_retry_still_performs_primary_identity_read():
    reader = fake_mrz_reader()
    stop = Event(); stop.set()
    result = reader.read(Image.new("RGB", (300, 190)), stop_event=stop)
    assert reader.calls == ["fastmrz"]
    assert result["stop_reason"] == "covered_by_qr_ocr"


def test_mrz_attempt_budget_covers_original_and_aligned_images_together():
    reader = fake_mrz_reader()
    result = reader.read(Image.new("RGB", (300, 190)), variant="aligned", fallback_image=Image.new("RGB", (600, 400)), budget_ms=10000)
    assert len(reader.calls) == 3
    assert [item["variant"] for item in result["attempts"]] == ["aligned", "original", "original"]
    assert result["stop_reason"] == "attempt_limit"


def test_mrz_soft_time_budget_prevents_starting_another_attempt():
    reader = fake_mrz_reader()
    result = reader.read(Image.new("RGB", (300, 190)), budget_ms=0)
    assert reader.calls == ["fastmrz"] and result["stop_reason"] == "time_budget"


def test_no_region_tries_opposite_orientation_and_its_bottom_band_within_three_reads():
    reader = fake_mrz_reader("", mrz_text())
    result = reader.read(Image.new("RGB", (300, 190)), budget_ms=10000)
    assert reader.calls == ["fastmrz", "fastmrz", "bottom_region"]
    assert [item["rotation_clockwise"] for item in result["attempts"]] == [0, 180, 180]
    assert result["status"] == "verified" and result["rotation_clockwise"] == 180
