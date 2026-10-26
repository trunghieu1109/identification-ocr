from copy import deepcopy
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from identification_demo.docu_region_engine import DocURegionGPUEngine
from identification_demo.docu_region_worker import DocURegionLocalEngine
from identification_demo.documents import parse_back
from identification_demo.engine import OCRUnavailable
from identification_demo.expiry import parse_expiry
from identification_demo.field_recovery import STRATEGY, recover_date
from identification_demo.gpu_worker import DocULocalEngine, create_gpu_app
from identification_demo.web import create_app
from test_gpu_engine import READY
from test_ocr_experiments import full_engine, submit
from test_pipeline import Reader


def line(text, y=200, confidence=.98, x=100):
    return {"text": text, "box": [[x, y], [x+500, y], [x+500, y+30], [x, y+30]],
            "confidence": confidence}


@pytest.mark.parametrize("side,caption,value", [
    ("front", "Có giá trị đến", "25/10/2036"),
    ("front", "Date of expiry", "Không thời hạn"),
    ("back", "Ngày cấp / Date of issue", "15/03/2021"),
    ("back", "Ngày, tháng, năm", "Ngày 15 tháng 3 năm 2021"),
])
def test_missing_date_is_read_once_next_to_its_caption(side, caption, value):
    initial = [line(caption)]
    snapshot = deepcopy(initial)
    calls = []
    result, meta = recover_date(initial, (1200, 760), side,
                               lambda bounds: (calls.append(bounds) or [line(value, 250)], 1))
    assert len(calls) == 1 and calls[0][0] >= 0 and calls[0][3] <= 760
    assert meta["status"] == "recovered" and meta["reread_regions"] == 1
    assert meta["elapsed_ms"] >= 0 and initial == snapshot
    assert parse_expiry(result) if side == "front" else parse_back(result)["issue_date"] == "15/03/2021"


@pytest.mark.parametrize("rows,needed,reason", [
    ([line("Có giá trị đến: 25/10/2036")], None, "already_found"),
    ([line("Có giá trị đến: Không thời hạn")], None, "already_found"),
    ([line("Ngày sinh: 01/02/1999")], None, "label_not_found"),
    ([line("Có giá trị đến")], ["name"], "not_requested"),
    ([line("Có giá trị đến"), line("Date of expiry", 500)], None, "ambiguous_label"),
    ([line("Có giá trị đến: 25/10/2036"), line("Date of expiry: 25/10/2037", 500)], None, "conflict"),
])
def test_clear_or_ambiguous_input_does_not_spend_gpu_time(rows, needed, reason):
    result, meta = recover_date(rows, (1200, 760), "front", lambda _: pytest.fail("Unexpected inference"), needed)
    assert result is rows and meta["reason"] == reason and meta["status"] == "skipped"


@pytest.mark.parametrize("recovered", [
    [line("31/02/2036", 250)], [line("25/10/2036", 250, confidence=.4)],
    [line("Ngày sinh: 25/10/2036", 250)], [line("NGUYỄN VĂN THỬ", 250)],
    [line("Có giá trị đến: 25/10/2036", 250), line("Date of expiry: 25/10/2037", 320)],
])
def test_bad_or_conflicting_local_dates_preserve_initial_text(recovered):
    rows = [line("Có giá trị đến")]
    result, meta = recover_date(rows, (1200, 760), "front", lambda _: (recovered, len(recovered)))
    assert result is rows and meta["status"] == "not_found"


def test_invalid_numeric_read_can_be_corrected_without_changing_other_fields():
    rows = [line("Ngày sinh: 01/02/1999", 80), line("Có giá trị đến"), line("25/IO/2036", 250)]
    result, meta = recover_date(rows, (1200, 760), "front", lambda _: ([line("25/10/2036", 250)], 1))
    assert meta["status"] == "recovered"
    assert result[0] == rows[0] and result[1] == rows[1]
    assert parse_expiry(result)[0]["expiry_date"] == "25/10/2036"


def test_local_failure_keeps_initial_fields():
    def fail(_bounds):
        raise RuntimeError("Synthetic inference failure")
    rows = [line("Có giá trị đến")]
    result, meta = recover_date(rows, (1200, 760), "front", fail)
    assert result is rows and meta["status"] == "error"


def test_worker_bounds_local_reads_and_does_not_add_expiry_to_full_image_retries(monkeypatch):
    rows = [line("Có giá trị đến")]
    monkeypatch.setattr(DocULocalEngine, "_document_payload", lambda *_args, **_kwargs: {
        "success": True, "text_lines": rows, "extracted_fields": {}, "doc_type": "cccd_front"})
    monkeypatch.setattr("identification_demo.docu_region_worker.detect_side", lambda _: "front")
    engine = DocURegionLocalEngine()
    engine._stage_timings, engine._work_counts = {}, {"reread_regions": 0}
    engine._field_needed = ["expiry_date"]
    calls = []
    engine._read_authority_region = lambda image, bounds: (calls.append(bounds) or [], 0)
    doc = SimpleNamespace(cropped_image=np.zeros((760, 1200, 3)))
    try:
        engine._document_payload(doc, "front")
        engine._document_payload(doc, "front")
        assert len(calls) == 1
        seen = []
        monkeypatch.setattr(DocULocalEngine, "_extract_document", lambda self, path, side, needed: seen.append(needed))
        engine._extract_document("unused", "front", ["name", "expiry_date"])
        assert seen == [["name"]] and engine._field_needed == ["name", "expiry_date"]
    finally:
        engine.close()


def test_new_http_adapter_requires_experimental_worker_and_passes_expiry():
    seen = []
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={**READY, "strategy": STRATEGY})
        seen.append(request.url.params["fields"])
        side = request.url.params["doc_type_hint"]
        return httpx.Response(200, json={
            "success": True, "doc_type": side, "extracted_fields": {},
            "text_lines": [line("Có giá trị đến: 25/10/2036")] if side == "cccd_front" else [],
            "timings_ms": {"field_refinement": 12.5},
            "field_recovery": {"strategy": STRATEGY, "target": "expiry_date", "status": "recovered",
                               "reread_regions": 1, "elapsed_ms": 12.5},
        })
    adapter = DocURegionGPUEngine("http://gpu.test", httpx.MockTransport(handler))
    app = create_app(full_engine(), Reader(), docu_region_engine=adapter)
    result = submit(TestClient(app), "docu_gpu_regions", use_qr="false").json()
    assert result["engine"] == "docu_gpu_regions" and result["ocr_engine"]["strategy"] == STRATEGY
    assert "expiry_date" in seen[0] and "expiry_date" not in seen[1]
    assert result["ocr"]["front"]["field_recovery"]["status"] == "recovered"
    assert result["ocr"]["front"]["timings_ms"]["field_refinement"] == 12.5
    wrong = DocURegionGPUEngine("http://gpu.test", httpx.MockTransport(lambda _: httpx.Response(200, json=READY)))
    assert not wrong.describe()["available"]
    with pytest.raises(OCRUnavailable):
        wrong._ready(wrong._client())


def test_main_api_does_not_use_the_new_worker():
    trial = full_engine()
    client = TestClient(create_app(full_engine(), Reader(), full_engine(), docu_region_engine=trial))
    from test_pipeline import image_data
    result = client.post("/api/cccd/extract", data={"quality": "high"}, files={
        "front_image": ("front.png", image_data(), "image/png"),
        "back_image": ("back.png", image_data("back"), "image/png"),
    }).json()
    assert result["total_fields"] == 12 and result["fields"]["expiry_date"] == "25/10/2036" and not trial.calls


@pytest.mark.parametrize("experimental", [False, True])
def test_expiry_whitelist_extension_is_scoped_to_the_experimental_front_worker(experimental):
    class Worker:
        def initialize(self):
            pass

        def describe(self):
            return {"pipeline_ready": True}

        def extract_image(self, image, side, needed_fields=None):
            return {"success": True, "fields_requested": needed_fields}

    from test_pipeline import image_data
    with TestClient(create_gpu_app(Worker(), extra_front_fields=("expiry_date",) if experimental else ())) as client:
        files = {"file": ("card.png", image_data(), "image/png")}
        front = client.post("/api/v1/extract", params={"doc_type_hint": "cccd_front", "fields": "expiry_date"}, files=files)
        assert front.status_code == (200 if experimental else 422)
        assert client.post("/api/v1/extract", params={"doc_type_hint": "cccd_back", "fields": "expiry_date"}, files=files).status_code == 422


def test_standalone_date_recovers_its_missing_caption_without_guessing_its_meaning():
    rows = [line("Ngày sinh: 01/02/1999", 20), line("Nơi thường trú: Hà Nội",200), line("25/10/2036", 250)]
    calls = []
    result, metadata = recover_date(rows, (1200,760), "front",
        lambda bounds: (calls.append(bounds) or [line("Có giá trị đến", 220)], 1))
    assert len(calls) == 1 and calls[0][1] <= 200
    assert metadata['anchor'] == 'unassigned_date' and metadata['status'] == 'recovered'
    assert parse_expiry(result)[0]['expiry_date'] == '25/10/2036'
    unchanged, metadata = recover_date(rows, (1200,760), "front", lambda _: ([line("25/10/2036",250)],1))
    assert unchanged is rows and metadata['status'] == 'not_found'


def test_standalone_birth_date_is_not_used_as_an_expiry_anchor():
    rows = [line('Ngày sinh',200),line('01/02/1999',250)]
    result, metadata = recover_date(rows,(1200,760),'front',lambda _: pytest.fail('Birth date is already owned'))
    assert result is rows and metadata['status']=='skipped'


def test_split_caption_and_split_numeric_value_are_kept_for_geometry_parser():
    rows=[{'text':'Có giá','box':[[100,200],[170,200],[170,230],[100,230]]},
          {'text':'trị đến','box':[[180,200],[280,200],[280,230],[180,230]]}]
    recovered=[{'text':'25/10/','box':[[100,250],[195,250],[195,280],[100,280]],'confidence':.98},
               {'text':'2036','box':[[200,250],[270,250],[270,280],[200,280]],'confidence':.98}]
    result,metadata=recover_date(rows,(1200,760),'front',lambda _:(recovered,2))
    assert metadata['status']=='recovered' and parse_expiry(result)[0]['expiry_date']=='25/10/2036'


def test_namlhn_region_reader_preserves_seed_boxes_and_uses_absolute_margins():
    from PIL import Image
    from identification_demo.namlhn_text_processing import prepare_text_image, read_text_region
    image=np.full((200,1200,3),220,dtype=np.uint8)
    prepared=prepare_text_image(image)
    assert max(prepared.shape[:2])<=2560 and image.shape==(200,1200,3)
    item=SimpleNamespace(text='CỤC CẢNH SÁT',confidence=.99,field_key='text_line',
                         bbox=SimpleNamespace(coordinates=[(100,30),(600,30),(600,60),(100,60)]))
    detector=SimpleNamespace(imgsz=640,conf_threshold=.25,detect_single_image=lambda _image:[])
    crops=[]
    def crop(image,coordinates,**padding):
        crops.append(padding)
        return Image.new('RGB',(500,40),'white')
    recognizer=SimpleNamespace(WHOLE_CARD_CLASSES=set(),crop_oriented_box=crop,
        predictor=SimpleNamespace(config={'dataset':{'image_max_width':512}},
            predict_batch=lambda patches,return_prob:(['CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH'],[.99])))
    result,count=read_text_region(detector,recognizer,image,(0,0,1200,180),[item])
    assert count==1 and result[0].text.endswith('HÀNH CHÍNH')
    assert crops[0]['pad_top']==crops[0]['pad_bot'] and crops[0]['pad_top']>.01
    assert detector.imgsz==640 and detector.conf_threshold==.25
    assert recognizer.predictor.config['dataset']['image_max_width']==512
    assert item.text=='CỤC CẢNH SÁT'
