from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from identification_demo.authority import incomplete_authority
from identification_demo.authority_recovery import recover_authority, recovery_bounds
from identification_demo.authority_region import select_authority_region
from identification_demo.gpu_engine import normalize_result
from identification_demo.gpu_worker import DocULocalEngine
from test_regions import fake_engine


AGENCY = "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"


def line(text, x=700, y=300, width=600, confidence=.98):
    return SimpleNamespace(text=text, confidence=confidence, field_key="text",
                           bbox=SimpleNamespace(coordinates=[(x, y), (x + width, y),
                                                             (x + width, y + 30), (x, y + 30)]))


@pytest.mark.parametrize("text", ["CỤC CẢNH SÁT", "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH",
                                      "CỤC CẢNH SÁT VỀ TRẬT TỰ XÃ HỘI"])
def test_truncated_names_remain_visible_and_are_marked_partial(text):
    assert incomplete_authority(text)
    payload = {"success": True, "doc_type": "cccd_back", "extracted_fields": {},
               "text_lines": ["Ngày cấp: 20/04/2021", text]}
    result = normalize_result(payload)
    assert result["structured_fields"]["issuing_authority"] == text
    assert result["authority_region"]["status"] == "partial"
    assert DocULocalEngine._quality(payload, "back", ["issuing_authority"])[:2] == (1, 0)


@pytest.mark.parametrize("text", ["CÔNG AN TỈNH", "", "CỤC TRƯỞNG", "VỀ TRẬT TỰ XÃ HỘI"])
def test_unidentifiable_fragments_still_do_not_invent_an_institution(text):
    payload = {"success": True, "doc_type": "cccd_back", "extracted_fields": {}, "text_lines": [text]}
    assert not normalize_result(payload)["structured_fields"]["issuing_authority"]


@pytest.mark.parametrize("text", [AGENCY, "CÔNG AN TỈNH HÀ NAM", "CỤC QUẢN LÝ XUẤT NHẬP CẢNH",
                                      "CỤC CẢNH SÁT ĐKQL CƯ TRÚ VÀ DLQG VỀ DÂN CƯ"])
def test_complete_or_different_institutions_do_not_need_recovery(text):
    assert not incomplete_authority(text)
    items = [line("Cơ quan cấp: " + text)]
    assert recover_authority((1600, 900), items, lambda _: pytest.fail("Already complete")) is None


def test_missed_continuation_is_detected_once_without_altering_date_or_other_columns():
    date = line("Ngày cấp: 20/04/2021", y=200)
    other_column = line("CÔNG AN TỈNH KHÁC", x=50, y=330, width=400)
    items = [date, other_column, line("CỤC TRƯỞNG CỤC CẢNH SÁT")]
    calls = []

    def read(bounds):
        calls.append(bounds)
        return [line("CỤC TRƯỞNG CỤC CẢNH SÁT"),
                line("QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", y=345),
                line("Thiếu tướng NGUYỄN VĂN MẪU", y=400)], 3

    metadata = recover_authority((1600, 900), items, read)
    assert len(calls) == 1 and calls[0][0] > 400
    assert select_authority_region(items).text == AGENCY
    assert date in items and other_column in items
    assert all("NGUYỄN" not in item.text for item in items)
    assert metadata["reread_regions"] == 3 and metadata["recovered_lines"] == 2
    assert recover_authority((1600, 900), items, lambda _: pytest.fail("Duplicate read")) is None


def test_recovery_replaces_a_clipped_box_instead_of_duplicating_its_words():
    items = [line("CỤC TRƯỞNG", y=250), line("CỤC CẢNH SÁT", x=850, width=300)]
    result = recover_authority((1600, 900), items,
                              lambda _: ([line("CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH"),
                                          line("VỀ TRẬT TỰ XÃ HỘI", y=345)], 2))
    assert result["recovered_lines"] == 2
    assert select_authority_region(items).text == AGENCY


def test_unique_title_can_anchor_recovery_even_if_all_body_lines_were_missed():
    items = [line("CỤC TRƯỞNG", y=250, x=900, width=250)]
    assert recovery_bounds((1600, 900), items)
    recover_authority((1600, 900), items,
                      lambda _: ([line("CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH"),
                                  line("VỀ TRẬT TỰ XÃ HỘI", y=345)], 2))
    assert select_authority_region(items).text == AGENCY


def test_no_title_or_institution_does_not_invent_a_card_location():
    assert recovery_bounds((1600, 900), [line("NGUYỄN VĂN MẪU")]) is None


def test_conflicting_anchors_do_not_trigger_a_crop():
    items = [line("CỤC TRƯỞNG", x=50, width=250), line("CỤC TRƯỞNG", x=900, width=250)]
    assert recovery_bounds((1600, 900), items) is None


def test_recovery_covers_lines_below_the_title_and_shifts_inside_card_edges():
    rows = [line("CỤC TRƯỞNG CỤC CẢNH SÁT", x=1200, y=250, width=300)]
    left, top, right, bottom = recovery_bounds((1600, 900), rows)
    assert right - left >= 1055 and right == 1600
    assert top < 250 and bottom >= 460
    # A continuation around y=440 was outside the previous 4-line-height ROI.
    assert left < 900 < right and top < 440 < bottom


def test_recovery_crop_limits_can_be_changed_without_restarting(monkeypatch):
    import identification_demo.authority_recovery as module
    rows = [line("CỤC TRƯỞNG CỤC CẢNH SÁT", x=800, y=200, width=100)]
    default = recovery_bounds((1600, 900), rows)
    monkeypatch.setattr(module, 'authority_rules', lambda: {'recovery_crop': {
        'min_width_ratio': .8, 'below_lines': 9, 'above_lines': 1, 'min_below_height_ratio': .25}})
    wide = recovery_bounds((1600, 900), rows)
    assert wide[2] - wide[0] > default[2] - default[0]
    assert wide[3] > default[3] and wide[1] < default[1]


def test_expanded_line_boxes_do_not_expand_the_search_region_a_second_time():
    rows = [line('CỤC TRƯỞNG CỤC CẢNH SÁT', x=950, y=250, width=300)]
    bounds = recovery_bounds((1600, 900), rows)
    rows[0].bbox.coordinates = [(200, 250), (1550, 250), (1550, 280), (200, 280)]
    calls = []
    recover_authority((1600, 900), rows, lambda box: (calls.append(box) or [], 0), bounds)
    assert calls == [bounds]


def test_unreadable_continuation_does_not_trigger_four_full_image_rotations():
    engine = fake_engine([])
    rotations = []

    def read(_path, _side, rotation):
        rotations.append(rotation)
        return {"success": True, "doc_type": "cccd_back", "extracted_fields": {},
                "text_lines": ["Ngày cấp: 20/04/2021", "CỤC TRƯỞNG CỤC CẢNH SÁT"],
                "enhancement": {"target": "issuing_authority", "recovery_attempted": True}}

    engine._read_full_image = read
    result = engine.extract_document("unused.png", "back", ["issue_date", "issuing_authority"])
    assert rotations == [0]
    assert normalize_result(result)["structured_fields"]["issuing_authority"] == "CỤC CẢNH SÁT"
    assert normalize_result(result)["authority_region"]["status"] == "partial"
    engine.close()


def test_noisier_crop_or_rotation_cannot_beat_an_equal_candidate_with_partial_institution():
    partial = {"success": True, "doc_type": "cccd_back", "extracted_fields": {},
               "text_lines": ["Ngày cấp: 20/04/2021", "CỤC TRƯỞNG CỤC CẢNH SÁT"]}
    empty = {"success": True, "doc_type": "cccd_back", "extracted_fields": {},
             "text_lines": ["Ngày cấp: 20/04/2021"] + ["NGUYỄN VĂN MẪU"] * 20}
    for needed in (["issuing_authority"], ["issue_date", "issuing_authority"]):
        assert DocULocalEngine._quality(partial, "back", needed) > DocULocalEngine._quality(empty, "back", needed)


@pytest.mark.parametrize("recovered", [[], [line("NGUYỄN VĂN MẪU", y=345)],
                                      [line(AGENCY, confidence=.4)],
                                      [line("QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", x=0, y=345)]])
def test_bad_or_other_column_recovery_preserves_original_without_filling_a_default(recovered):
    items = [line("CỤC TRƯỞNG CỤC CẢNH SÁT")]
    original = list(items)
    result = recover_authority((1600, 900), items, lambda _: (recovered, len(recovered)))
    assert items == original and result["recovered_lines"] == 0


@pytest.mark.parametrize("original", ["CỤC CẢNH SÁT", AGENCY])
@pytest.mark.parametrize("failure", ["exception", "erase", "weaker"])
def test_optional_enhancement_cannot_erase_initial_institution(monkeypatch, original, failure):
    import sys
    from types import ModuleType
    import identification_demo.gpu_worker as worker

    for module in ("core", "core.postprocessing", "core.postprocessing.administrative_corrector"):
        monkeypatch.setitem(sys.modules, module, ModuleType(module))
    sys.modules["core.postprocessing.administrative_corrector"].AdministrativeCorrector = SimpleNamespace(PROVINCES=[])
    items = [line("Ngày cấp: 20/04/2021", y=200), line("CỤC TRƯỞNG", y=250), line(original)]
    doc = SimpleNamespace(recognized_texts=items, doc_type="cccd_back", cropped_image=np.zeros((900, 1600, 3)),
                          extracted_fields={})
    extractor = SimpleNamespace(sort_reading_order=lambda rows, **kwargs: rows, extract=lambda *_: {})
    engine = DocULocalEngine()
    engine._pipeline = SimpleNamespace(recognizer=object(), extractor=SimpleNamespace(extractors={"cccd_back": extractor}))
    dedicated_reader = object()
    engine._authority_recognizer = dedicated_reader

    def damage(_recognizer, _image, rows, **kwargs):
        assert _recognizer is dedicated_reader
        rows[2].text = "VỀ TRẬT TỰ XÃ HỘI" if failure != "weaker" else "CỤC CẢNH SÁT"
        if failure == "exception":
            raise RuntimeError("Optional reread failed")
        return {"target": "issuing_authority", "changed_lines": 1, "reread_lines": 1, "reread_regions": 2}

    monkeypatch.setattr(worker, "refine_authority", damage)
    monkeypatch.setattr(worker, "recover_authority", lambda *_: None)
    result = normalize_result(engine._document_payload(doc, "back"))
    assert result["structured_fields"]["issuing_authority"] == original
    assert result["structured_fields"]["issue_date"] == "20/04/2021"
    assert (result.get("authority_region", {}).get("status") == "partial") == (original != AGENCY)
    engine.close()


@pytest.mark.parametrize("fail", [None, "detect", "recognize"])
@pytest.mark.parametrize("dedicated", [False, True])
def test_worker_reuses_models_maps_roi_coordinates_and_restores_settings(fail, dedicated):
    engine = DocULocalEngine()
    detector = SimpleNamespace(imgsz=640, conf_threshold=.25)
    cropped = []
    patches = []

    def detect(image):
        cropped.append(image.shape)
        assert detector.imgsz == 960 and detector.conf_threshold == .15
        if fail == "detect":
            raise RuntimeError("bad detector")
        return [line("", x=10, y=20, width=300)]

    def crop(image, coordinates, **kwargs):
        assert coordinates[0] == (710, 220)
        assert kwargs['pad_x'] <= .1  # A wide search must not turn each short line into a mostly blank patch.
        patch = Image.new("RGB", (300, 30))
        patches.append(patch)
        return patch

    def predict(images, return_prob):
        assert predictor.config["dataset"]["image_max_width"] == 1024
        if fail == "recognize":
            raise RuntimeError("bad recognizer")
        return [AGENCY], [.96]

    detector.detect_single_image = detect
    predictor = SimpleNamespace(config={"dataset": {"image_max_width": 512}}, predict_batch=predict)
    recognizer = SimpleNamespace(predictor=predictor, crop_oriented_box=crop, WHOLE_CARD_CLASSES={"cccd"})
    engine._pipeline = SimpleNamespace(detector=detector, recognizer=object() if dedicated else recognizer)
    if dedicated:
        engine._authority_recognizer = recognizer
    result, count = engine._read_authority_region(np.zeros((900, 1600, 3)), (700, 200, 1300, 450))
    assert cropped == [(250, 600, 3)]
    assert detector.imgsz == 640 and detector.conf_threshold == .25
    assert predictor.config["dataset"]["image_max_width"] == 512
    assert count == (0 if fail == "detect" else 1)
    if fail:
        assert not result
    else:
        assert result[0].text == AGENCY and result[0].bbox.coordinates[0][0] < 710
    for patch in patches:
        with pytest.raises(ValueError):
            patch.getpixel((0, 0))
    engine.close()
