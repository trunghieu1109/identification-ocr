from types import SimpleNamespace

import pytest

from identification_demo.engine import OCRMultipleDocuments
from identification_demo.gpu_worker import DocULocalEngine
from identification_demo.regions import deduplicate_documents


def document(name, points=None, issue_date="2021-03-15"):
    return SimpleNamespace(name=name, polygon=SimpleNamespace(coordinates=points) if points else None,
                           doc_type="cccd_back", cropped_image=None, recognized_texts=[],
                           extracted_fields={"issue_date": SimpleNamespace(normalized_value=issue_date, is_valid=True)})


def box(left, top, right, bottom):
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def test_duplicate_card_crops_keep_largest_region_instead_of_losing_edge_text():
    full = document("full", box(0, 0, 100, 60))
    smaller = document("smaller", box(3, 2, 97, 58))
    repeated = document("same", box(0, 0, 100, 60))
    assert deduplicate_documents([smaller, full, repeated]) == [full]


@pytest.mark.parametrize("second", [box(150, 0, 250, 60), box(50, 0, 150, 60), box(40, 20, 60, 40)])
def test_separate_partly_overlapping_or_small_nested_regions_remain_ambiguous(second):
    documents = [document("one", box(0, 0, 100, 60)), document("two", second)]
    assert len(deduplicate_documents(documents)) == 2


def test_intersection_uses_polygons_instead_of_axis_aligned_boxes():
    first = document("diagonal", [(0, 0), (10, 0), (100, 90), (100, 100)])
    second = document("other diagonal", [(90, 0), (100, 0), (10, 100), (0, 100)])
    assert len(deduplicate_documents([first, second])) == 2


def test_absent_or_degenerate_polygons_are_not_silently_merged():
    missing = document("missing")
    degenerate = document("invalid", [(0, 0), (1, 1), (2, 2)])
    assert deduplicate_documents([missing, degenerate]) == [missing, degenerate]
    assert deduplicate_documents([]) == []


def fake_engine(documents, recovery=None):
    engine = DocULocalEngine()
    engine._pipeline = SimpleNamespace(
        process_image=lambda *_args, **_kwargs: SimpleNamespace(success=True, documents=documents),
        extractor=SimpleNamespace(extractors={key: SimpleNamespace(
            sort_reading_order=lambda items, **_kwargs: items,
            extract=lambda items, _image: next(doc.extracted_fields for doc in documents if doc.recognized_texts is items),
        ) for key in ("cccd_back", "cccd_front")}),
    )
    engine._read_full_image = lambda *_args: recovery
    return engine


def test_real_worker_selects_deduplicated_card_and_releases_lock():
    full = document("full", box(0, 0, 100, 60))
    crop = document("crop", box(3, 2, 97, 58), issue_date="2000-01-01")
    engine = fake_engine([crop, full])
    result = engine.extract_document("unused.png", "back")
    assert result["extracted_fields"]["issue_date"] == "2021-03-15"
    assert not engine._lock.locked()


def readable_back(name, points, date="2021-03-15", complete=True):
    doc = document(name, points, date)
    doc.recognized_texts = [SimpleNamespace(text=f"Ngày cấp: {date[8:10]}/{date[5:7]}/{date[:4]}"),
                            SimpleNamespace(text="Cơ quan cấp: Công an tỉnh Hà Nam")]
    if complete:
        doc.recognized_texts.append(SimpleNamespace(text="Đặc điểm nhân dạng: Nốt ruồi má trái"))
    return doc


def test_distinct_readable_conflicting_cards_report_error_without_selecting_a_person():
    engine = fake_engine([readable_back("one", box(0, 0, 100, 60)), readable_back("two", box(150, 0, 250, 60), "2022-04-20")])
    with pytest.raises(OCRMultipleDocuments) as caught:
        engine.extract_document("unused.png", "back")
    assert caught.value.detected_regions == 2
    assert not engine._lock.locked()


FULL_BACK = {
    "success": True, "doc_type": "cccd_back", "extracted_fields": {
        "issue_date": "2021-03-15", "characteristics": "Nốt ruồi má trái",
    }, "text_lines": [
        {"text": "Ngày cấp: 15/03/2021"}, {"text": "Cơ quan cấp: Công an tỉnh Hà Nam"},
        {"text": "Đặc điểm nhân dạng: Nốt ruồi má trái"},
    ],
}


def test_two_noisy_regions_trigger_real_full_image_ocr_instead_of_multiple_cards_error():
    calls = []
    engine = fake_engine([document("fragment one", box(0, 0, 100, 60)), document("fragment two", box(150, 0, 250, 60))])
    def recover(path, side, rotation):
        calls.append((path, side, rotation))
        return {**FULL_BACK}
    engine._read_full_image = recover
    result = engine.extract_document("original.png", "back")
    assert calls == [("original.png", "back", 0)]
    assert result["extracted_fields"]["issue_date"] == "2021-03-15"
    assert result["recovery"] == {"method": "full_image", "detected_regions": 2, "rotation_degrees": 0, "attempted_rotations": [0]}


def test_single_badly_rotated_region_is_recovered_from_original_image():
    engine = fake_engine([document("upside down", box(0, 0, 100, 60))], {**FULL_BACK})
    result = engine.extract_document("original.png", "back")
    assert result["recovery"]["method"] == "full_image"
    assert result["recovery"]["rotation_degrees"] == 0


def test_regions_without_readable_card_evidence_do_not_cause_multiple_cards_error():
    engine = fake_engine([document("noise", box(0, 0, 100, 60)), document("noise2", box(150, 0, 250, 60))])
    result = engine.extract_document("original.png", "back")
    assert result["success"]
    assert result["recovery"]["attempted_rotations"] == [0, 180, 90, 270]


def test_repeated_metadata_in_separate_predicted_regions_is_not_proof_of_two_cards():
    engine = fake_engine([readable_back("one", box(0, 0, 100, 60)), readable_back("two", box(150, 0, 250, 60))], {**FULL_BACK})
    result = engine.extract_document("original.png", "back")
    assert result["recovery"]["method"] == "full_image"


def test_partial_crops_with_inconsistent_ocr_still_retry_the_original_photo():
    engine = fake_engine([readable_back("part one", box(0, 0, 100, 60), complete=False),
                          readable_back("part two", box(150, 0, 250, 60), "2022-04-20", complete=False)], {**FULL_BACK})
    result = engine.extract_document("original.png", "back")
    assert result["recovery"]["method"] == "full_image"
    assert result["extracted_fields"]["issue_date"] == "2021-03-15"


def test_retry_rotations_stop_when_expected_side_has_readable_fields():
    engine = fake_engine([])
    rotations = []
    def recover(_path, _side, rotation):
        rotations.append(rotation)
        return {**FULL_BACK} if rotation == 180 else None
    engine._read_full_image = recover
    result = engine.extract_document("original.png", "back")
    assert rotations == [0, 180]
    assert result["recovery"]["rotation_degrees"] == 180


def test_two_readable_fronts_with_different_ids_are_not_combined():
    first = document("first", box(0, 0, 100, 60))
    second = document("second", box(150, 0, 250, 60))
    for doc, number, name in ((first, "001099000001", "NGUYỄN VĂN MỘT"), (second, "001099000002", "NGUYỄN VĂN HAI")):
        doc.doc_type = "cccd_front"
        doc.recognized_texts = [SimpleNamespace(text="Họ và tên"), SimpleNamespace(text="Ngày sinh"),
                                SimpleNamespace(text="Giới tính: Nam"), SimpleNamespace(text="Quốc tịch: Việt Nam")]
        values = {"id_number": number, "full_name": name, "dob": "1999-02-01", "gender": "Nam", "nationality": "Việt Nam"}
        doc.extracted_fields = {key: SimpleNamespace(normalized_value=value, is_valid=True) for key, value in values.items()}
    with pytest.raises(OCRMultipleDocuments):
        fake_engine([first, second]).extract_document("two-fronts.png", "front")
