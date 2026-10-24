import json
from types import SimpleNamespace

import pytest
from PIL import Image

from identification_demo.authority import normalize_authority
from identification_demo.authority_recognition import refine_authority
from identification_demo.documents import parse_back
from identification_demo.image_enhancement import enhance_text_patch, trim_text_band


CANONICAL = "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"


@pytest.mark.parametrize("observed", [
    CANONICAL,
    "cuc truong cuc canh sat quan ly hanh chinh ve trat tu xa hoi",
    "CỤC TRƯỞNG CỤC CẢNH SÁT Ý HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI",
])
def test_known_name_needs_enough_recognized_evidence(observed):
    assert normalize_authority(observed) == CANONICAL


@pytest.mark.parametrize("observed,expected", [
    ("", ""), ("CỤC TRƯỞNG CỤC CẢNH SÁT", "CỤC CẢNH SÁT"), ("BỘ CÔNG AN", "BỘ CÔNG AN"),
    ("Giám đốc Công an tỉnh Hà Nam", "Công an tỉnh Hà Nam"),
    ("CỤC TRƯỞNG CỤC CẢNH SÁT ĐKQL CƯ TRÚ VÀ DLQG VỀ DÂN CƯ", "CỤC CẢNH SÁT ĐKQL CƯ TRÚ VÀ DLQG VỀ DÂN CƯ"),
    ("CỤC QUẢN LÝ XUẤT NHẬP CẢNH", "CỤC QUẢN LÝ XUẤT NHẬP CẢNH"),
])
def test_unknown_or_truncated_names_do_not_become_a_default_agency(observed, expected):
    assert normalize_authority(observed) == expected


def test_editable_rules_reload_after_changes_and_support_disabling(tmp_path, monkeypatch):
    path = tmp_path / "rules.json"
    monkeypatch.setenv("ISSUING_AUTHORITY_RULES_FILE", str(path))
    name = "CÔNG AN TỈNH NAM ĐỊNH"
    rules = {"enabled": True, "min_similarity": 0.9, "authorities": [
        {"name": name, "required_fragments": ["CÔNG AN TỈNH", "NAM ĐỊNH"]},
    ]}
    path.write_text(json.dumps(rules))
    observed = "giam doc cong an tinh nam dinh"
    assert normalize_authority(observed) == name
    rules["enabled"] = False
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps(rules))
    replacement.replace(path)
    assert normalize_authority(observed) == "cong an tinh nam dinh"


@pytest.mark.parametrize("content", ["bad json", "[]", '{"min_similarity":0.1}', '{"authorities":[{}]}'])
def test_bad_rules_preserve_read_text_instead_of_breaking_ocr(tmp_path, monkeypatch, content):
    path = tmp_path / "rules.json"
    path.write_text(content)
    monkeypatch.setenv("ISSUING_AUTHORITY_RULES_FILE", str(path))
    assert normalize_authority("Công an tỉnh Hà Nam") == "Công an tỉnh Hà Nam"


def test_authority_spanning_many_lines_excludes_headers_signer_and_fingerprints():
    rows = ["BỘ CÔNG AN", "Đặc điểm nhân dạng", "Ngày cấp: 20/04/2021", "CỤC TRƯỞNG CỤC CẢNH SÁT",
            "QUẢN LÝ HÀNH CHÍNH", "VỀ TRẬT TỰ", "XÃ HỘI", "NGUYỄN VĂN MẪU", "Ngón trỏ trái"]
    assert parse_back(rows)["issuing_authority"] == CANONICAL


def test_overlapping_ocr_lines_are_joined_once():
    rows = ["CỤC TRƯỞNG CỤC CẢNH SÁT", "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH", "VỀ TRẬT TỰ XÃ HỘI"]
    assert parse_back(rows)["issuing_authority"] == CANONICAL


def test_provincial_authority_split_after_province_label_keeps_the_actual_place():
    rows = ["Giám đốc Công an", "TỈNH", "HÀ NAM", "NGUYỄN VĂN MẪU"]
    assert parse_back(rows)["issuing_authority"] == "Công an TỈNH HÀ NAM"


def test_signatory_titles_are_removed_without_changing_the_institution():
    deputy = "PHÓ CỤC TRƯỞNG " + CANONICAL
    assert normalize_authority(deputy) == CANONICAL
    assert parse_back([deputy])["issuing_authority"] == CANONICAL
    acting = "KT. CỤC TRƯỞNG " + CANONICAL
    assert normalize_authority(acting) == CANONICAL
    assert parse_back([acting])["issuing_authority"] == CANONICAL


@pytest.mark.parametrize("value", ["SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", "CỤC QUẢN LÝ XUẤT NHẬP CẢNH"])
def test_unmatched_normalization_keeps_the_original_instead_of_a_rule_fragment(value):
    assert normalize_authority(value) == value


def test_enhancement_increases_low_contrast_text_range_without_changing_source_pixels():
    image = Image.new("RGB", (120, 30), (100, 100, 100))
    for x in range(30, 80):
        image.putpixel((x, 15), (120, 120, 120))
    before = image.tobytes()
    enhanced = enhance_text_patch(image)
    low, high = enhanced.convert("L").getextrema()
    assert high - low > 20
    assert image.tobytes() == before and image.size == (120, 30)
    assert enhanced.mode == "RGB" and enhanced.width <= 2200


def test_large_text_crops_are_bounded_before_recognition():
    image = Image.new("RGB", (4400, 100), "white")
    enhanced = enhance_text_patch(image)
    assert enhanced.size == (2200, 50)


def test_missing_line_crop_trims_empty_margins_without_removing_the_text_band():
    from PIL import ImageDraw
    image = Image.new("RGB", (240, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 38, 180, 62), fill="black")
    draw.point((0, 0), fill="black")
    before = image.tobytes()
    trimmed = trim_text_band(image)
    assert trimmed.width == image.width and 25 <= trimmed.height < 60
    assert sum(1 for pixel in trimmed.convert("L").getdata() if pixel == 0) == 121 * 25
    assert image.tobytes() == before


class Recognizer:
    def __init__(self, fail=False):
        self.padding = []
        self.fail = fail
        self.results = iter([
            ("CỤC TRƯỞNG CỤC CẢNH SÁT", .9), ("CỤC TRƯỞNG CỤC CẢNH SÁT", .85),
            ("Ý HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", .9),
            ("QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", .84),
        ])
        self.predictor = SimpleNamespace(config={"dataset": {"image_max_width": 512}}, predict=self.predict)

    def predict(self, _image, return_prob):
        if self.fail:
            raise RuntimeError("reader failed")
        return next(self.results)

    def crop_oriented_box(self, _image, _coordinates, **padding):
        self.padding.append(padding.get("pad_x", .02))
        return Image.new("RGB", (600, 50), "white")


def authority_items():
    return [SimpleNamespace(text="CỤC TRƯỞNG CỤC CẢNH SÁT", bbox=SimpleNamespace(coordinates=[(0, 0), (500, 0), (500, 50), (0, 50)])),
            SimpleNamespace(text="Ý HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", bbox=SimpleNamespace(coordinates=[(150, 70), (600, 70), (600, 120), (150, 120)]))]


def test_reread_expands_the_clipped_line_and_recovers_its_missing_words():
    recognizer, items = Recognizer(), authority_items()
    metadata = refine_authority(recognizer, object(), items)
    assert items[1].text.startswith("QUẢN LÝ")
    assert max(recognizer.padding) > .3
    assert recognizer.predictor.config["dataset"]["image_max_width"] == 512
    assert metadata["reread_lines"] == 2 and metadata["changed_lines"] == 1


def test_failed_reread_keeps_the_original_and_restores_reader_settings():
    recognizer, items = Recognizer(fail=True), authority_items()
    before = [item.text for item in items]
    refine_authority(recognizer, object(), items)
    assert [item.text for item in items] == before
    assert recognizer.predictor.config["dataset"]["image_max_width"] == 512


def test_confident_known_lines_skip_extra_recognition():
    recognizer, items = Recognizer(), authority_items()
    items[1].text = "QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"
    for item in items:
        item.confidence = .99
    recognizer.predictor.predict = lambda *_args, **_kwargs: pytest.fail("Already clear")
    metadata = refine_authority(recognizer, object(), items)
    assert metadata["reread_lines"] == metadata["reread_regions"] == 0
    assert not recognizer.padding


def test_confident_header_does_not_skip_rereading_when_institution_is_incomplete():
    recognizer, items = Recognizer(), authority_items()[:1]
    items[0].confidence = .99
    calls = []
    def read(patches, **kwargs):
        calls.append(len(patches))
        return ["CỤC TRƯỞNG " + CANONICAL] * len(patches), [.96] * len(patches)
    recognizer.predictor.predict_batch = read
    metadata = refine_authority(recognizer, object(), items)
    assert calls == [2] and metadata['changed_lines'] == 1
    assert parse_back([item.text for item in items])['issuing_authority'] == CANONICAL


def test_uncertain_authority_variants_use_one_batch_and_preserve_correct_title():
    recognizer, items = Recognizer(), authority_items()
    items[0].confidence = .99
    items[1].confidence = .8
    batches = []
    def batch(patches, return_prob):
        batches.append(len(patches))
        return ["QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"] * len(patches), [.95] * len(patches)
    recognizer.predictor.predict_batch = batch
    metadata = refine_authority(recognizer, object(), items)
    assert batches == [2] and metadata["reread_lines"] == 1
    assert items[1].text.startswith("QUẢN LÝ")


def test_missing_institution_head_widens_only_the_first_body_line():
    recognizer = Recognizer()
    items = authority_items()
    items[0].text = "CỤC TRƯỞNG"
    items[1].text = "SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"
    recognizer.predictor.predict_batch = lambda patches, **_kwargs: (
        [CANONICAL] * len(patches), [.96] * len(patches))
    metadata = refine_authority(recognizer, object(), items)
    assert max(recognizer.padding) >= .7
    assert metadata["reread_lines"] == 1
    assert parse_back([item.text for item in items])["issuing_authority"] == CANONICAL


def test_short_institution_head_can_be_extended_by_reading_real_wider_pixels():
    recognizer = Recognizer()
    items = authority_items()[:1]
    items[0].text = "CỤC CẢNH SÁT"
    recognizer.predictor.predict_batch = lambda patches, **kwargs: ([CANONICAL] * len(patches), [.96] * len(patches))
    metadata = refine_authority(recognizer, object(), items)
    assert max(recognizer.padding) >= .7
    assert items[0].text == CANONICAL
    assert metadata["changed_lines"] == 1


@pytest.mark.parametrize("read_text,accepted", [("HÀ NAM", True), ("NGUYỄN VĂN MẪU", False), ("NAM", False)])
def test_missing_province_band_requires_an_exact_place_reading(read_text, accepted):
    import numpy as np
    recognizer = Recognizer()
    items = [SimpleNamespace(text="GIÁM ĐỐC", confidence=.99,
                             bbox=SimpleNamespace(coordinates=[(80, 100), (300, 100), (300, 130), (80, 130)])),
             SimpleNamespace(text="CÔNG AN TỈNH", confidence=.95,
                             bbox=SimpleNamespace(coordinates=[(80, 150), (300, 150), (300, 180), (80, 180)]))]
    calls = []
    def batch(patches, return_prob):
        calls.append(len(patches))
        return ["CÔNG AN TỈNH", "CÔNG AN TỈNH", read_text, read_text], [.95, .95, .67, .67]
    recognizer.predictor.predict_batch = batch
    metadata = refine_authority(recognizer, np.zeros((500, 500, 3), dtype=np.uint8), items, province_names=["Hà Nam"])
    assert calls == [4]
    assert metadata["reread_lines"] == 2
    assert len(items) == (3 if accepted else 2)
    assert parse_back([{"text": item.text, "box": item.bbox.coordinates} for item in items])["issuing_authority"] == (
        "CÔNG AN TỈNH HÀ NAM" if accepted else "")
