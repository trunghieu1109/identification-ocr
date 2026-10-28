import json

import pytest

from identification_demo.authority import normalize_authority
from identification_demo.documents import parse_back


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
