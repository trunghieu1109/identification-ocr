from types import SimpleNamespace

import pytest

from identification_demo.authority_region import select_authority_region
from identification_demo.cpu_pipeline import reread_indexes
from identification_demo.documents import parse_back

AGENCY = "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"


def line(text, x, y, width=500, height=30, confidence=.98):
    return {"text": text, "box": [[x, y], [x + width, y], [x + width, y + height], [x, y + height]], "confidence": confidence}


def card_lines():
    return [line("BỘ CÔNG AN", 80, 40, 300),
            line("Ngày 20 tháng 4 năm 2021", 760, 230),
            line("CỤC TRƯỞNG", 900, 290, 220),
            line("Ngón trỏ trái", 80, 330, 250),
            line("CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH", 740, 340, 650),
            line("CÔNG AN TỈNH KHÁC", 80, 365, 450),
            line("VỀ TRẬT TỰ XÃ HỘI", 800, 390, 500),
            line("Thiếu tướng NGUYỄN VĂN MẪU", 800, 440),
            line("IDVNM000000000000<<<<<<<<<<<<<<", 80, 650, 1250)]


def test_unlabelled_issuer_uses_title_and_column_without_fingerprint_or_signer():
    rows = card_lines()
    region = select_authority_region(rows)
    assert region.text == AGENCY
    assert region.anchor == "title" and region.method == "layout"
    assert region.indices == (2, 4, 6)
    assert region.body_indices == (4, 6)
    assert region.bbox == (740, 340, 1390, 420)
    fields = parse_back(rows)
    assert fields["issuing_authority"] == AGENCY
    assert fields["issue_date"] == "20/04/2021"
    assert region.metadata((1600, 900))["line_count"] == 2
    assert reread_indexes(rows, {}, fields, "back", ["issuing_authority"]) == [4, 6]


def test_detected_text_order_does_not_determine_region_grouping():
    rows = list(reversed(card_lines()))
    region = select_authority_region(rows)
    assert region.text == AGENCY
    assert region.bbox == (740, 340, 1390, 420)


@pytest.mark.parametrize("title", ["CỤC TRƯỞNG", "PHÓ CỤC TRƯỞNG", "KT. CỤC TRƯỞNG", "Q. CỤC TRƯỞNG"])
def test_title_inline_with_institution_is_not_part_of_returned_issuer(title):
    rows = [line(title + " CỤC CẢNH SÁT", 700, 200, 650),
            line("QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", 700, 245, 650)]
    assert parse_back(rows)["issuing_authority"] == AGENCY


def test_missing_agency_below_title_does_not_pull_another_column_into_issuer():
    rows = [line("CỤC TRƯỞNG", 900, 200, 250), line("CÔNG AN TỈNH HÀ NAM", 50, 245),
            line("NGUYỄN VĂN MẪU", 800, 250)]
    assert parse_back(rows)["issuing_authority"] == ""


def test_title_only_or_header_only_does_not_become_an_issuer():
    for text in ("CỤC TRƯỞNG", "BỘ CÔNG AN"):
        assert parse_back([text])["issuing_authority"] == ""
        assert parse_back([line(text, 50, 40), line("NGUYỄN VĂN MẪU", 50, 120)])["issuing_authority"] == ""


def test_two_equally_readable_institutions_stay_ambiguous_and_do_not_reread_everything():
    rows = [line("GIÁM ĐỐC", 50, 200, 250), line("CÔNG AN TỈNH HÀ NAM", 50, 250),
            line("GIÁM ĐỐC", 750, 200, 250), line("CÔNG AN TỈNH NAM ĐỊNH", 750, 250)]
    region = select_authority_region(rows)
    assert region.ambiguous and not region.text
    fields = parse_back(rows)
    assert fields["issuing_authority"] == ""
    assert reread_indexes(rows, {}, fields, "back", ["issuing_authority"]) == []


def test_unknown_institution_is_kept_instead_of_replaced_with_a_common_cuc():
    rows = [line("GIÁM ĐỐC", 800, 200, 250), line("CÔNG AN TỈNH", 700, 250),
            line("HÀ NAM", 800, 300, 250), line("NGUYỄN VĂN MẪU", 700, 350)]
    assert parse_back(rows)["issuing_authority"] == "CÔNG AN TỈNH HÀ NAM"




def test_real_gpu_text_dto_boxes_are_accepted_by_the_same_selector():
    rows = [SimpleNamespace(text=item["text"], confidence=item["confidence"],
                            bbox=SimpleNamespace(coordinates=item["box"])) for item in card_lines()]
    assert select_authority_region(rows).text == AGENCY


def test_invalid_boxes_have_a_text_fallback_and_never_create_nan_metadata():
    rows = [{"text": "CỤC TRƯỞNG CỤC CẢNH SÁT", "box": [[float("nan"), 0]] * 4},
            {"text": "QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", "box": None}]
    region = select_authority_region(rows)
    assert region.text == AGENCY and region.bbox is None
    assert region.metadata((1000, 600)) is None


def test_clipped_paddle_title_is_included_in_the_region_for_vietnamese_rereading():
    rows = [line("CC TRUNG CC CÁNH SÁT", 50, 200),
            line("QUÅN LÝ HÀNH CHÍNH VÊ TRT Tų XÃ HÔI", 50, 250, 800),
            line("NGUYĚN VĂN MÃU", 50, 300)]
    region = select_authority_region(rows)
    assert region.body_indices == (0, 1)
    assert reread_indexes(rows, {}, {"issuing_authority": region.text}, "back", ["issuing_authority"]) == [0, 1]


def test_date_and_title_in_one_detector_box_keep_only_the_institution():
    rows = [line("Ngày cấp: 20/04/2021 CỤC TRƯỞNG CỤC CẢNH SÁT", 200, 200, 1000),
            line("QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", 250, 245, 900)]
    assert parse_back(rows)["issuing_authority"] == AGENCY


def test_signer_rank_with_agency_words_is_not_a_second_institution():
    rows = [line("CỤC TRƯỞNG", 900, 200, 250),
            line("Thiếu tướng CỤC CẢNH SÁT", 800, 300)]
    assert parse_back(rows)["issuing_authority"] == ""


def test_clipped_institution_can_be_reread_but_not_returned_as_a_complete_issuer():
    rows = [line("CỤC TRƯỞNG", 1000, 200, 250),
            line("SÁT QUẢN LÝ HÀNH CHÍNH", 950, 250, 400),
            line("VỀ TRẬT TỰ XÃ HỘI", 950, 300, 400)]
    region = select_authority_region(rows)
    assert region.body_indices == (1, 2)
    assert not region.text
    assert parse_back(rows)["issuing_authority"] == ""
    assert reread_indexes(rows, {}, {"issuing_authority": ""}, "back", ["issuing_authority"]) == [1, 2]
    assert region.metadata((1600, 900)) is None


@pytest.mark.parametrize("fragment", ["QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", "VỀ TRẬT TỰ XÃ HỘI"])
def test_description_without_institution_head_is_not_treated_as_an_agency(fragment):
    assert parse_back([fragment])["issuing_authority"] == ""
    assert parse_back([line(fragment, 50, 200), line("NGUYỄN VĂN MẪU", 50, 270)])["issuing_authority"] == ""


def test_provincial_institution_missing_the_province_is_not_complete():
    assert parse_back(["GIÁM ĐỐC", "CÔNG AN TỈNH"])["issuing_authority"] == ""
    assert parse_back([line("GIÁM ĐỐC", 900, 200), line("CÔNG AN TỈNH", 900, 250)])["issuing_authority"] == ""
