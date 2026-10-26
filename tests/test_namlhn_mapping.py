from copy import deepcopy
import random

import pytest

from identification_demo.namlhn_mapping import map_front_fields


def line(text, left, top, right, height=20):
    return {"text": text, "box": [[left, top], [right, top], [right, top + height], [left, top + height]],
            "score": .99}


def card():
    return [line("Số / No.:", 180, 30, 270), line("00109900000", 280, 30, 470), line("7", 475, 27, 495),
            line("Họ và tên / Full name:", 180, 90, 480), line("HOÀNG THÀNH", 180, 125, 390),
            line("VINH", 400, 122, 480), line("Ngày sinh / Dato of birth:", 180, 175, 460),
            line("Giới tính / Sex:", 550, 175, 740), line("Quốc tịch / Nationality:", 820, 175, 1130),
            line("01/02/1999", 180, 210, 390), line("Nam", 550, 210, 610),
            line("Việt Nam", 820, 210, 970), line("Quê quán / Place of origin", 180, 270, 500),
            line("Hà Nội", 180, 305, 310), line("Nơi thường trú / Place of residence", 180, 365, 650),
            line("1 Trưng Vương,", 180, 400, 450), line("Thành phố Việt Trì, Phú Thọ", 180, 435, 640),
            line("Có giá trị đến / Date of expiry:", 180, 490, 600), line("25/10/2036", 180, 525, 390)]


EXPECTED = {"id_number": "001099000007", "name": "HOÀNG THÀNH VINH", "date_of_birth": "01/02/1999",
            "gender": "Nam", "nationality": "Việt Nam", "place_of_origin": "Hà Nội",
            "place_of_residence": "1 Trưng Vương, Thành phố Việt Trì, Phú Thọ"}


@pytest.mark.parametrize("seed", [0, 7, 31, 96])
@pytest.mark.parametrize("scale,offset", [(1, 0), (.5, 200), (2, 37)])
@pytest.mark.parametrize("slope", [-.12, 0, .12])
def test_mapping_is_independent_of_ocr_array_order_scale_and_origin(seed, scale, offset, slope):
    lines = card()
    for item in lines:
        item["box"] = [[x * scale + offset, (y + slope * x) * scale + offset] for x, y in item["box"]]
    random.Random(seed).shuffle(lines)
    original = deepcopy(lines)
    fields = map_front_fields(lines)
    assert {key: item["value"] for key, item in fields.items()} == EXPECTED
    assert lines == original
    used = {}
    for key, item in fields.items():
        for index in item["value_indices"]:
            assert index not in used, (key, used.get(index))
            used[index] = key


def test_expiry_in_an_interleaved_column_cannot_become_date_of_birth():
    lines = [line("Ngày sinh:", 40, 100, 220), line("Có giá trị đến:", 600, 100, 850),
             line("25/10/2036", 600, 130, 800), line("01/02/1999", 40, 130, 240)]
    fields = map_front_fields(lines)
    assert fields["date_of_birth"]["value"] == "01/02/1999"
    assert fields["date_of_birth"]["value_indices"] == [3]


def test_displaced_birth_caption_does_not_hide_the_name():
    lines = [line("Ngày sinh / Dato of birth:", 180, 55, 550), line("HOÀNG THÀNH VINH", 180, 90, 550),
             line(".Họ và tên I Full name:", 180, 20, 550), line("01/02/1999", 180, 125, 380)]
    fields = map_front_fields(lines)
    assert fields["name"]["value"] == "HOÀNG THÀNH VINH"
    assert fields["date_of_birth"]["value"] == "01/02/1999"


def test_missing_origin_must_not_claim_the_residence_in_an_adjacent_column():
    lines = [line("Quê quán:", 40, 100, 300), line("Nơi thường trú:", 600, 100, 900),
             line("1 Trưng Vương, Hà Nội", 600, 140, 1000)]
    fields = map_front_fields(lines)
    assert fields["place_of_origin"]["value"] == ""
    assert fields["place_of_residence"]["value"] == "1 Trưng Vương, Hà Nội"


def test_numeric_fragments_beside_the_id_label_win_over_a_far_complete_number():
    lines = [line("001099000008", 850, 40, 1100), line("7", 475, 37, 495),
             line("Số:", 180, 40, 270), line("00109900000", 280, 40, 470)]
    assert map_front_fields(lines)["id_number"]["value"] == "001099000007"


def test_bilingual_captions_can_be_fragmented_across_boxes_and_rows():
    lines = [line("Quê quán", 40, 100, 220), line("Place of", 40, 130, 210),
             line("rrgin", 40, 160, 160), line("Hà Nội", 40, 195, 200),
             line("Họ và tên", 40, 10, 220), line("Full name:", 240, 13, 430),
             line("HOÀNG THÀNH VINH", 40, 45, 430)]
    fields = map_front_fields(list(reversed(lines)))
    assert fields["place_of_origin"]["value"] == "Hà Nội"
    assert fields["name"]["value"] == "HOÀNG THÀNH VINH"


def test_values_on_the_right_and_wrapped_inline_names_are_kept():
    lines = [line("Họ và tên: HOÀNG THÀNH", 40, 10, 430), line("VINH", 40, 45, 140),
             line("Ngày sinh:", 40, 90, 240), line("01/", 250, 90, 290),
             line("02/", 295, 92, 335), line("1999", 340, 90, 420),
             line("Giới tính:", 40, 130, 240), line("Nam", 250, 130, 320)]
    fields = map_front_fields(lines)
    assert fields["name"]["value"] == "HOÀNG THÀNH VINH"
    assert fields["date_of_birth"]["value"] == "01/02/1999"
    assert fields["gender"]["value"] == "Nam"


def test_ambiguous_dates_are_not_resolved_using_input_order():
    lines = [line("Ngày sinh:", 100, 100, 400), line("01/02/1999", 100, 135, 250),
             line("03/04/2000", 260, 135, 410)]
    assert map_front_fields(lines)["date_of_birth"]["value"] == ""
    assert map_front_fields(list(reversed(lines)))["date_of_birth"]["value"] == ""


def test_missing_geometry_is_left_for_the_existing_text_parser():
    assert map_front_fields([{"text": "Họ và tên:"}, {"text": "HOÀNG THÀNH VINH"}]) == {}


def test_two_fields_recognized_in_one_box_are_split_by_their_captions():
    lines = [line("Giới tính / Sex: Nam Quốc tịch / Nationality: Việt Nam", 40, 100, 900),
             line("Họ và tên: HOÀNG THÀNH VINH Ngày sinh: 01/02/1999", 40, 40, 900)]
    fields = map_front_fields(lines)
    assert fields["name"]["value"] == "HOÀNG THÀNH VINH"
    assert fields["date_of_birth"]["value"] == "01/02/1999"
    assert fields["gender"]["value"] == "Nam"
    assert fields["nationality"]["value"] == "Việt Nam"


def test_nationality_does_not_absorb_an_unlabelled_line_below_its_value():
    lines = [line("Quốc tịch:", 40, 20, 300), line("Việt", 40, 55, 100),
             line("Nam", 105, 55, 165), line("Hà Nội", 40, 90, 200)]
    fields = map_front_fields(lines)
    assert fields["nationality"]["value"] == "Việt Nam"
    assert fields["nationality"]["value_indices"] == [1, 2]
