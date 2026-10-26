import unicodedata

import pytest

from identification_demo.namlhn_identity import extract_id_number, extract_name
from identification_demo.namlhn_layout import reading_order


def line(text, left=180, top=20, right=500, height=20, slope=0):
    return {"text": text, "box": [[left, top + slope * left], [right, top + slope * right],
                                  [right, top + height + slope * right], [left, top + height + slope * left]]}


@pytest.mark.parametrize("offset", [0, 3, 15, 250])
@pytest.mark.parametrize("slope", [-.12, 0, .12])
def test_number_fragments_keep_left_to_right_order_under_translation_and_tilt(offset, slope):
    fragments = [line("7", 475, offset + 18, 495, slope=slope),
                 line("00109900000", 280, offset + 21, 470, slope=slope),
                 line("Số / No.:", 180, offset + 21, 270, slope=slope)]
    other = line("Ngày sinh: 01/02/1999", top=offset + 80, slope=slope)
    original = [*fragments, other]
    ordered = reading_order(original)
    assert [item["text"] for item in ordered] == ["Số / No.:", "00109900000", "7", other["text"]]
    assert extract_id_number(original)[0] == "001099000007"
    assert original[0] is fragments[0]


@pytest.mark.parametrize("number", ["001099000007", "012345678"])
def test_complete_number_and_leading_zero_are_preserved(number):
    assert extract_id_number([line(f"Số / No.: {number}")])[0] == number
    assert extract_id_number([line(number)])[0] == number
    assert extract_id_number([line(unicodedata.normalize("NFD", f"Số: {number}"))])[0] == number


def test_number_caption_zero_is_not_mistaken_for_an_extra_id_digit():
    assert extract_id_number([line("Số / N0.001099000007")])[0] == "001099000007"


@pytest.mark.parametrize("text", [
    "Sổ / Nq: 001099000007",
    "Số / No001099000007",
    "Số / No：001099000007",
    "Số / Nº: 001099000007",
    "Số / N0001099000007",
    "S6 / Nq: 001099000007",
    "Số căn cước công dân: 001 099 000 007",
    "Số: 001.099.000.007",
    "001-099-000-007",
    "Số / No.: 001099000007 |",
    "Số / No.: 001099000007 (CCCD)",
    "Số / No.: 001099000007 7",
])
def test_complete_observed_number_survives_damaged_caption_and_punctuation(text):
    assert extract_id_number([line(text)])[0] == "001099000007"


def test_complete_number_survives_an_extra_detected_digit_on_the_same_row():
    lines = [line("Số: 001099000007", right=500), line("7", left=505, right=520)]
    assert extract_id_number(lines)[0] == "001099000007"


def test_a_nine_digit_fragment_is_extended_to_twelve_before_selecting_the_number():
    lines = [line("001099000", right=400), line("007", left=405, right=450)]
    assert extract_id_number(lines)[0] == "001099000007"


@pytest.mark.parametrize("text", [
    "Ngày sinh / Date of birth: 001099000007",
    "Có giá trị đến: 001099000007",
    "Nơi thường trú: 001099000007 Đường Kiểm Thử",
    "Quê quán: 001099000007",
    "01/02/1999 2036",
    "01.02.1999 2036",
])
def test_a_number_from_another_labelled_field_cannot_fill_the_id(text):
    assert extract_id_number([line(text)]) == ("", [])


@pytest.mark.parametrize("number", ["0010990000", "00109900000", "0010990000078"])
def test_incomplete_or_oversized_numbers_are_not_truncated(number):
    value, box, validation = extract_id_number([line(number)], with_details=True)
    assert value == number and box
    assert not validation["format_valid"] and validation["status"] == "unverified"


def test_fragments_on_different_rows_or_far_apart_are_not_joined():
    assert extract_id_number([line("00109900000"), line("7", top=65)])[0] == "00109900000"
    assert extract_id_number([line("00109900000"), line("7", left=900, right=920)])[0] == "00109900000"


def test_labelled_number_wins_and_conflicts_retain_an_observed_value_with_metadata():
    first = line("001099000007")
    labelled = line("Số: 001099000008", top=65)
    assert extract_id_number([first, labelled])[0] == "001099000008"
    value, _box_value, validation = extract_id_number([first, line("001099000008", top=65)], with_details=True)
    assert value == "001099000007" and validation["ambiguous"] and validation["status"] == "ambiguous"


@pytest.mark.parametrize("text,expected", [
    ("Số 1001205025331", "1001205025331"),
    ("Số / No.: 0010990000", "0010990000"),
    ("Số căn cước công dân: 001 099 000 0078", "0010990000078"),
    ("Số / No.: O01205025331", "O01205025331"),
])
def test_imperfect_labelled_id_is_returned_literally_without_fixing_or_dropping_characters(text, expected):
    value, _box_value, validation = extract_id_number([line(text)], with_details=True)
    assert value == expected and validation["observed_length"] == len(expected)
    assert not validation["format_valid"]


def test_labelled_imperfect_number_wins_over_an_unrelated_twelve_digit_number():
    lines = [line("001099000007", top=20), line("Số: 1001205025331", top=80)]
    assert extract_id_number(lines)[0] == "1001205025331"


def test_labelled_nine_digit_fragment_is_not_kept_when_the_observed_extension_is_imperfect():
    lines = [line("Số: 001099000", right=400), line("0078", left=405, right=460)]
    assert extract_id_number(lines)[0] == "0010990000078"


def test_dates_are_not_used_as_id_fragments():
    lines = [line("Ngày sinh: 01/02/1999"), line("Có giá trị đến: 2036", top=65)]
    assert extract_id_number(lines) == ("", [])


def test_name_fragments_are_joined_despite_detector_order_and_different_heights():
    lines = [line("VINH", left=400, top=57, right=480, height=25),
             line("Ngày sinh / Dato of birth:", top=100),
             line("Họ và tên I Full name:", top=20),
             line("HOÀNG THÀNH", left=180, top=60, right=390)]
    name, box = extract_name(lines)
    assert name == "HOÀNG THÀNH VINH"
    assert box == [[180, 57], [480, 57], [480, 82], [180, 82]]


@pytest.mark.parametrize("texts", [
    [".Họ và tên I Full name:", "Ngày sinh / Dato of birth:", "HOÀNG THÀNH VINH"],
    ["Họ và tên /", "Full name:", "HOÀNG THÀNH VINH"],
    ["Họ và tên:", "HOÀNG THÀNH", "VINH", "Ngày sinh: 01/02/1999"],
    ["Họ và tên:", "Ngày sinh: 01/02/1999", "HOÀNG THÀNH VINH"],
    ["Họ và tên: HOÀNG THÀNH", "VINH", "Ngày sinh: 01/02/1999"],
])
def test_name_value_can_follow_translated_or_displaced_captions(texts):
    lines = [line(text, top=20 + 35 * index) for index, text in enumerate(texts)]
    assert extract_name(lines)[0] == "HOÀNG THÀNH VINH"


def test_name_with_decomposed_diacritics_is_normalized_before_caption_offsets():
    text = unicodedata.normalize("NFD", "Họ và tên / Full name: HOÀNG THÀNH VINH")
    assert extract_name([line(text)])[0] == "HOÀNG THÀNH VINH"


def test_uppercase_name_fragment_nam_is_preserved_before_gender_caption():
    lines = [line("Họ và tên:"), line("NGUYỄN VĂN", top=55), line("NAM", top=90),
             line("Giới tính: Nam", top=125)]
    assert extract_name(lines)[0] == "NGUYỄN VĂN NAM"


@pytest.mark.parametrize("text", ["Giới tính: Nam", "Quốc tịch: Việt Nam", "Quê quán: Hà Nội",
                                  "Có giá trị đến: 25/10/2036", "Việt Nam", "Nam"])
def test_other_field_values_are_not_names(text):
    assert extract_name([line("Họ và tên:"), line(text, top=55)]) == ("", [])


def test_distant_or_other_column_name_cannot_fill_a_caption():
    anchor = line("Họ và tên:")
    assert extract_name([anchor, line("HOÀNG THÀNH VINH", top=300)]) == ("", [])
    assert extract_name([anchor, line("HOÀNG THÀNH VINH", left=850, top=55, right=1150)]) == ("", [])


def test_missing_geometry_retains_text_order_and_caption_guards():
    lines = [{"text": text} for text in ["Họ và tên:", "Ngày sinh / Dato of birth:",
                                         "HOÀNG THÀNH VINH", "Giới tính: Nam"]]
    assert reading_order(lines) == lines
    assert extract_name(lines)[0] == "HOÀNG THÀNH VINH"
