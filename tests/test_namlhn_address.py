import pytest

from identification_demo.namlhn_address import address_fields


ADDRESS = "Trưng Vương, Thành phố Việt Trì, Phú Thọ"


def line(text, top, left=40, right=800, height=25):
    return {"text": text, "box": [[left, top], [right, top], [right, top + height], [left, top + height]]}


@pytest.mark.parametrize("caption", [
    "Quê quán 1 Place of rrgin", "Quê quán / Place of origin",
    "Quê quán I Place of orrgin", "Quê quán / Place of orgin", "Place of rrgin",
])
def test_origin_caption_is_not_returned_as_the_value(caption):
    rows = [line(caption, 400), line(ADDRESS, 435), line("Nơi thường trú / Place of residence", 480),
            line("Phường Minh Khai, Hà Nội", 515)]
    assert address_fields(rows) == {"place_of_origin": ADDRESS,
                                    "place_of_residence": "Phường Minh Khai, Hà Nội"}


def test_vietnamese_and_damaged_english_caption_boxes_on_one_row_are_one_field():
    rows = [line("Quê quán", 400, right=200), line("1 Place of rrgin", 400, left=215, right=450),
            line(ADDRESS, 435)]
    assert address_fields(rows)["place_of_origin"] == ADDRESS


def test_caption_fragments_on_separate_rows_do_not_become_address_text():
    rows = [line("Quê quán", 400), line("Place of", 425), line("rrgin", 450), line(ADDRESS, 485)]
    assert address_fields(rows)["place_of_origin"] == ADDRESS


@pytest.mark.parametrize("caption", ["Quê quán 1 Place of rrgin", "Quê quán / Place of origin"])
def test_inline_value_is_kept_and_wrapped_address_lines_are_joined(caption):
    rows = [line(caption + ": Trưng Vương,", 400), line("Thành phố Việt Trì,", 435),
            line("Phú Thọ", 470), line("Nơi thường trú", 515), line("Hà Nội", 550)]
    assert address_fields(rows) == {"place_of_origin": ADDRESS, "place_of_residence": "Hà Nội"}


@pytest.mark.parametrize("boundary", [
    "Nơi thường trú / Place of residence: Hà Nội", "Có giá trị đến: 25/10/2036", "Ngày sinh: 01/02/1999",
    "Place of issue: Hà Nội",
])
def test_a_missing_origin_value_never_steals_the_next_field(boundary):
    rows = [line("Quê quán 1 Place of rrgin", 400), line(boundary, 435)]
    assert address_fields(rows)["place_of_origin"] == ""


def test_caption_without_a_value_does_not_leak_damaged_english():
    assert address_fields([line("Quê quán 1 Place of rrgin", 400)]) == {"place_of_origin": ""}


def test_address_numbers_and_printed_spelling_are_preserved():
    rows = [line("Nơi thường trú / Place of residence: 1 Trưng Vương,", 400),
            line("Thành phố Việt Trì, Phú Thọ", 435), line("Có giá trị đến", 480), line("25/10/2036", 515)]
    assert address_fields(rows)["place_of_residence"] == "1 " + ADDRESS


def test_unrelated_column_and_distant_lines_are_not_appended():
    rows = [line("Quê quán 1 Place of rrgin", 400, right=450),
            line("Nội dung cột khác", 435, left=1000, right=1200), line(ADDRESS, 435),
            line("Không thuộc địa chỉ", 800)]
    assert address_fields(rows)["place_of_origin"] == ADDRESS


def test_text_only_ocr_results_also_group_labels_and_values():
    assert address_fields(["Quê quán 1 Place of rrgin", ADDRESS])["place_of_origin"] == ADDRESS
