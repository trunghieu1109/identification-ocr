"""Expiry values follow their printed caption, not detector list ordering."""
import pytest

from identification_demo.cpu_pipeline import reread_indexes
from identification_demo.expiry import expiry_rows, parse_expiry


def line(text, left, top, right, bottom):
    return {"text": text, "box": [[left, top], [right, top], [right, bottom], [left, bottom]], "score": .99}


@pytest.mark.parametrize("date_box", [(300, 500, 475, 525), (100, 545, 275, 570)])
def test_caption_value_association_survives_scrambled_lines_and_other_columns(date_box):
    rows = [line("25/10/2036", *date_box), line("Ngày sinh: 01/02/1999", 650, 480, 1000, 505),
            line("Hà Nội", 650, 520, 900, 545), line("Ngày cấp: 15/03/2021", 650, 560, 1000, 585),
            line("Có giá trị đến:", 100, 500, 280, 525)]
    matches = parse_expiry(rows)
    assert {match["expiry_date"] for match in matches} == {"25/10/2036"}
    assert all(match["method"] == "layout" for match in matches)
    assert {item["number"] for item in matches[0]["evidence"]} == {1, 5}


def test_nearby_value_is_found_even_after_many_interleaved_detector_rows():
    rows = [line("Có giá trị đến", 50, 500, 230, 525)]
    rows += [line(f"Nội dung {i}", 600, 430 + i * 30, 900, 450 + i * 30) for i in range(8)]
    rows += [line("25/10/2036", 50, 545, 225, 570)]
    assert parse_expiry(rows)[0]["expiry_date"] == "25/10/2036"


@pytest.mark.parametrize("rows", [
    [line("Có giá trị đến", 50, 500, 230, 525), line("25/10/2036", 650, 550, 850, 575)],
    [line("Có giá trị đến", 50, 500, 230, 525), line("25/10/2036", 50, 900, 225, 925)],
    [line("Có giá trị đến", 50, 500, 230, 525), line("01/02/1999", 300, 500, 475, 525),
     line("Ngày sinh", 240, 500, 290, 525)],
    [line("Có giá trị đến", 50, 500, 230, 525), line("01/02/1999", 50, 570, 225, 595),
     line("Ngày sinh", 50, 540, 230, 565)],
])
def test_an_unrelated_or_birth_date_is_not_borrowed_from_text_order(rows):
    assert parse_expiry(rows) == []


def test_fragmented_caption_is_joined_by_location_even_in_reverse_detector_order():
    rows = [line("25/10/2036", 320, 500, 495, 525), line("trị đến:", 175, 500, 280, 525),
            line("Có giá", 70, 500, 160, 525)]
    assert {match["expiry_date"] for match in parse_expiry(rows)} == {"25/10/2036"}


def test_a_date_split_into_two_boxes_is_joined_after_the_caption():
    rows = [line("Có giá trị đến:", 50, 500, 230, 525), line("2036", 335, 545, 400, 570),
            line("25/10/", 230, 545, 325, 570)]
    assert {match["expiry_date"] for match in parse_expiry(rows)} == {"25/10/2036"}


@pytest.mark.parametrize("label", ["Có giá trị đến", "CÓ GIÁ TRỊ ĐẾN", "Có giá trị", "Giá trị đến", "Cogiatriđến"])
def test_common_vietnamese_caption_readings_are_anchors(label):
    assert parse_expiry([label, "25/10/2036"])[0]["expiry_date"] == "25/10/2036"


def test_cpu_expiry_pass_rereads_caption_and_nearby_date_instead_of_main_fields():
    rows = [line("Ngày sinh: 01/02/1999", 500, 400, 900, 425),
            line("Có giá trị đến", 50, 500, 230, 525), line("25/10/20B6", 50, 540, 225, 565),
            line("Ngày cấp: 15/03/2021", 600, 540, 1000, 565)]
    assert reread_indexes(rows, {}, {}, "front", ["expiry_date"]) == [1, 2]
    assert reread_indexes(rows, {}, {}, "back", ["expiry_date"]) == [1, 2]


def test_cpu_corrects_the_expiry_crop_even_with_confident_initial_ocr():
    from types import SimpleNamespace
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from PIL import Image
    from identification_demo.cpu_pipeline import extract_pixels
    rows = [line("Ngày sinh: 01/02/1999", 500, 400, 900, 425),
            line("Có giá trị đến", 50, 500, 230, 525), line("25/10/20B6", 50, 540, 225, 565)]
    batches = []

    def recognize(patches, return_prob):
        batches.append(len(patches))
        return ["Có giá trị đến", "25/10/2036"], [.99, .99]

    extractor = SimpleNamespace(enhance=False, resize=False, detect=lambda _image: rows,
                                _warp_region=lambda *_args, **_kwargs: np.full((25, 200, 3), 255, dtype=np.uint8),
                                _postprocess_text=lambda text: text,
                                _vietocr=SimpleNamespace(predict_batch=recognize))
    with Image.new("RGB", (1200, 760)) as image:
        result = extract_pixels(extractor, image, "front", ["expiry_date"])
    assert batches == [2]
    assert parse_expiry(result["text_lines"])[0]["expiry_date"] == "25/10/2036"
    assert result["text_lines"][0]["text"] == "Ngày sinh: 01/02/1999"


@pytest.mark.parametrize("box", [None, [], [0, 0, 0, 0], [0, 0, float("nan"), 10], [[1, 2], [3, 4]]])
def test_invalid_box_metadata_does_not_drop_recognized_text(box):
    rows = [{"text": "Có giá trị đến: 25/10/2036", "box": box}]
    assert expiry_rows(rows)[0]["bbox"] is None
    assert parse_expiry(rows)[0]["expiry_date"] == "25/10/2036"
