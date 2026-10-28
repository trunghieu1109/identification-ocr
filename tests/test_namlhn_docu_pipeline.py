from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from identification_demo.engine import OCRUnavailable
from identification_demo.expiry import parse_expiry
from identification_demo.namlhn_docu_pipeline import STRATEGY, _detections, _fields, extract_pixels
from identification_demo.namlhn_gpu_worker import NamLhnDocUCudaEngine, create_namlhn_gpu_app


@pytest.fixture(autouse=True)
def namlhn_source(monkeypatch):
    source = Path(__file__).resolve().parents[1] / ".vendor/vietnam-cccd-ocr"
    if not (source / "ocr").is_dir():
        pytest.skip("The namlhn parser checkout is required")
    monkeypatch.syspath_prepend(str(source))


def polygon(left=20, top=20, right=600, bottom=42):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def components(monkeypatch, texts, probabilities=None, failure=None):
    from ocr import preprocess
    boxes = [polygon(top=20 + i * 35, bottom=42 + i * 35) for i in range(len(texts))]
    # Make the detector image visibly different and twice the original size.
    monkeypatch.setattr(preprocess, "prepare_image", lambda frame, **_kwargs:
                        np.zeros((frame.shape[0] * 2, frame.shape[1] * 2, 3), dtype=np.uint8))
    det_boxes = [[[x * 2, y * 2] for x, y in box] for box in boxes]
    detector = SimpleNamespace(predict=lambda _image: [{"dt_polys": det_boxes,
        "dt_scores": [.99] * len(texts), "rec_texts": ["WRONG PADDLE TEXT"] * len(texts)}])
    crops, seen = [], []

    def crop(frame, coordinates, **padding):
        seen.append((frame.copy(), coordinates, padding))
        patch = Image.new("RGB", (100, 32), "white")
        crops.append(patch)
        return patch

    def batch(patches, return_prob):
        assert return_prob and len(patches) == len(texts)
        if failure:
            raise failure
        return texts, probabilities if probabilities is not None else [.99] * len(texts)

    predictor = SimpleNamespace(predict_batch=batch,
        predict=lambda patch, return_prob: (batch([patch], return_prob)[0][0],
                                            batch([patch], return_prob)[1][0]))
    recognizer = SimpleNamespace(crop_oriented_box=crop, predictor=predictor,
                                pad_top=.04, pad_bot=.01, pad_x=.02)
    return detector, recognizer, boxes, crops, seen


def test_all_lines_are_read_from_original_pixels_with_docu_padding_and_namlhn_parser(monkeypatch):
    texts = ["CĂN CƯỚC CÔNG DÂN", "Số: 001099000001", "Họ và tên: NGUYỄN VĂN THỦY",
             "Ngày sinh: 01/02/1999", "Giới tính: Nam", "Quốc tịch: Việt Nam",
             "Quê quán: Hà Nội", "Nơi thường trú: Hà Nội", "Có giá trị đến: 25/10/2036"]
    detector, recognizer, boxes, crops, seen = components(monkeypatch, texts)
    with Image.new("RGB", (800, 500), (13, 71, 119)) as image:
        result = extract_pixels(detector, recognizer, image, "front")
    assert result["document_side"] == "front"
    assert result["structured_fields"]["name"] == "NGUYỄN VĂN THỦY"
    assert result["structured_fields"]["place_of_residence"] == "Hà Nội"
    assert parse_expiry(result["text_lines"])[0]["expiry_date"] == "25/10/2036"
    assert [line["text"] for line in result["text_lines"]] == texts
    assert result["strategy"] == STRATEGY and result["coordinate_space"] == "original_image"
    assert result["work"]["recognized_regions"] == len(texts)
    assert result["work"]["recognition_batches"] == 1
    assert "paddle_text_lines" not in result
    for (frame, coordinates, padding), box, patch in zip(seen, boxes, crops):
        assert frame.shape == (500, 800, 3) and frame[0, 0].tolist() == [119, 71, 13]
        assert coordinates == box and padding == {"pad_top": .04, "pad_bot": .01, "pad_x": .02}
        with pytest.raises(ValueError):
            patch.getpixel((0, 0))


def test_recognition_probability_is_not_replaced_by_detector_confidence(monkeypatch):
    detector, recognizer, *_ = components(monkeypatch, ["Có giá trị đến: 25/10/2036"], [.55])
    with Image.new("RGB", (800, 500)) as image:
        result = extract_pixels(detector, recognizer, image, "front")
    line = result["text_lines"][0]
    assert line["score"] == line["confidence"] == .55
    assert line["detection_score"] == .99
    assert parse_expiry(result["text_lines"])[0]["expiry_date"] == "25/10/2036"


def test_empty_or_failed_reader_cannot_fall_back_to_wrong_paddle_text(monkeypatch):
    detector, recognizer, *_ = components(monkeypatch, [""])
    with Image.new("RGB", (800, 500)) as image:
        result = extract_pixels(detector, recognizer, image, "front")
    assert result["text_lines"] == [] and not any(result["structured_fields"].values())


@pytest.mark.parametrize("failure", [RuntimeError("reader failed"), None])
def test_all_crops_are_closed_on_failed_or_incomplete_recognition(monkeypatch, failure):
    detector, recognizer, _boxes, crops, _seen = components(monkeypatch, ["one", "two"], [], failure)
    expected = RuntimeError if failure else OCRUnavailable
    with Image.new("RGB", (800, 500)) as image:
        with pytest.raises(expected):
            extract_pixels(detector, recognizer, image, "front")
    assert len(crops) == 2
    for patch in crops:
        with pytest.raises(ValueError):
            patch.getpixel((0, 0))


def test_detector_keeps_unrecognized_boxes_and_rejects_invalid_geometry():
    results = [{"dt_polys": [polygon(), [[0, 0]] * 4, polygon(right=float("nan"))],
                "dt_scores": [.2, .9, .9], "rec_texts": ["", "wrong", "wrong"]}]
    lines = _detections(results, .5, .5)
    assert len(lines) == 1 and lines[0]["detection_score"] == .2
    assert lines[0]["box"] == [[x * .5, y * .5] for x, y in polygon()]


@pytest.mark.parametrize("texts,name", [
    (["Họ và tên / Full name: NGUYỄN VĂN THỦY", "Ngày sinh: 01/02/1999", "Giới tính: Nam"], "NGUYỄN VĂN THỦY"),
    (["Họ và tên / Full name", "NGUYỄN VĂN THỦY", "Ngày sinh: 01/02/1999"], "NGUYỄN VĂN THỦY"),
    ([".Họ và tên I Full name:", "Ngày sinh / Dato of birth:", "HOÀNG THÀNH VINH"], "HOÀNG THÀNH VINH"),
    (["Họ và tên:", "Full name:", "HOÀNG THÀNH VINH", "Ngày sinh:"], "HOÀNG THÀNH VINH"),
    (["Họ và tên:", "HOÀNG THÀNH", "VINH", "Ngày sinh:"], "HOÀNG THÀNH VINH"),
    (["Họ và tên", "Ngày sinh: 01/02/1999", "Giới tính: Nam"], ""),
])
def test_name_parser_preserves_inline_and_separate_values_and_never_uses_gender(texts, name):
    lines = [{"text": text, "box": polygon(top=20 + i * 35, bottom=42 + i * 35)} for i, text in enumerate(texts)]
    assert _fields(lines, "front")["name"] == name


def test_id_fragments_follow_their_horizontal_position_despite_vertical_jitter():
    lines = [
        {"text": "7", "box": polygon(475, 18, 495, 38)},
        {"text": "Số / No.:", "box": polygon(180, 21, 270, 41)},
        {"text": "00109900000", "box": polygon(280, 21, 470, 41)},
        {"text": "Ngày sinh: 01/02/1999", "box": polygon(180, 80, 500, 100)},
    ]
    assert _fields(lines, "front")["id_number"] == "001099000007"


def test_unrelated_digits_cannot_be_combined_into_an_id_number():
    lines = [
        {"text": "Ngày sinh: 01/02/1999", "box": polygon(top=20, bottom=40)},
        {"text": "Có giá trị đến: 2036", "box": polygon(top=80, bottom=100)},
    ]
    assert _fields(lines, "front")["id_number"] == ""


def test_whole_id_from_reader_survives_a_damaged_caption_and_an_extra_box(monkeypatch):
    texts = ["Số / No：001099000007", "7"]
    detector, recognizer, _boxes, _crops, _seen = components(monkeypatch, texts)
    boxes = [polygon(180, 20, 500, 40), polygon(505, 20, 520, 40)]
    detector.predict = lambda _image: [{"dt_polys": [[[x * 2, y * 2] for x, y in box] for box in boxes]}]
    with Image.new("RGB", (800, 500)) as image:
        result = extract_pixels(detector, recognizer, image, "front")
    assert result["structured_fields"]["id_number"] == "001099000007"
    assert [line["text"] for line in result["text_lines"]] == texts


def test_detector_to_parser_keeps_jittered_number_and_displaced_name_in_their_fields(monkeypatch):
    texts = ["Số / No.:", "00109900000", "7", ".Họ và tên I Full name:",
             "Ngày sinh / Dato of birth:", "HOÀNG THÀNH VINH"]
    detector, recognizer, _boxes, _crops, seen = components(monkeypatch, texts)
    boxes = [polygon(180, 12, 270, 32), polygon(280, 12, 470, 32), polygon(475, 9, 495, 29),
             polygon(180, 70, 600, 90), polygon(180, 105, 600, 125), polygon(180, 140, 600, 160)]
    # A last digit above the row boundary used to be cropped/read first.
    shuffled = [boxes[index] for index in (2, 1, 0, 4, 5, 3)]
    detector.predict = lambda _image: [{"dt_polys": [[[x * 2, y * 2] for x, y in box] for box in shuffled]}]
    with Image.new("RGB", (800, 500)) as image:
        result = extract_pixels(detector, recognizer, image, "front")
    assert [entry[1] for entry in seen] == boxes
    assert result["structured_fields"]["id_number"] == "001099000007"
    assert result["structured_fields"]["name"] == "HOÀNG THÀNH VINH"
    assert result["structured_fields"]["date_of_birth"] == ""
    assert [line["text"] for line in result["text_lines"]] == texts
    assert result["work"]["recognition_batches"] == 1


def test_front_parser_groups_damaged_bilingual_origin_caption_with_the_address_below():
    texts = ["Quê quán 1 Place of rrgin", "Trưng Vương, Thành phố Việt Trì, Phú Thọ",
             "Nơi thường trú / Place of residence", "Phường Minh Khai, Hà Nội"]
    lines = [{"text": text, "box": polygon(top=400 + i * 35, bottom=425 + i * 35)} for i, text in enumerate(texts)]
    fields = _fields(lines, "front")
    assert fields["place_of_origin"] == texts[1]
    assert fields["place_of_residence"] == texts[3]


def test_front_parser_maps_interleaved_columns_without_relying_on_list_order():
    from test_namlhn_mapping import EXPECTED, card
    assert _fields(list(reversed(card())), "front") == EXPECTED


def test_cpu_mapping_evidence_references_the_original_unsorted_input():
    from identification_demo.cpu_pipeline import _parse
    from test_namlhn_mapping import card
    lines = list(reversed(card()))
    raw, fields = _parse(lines, "front")
    assert fields["date_of_birth"] == "01/02/1999"
    selected = raw["_field_mapping"]["date_of_birth"]["value_indices"]
    assert [lines[index]["text"] for index in selected] == ["01/02/1999"]


def test_spatial_uncertainty_keeps_a_complete_observed_id_with_explicit_fallback_evidence():
    from identification_demo.cpu_pipeline import _parse
    lines = [{"text": "001099000007", "box": polygon(40, 300, 400, 320)},
             {"text": "Số / No.:", "box": polygon(40, 20, 200, 40)}]
    raw, fields = _parse(lines, "front")
    assert fields["id_number"] == "001099000007"
    assert raw["_field_mapping"]["id_number"]["method"] == "observed_number"
    assert raw["_field_mapping"]["id_number"]["value_indices"] == [0]


def test_hybrid_default_is_seq2seq_and_has_independent_readiness(monkeypatch):
    monkeypatch.delenv("NAMLHN_GPU_MODEL", raising=False)
    monkeypatch.setenv("OCR_MODEL", "vgg_transformer")
    engine = NamLhnDocUCudaEngine()
    try:
        assert engine.model == "vgg_seq2seq"
        assert not engine.describe()["pipeline_ready"]
        engine._detector, engine._recognizer, engine.cuda_available = object(), object(), True
        info = engine.describe()
        assert info["model_loaded"] and info["pipeline_ready"] and info["strategy"] == STRATEGY
        assert engine._extractor is None  # No second VietOCR/Paddle recognizer was loaded.
    finally:
        engine.close()


def test_worker_rejects_retired_recognizer_models_without_loading_them(monkeypatch):
    monkeypatch.setenv("NAMLHN_GPU_MODEL", "vgg_transformer")
    with pytest.raises(ValueError, match="vgg_seq2seq"):
        create_namlhn_gpu_app()
