import numpy as np
import pytest
from PIL import Image

from identification_demo.card_preprocessing import crop_card, enhance_card, prepare_card, project_polygon
from identification_demo.engine import OCRMultipleDocuments


def polygon(left, top, width, height):
    return [[left, top], [left + width, top], [left + width, top + height], [left, top + height]]


def test_crop_excludes_background_adds_margin_and_projects_back_to_source():
    image = np.full((600, 1000, 3), 25, dtype=np.uint8)
    image[150:400, 250:650] = (40, 100, 200)
    crop, inverse, metadata = crop_card(image, [(polygon(250, 150, 400, 250), .95)])
    assert metadata["status"] == "cropped" and crop.shape[1] < image.shape[1]
    np.testing.assert_allclose(crop[crop.shape[0] // 2, crop.shape[1] // 2], [40, 100, 200])
    corners = project_polygon(polygon(0, 0, crop.shape[1] - 1, crop.shape[0] - 1), inverse, (1000, 600))
    np.testing.assert_allclose(corners, metadata["polygon"], atol=1e-3)
    assert corners[0][0] < 250 and corners[0][1] < 150  # Keep edge text outside a tight mask.


def test_already_framed_card_and_no_detection_preserve_original_pixels():
    image = np.full((600, 1000, 3), 170, dtype=np.uint8)
    for predictions, status in [([], "not_found"), ([(polygon(10, 10, 980, 580), .9)], "already_framed")]:
        crop, inverse, metadata = crop_card(image, predictions)
        assert crop is image and metadata["status"] == status
        np.testing.assert_array_equal(inverse, np.eye(3))


def test_nested_duplicate_masks_keep_the_larger_card():
    image = np.zeros((800, 1200, 3), dtype=np.uint8)
    small, large = polygon(320, 220, 350, 220), polygon(300, 200, 400, 250)
    _crop, _inverse, metadata = crop_card(image, [(small, .98), (large, .9)])
    assert metadata["detected_regions"] == 1
    assert metadata["polygon"][0][0] < 300


def test_two_comparable_cards_raise_instead_of_mixing_their_fields():
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    with pytest.raises(OCRMultipleDocuments):
        crop_card(image, [(polygon(100, 100, 600, 375), .95), (polygon(850, 300, 600, 375), .95)])


@pytest.mark.parametrize("points,score", [(polygon(20, 20, 400, 250), .2),
                                          ([[float("nan"), 0], [100, 0], [100, 100]], .9),
                                          (polygon(10, 10, 20, 20), .9)])
def test_weak_invalid_or_tiny_masks_fall_back_to_the_whole_image(points, score):
    image = np.zeros((600, 1000, 3), dtype=np.uint8)
    crop, _inverse, metadata = crop_card(image, [(points, score)])
    assert crop is image and metadata["status"] == "not_found"


def test_dark_low_contrast_noisy_card_gets_adaptive_adjustments_without_modifying_source():
    rng = np.random.default_rng(7)
    pixels = np.clip(45 + rng.normal(0, 12, (160, 260)), 0, 255).astype(np.uint8)
    image = np.repeat(pixels[:, :, None], 3, axis=2)
    original = image.copy()
    adjusted, metadata = enhance_card(image)
    assert metadata["brightness"] and metadata["contrast"] and metadata["denoise"]
    assert np.median(adjusted) > np.median(image)
    np.testing.assert_array_equal(image, original)


def test_clean_or_disabled_enhancement_does_not_change_pixels():
    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    image[30:50, 40:120] = 0
    for mode in ("off", "adaptive"):
        adjusted, metadata = enhance_card(image, mode)
        assert adjusted is image and not any(metadata[k] for k in ("brightness", "contrast", "denoise"))


def test_localization_failure_keeps_image_and_export_artifacts_are_caller_owned():
    image = np.full((80, 128, 3), 255, dtype=np.uint8)
    artifacts = {}

    def failed(_frame):
        raise RuntimeError("localization failed")

    try:
        prepared, _inverse, metadata = prepare_card(image, failed, "off", artifacts)
        assert prepared is image and metadata["crop"]["status"] == "error"
        assert set(artifacts) == {"card", "prepared_card"}
        assert artifacts["card"].mode == "RGB" and artifacts["card"].size == (128, 80)
    finally:
        for artifact in artifacts.values():
            artifact.close()


def test_pipeline_maps_text_and_mapping_to_original_after_crop(monkeypatch):
    from pathlib import Path
    from identification_demo.namlhn_docu_pipeline import extract_pixels
    from test_namlhn_docu_pipeline import components

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / ".vendor/vietnam-cccd-ocr"))
    detector, recognizer, boxes, _patches, seen = components(monkeypatch, ["Số: 1001205025331"])

    def localize(frame):
        return frame[80:400, 120:760], np.array([[1., 0, 120], [0, 1, 80], [0, 0, 1]]), {"status": "cropped"}

    readings = []
    try:
        with Image.new("RGB", (800, 500), (13, 71, 119)) as image:
            result = extract_pixels(detector, recognizer, image, "front", readings=readings, card_preprocessor=localize)
        expected = [[x + 120, y + 80] for x, y in boxes[0]]
        assert result["text_lines"][0]["box"] == expected
        assert result["text_lines"][0]["working_box"] == boxes[0]
        assert result["field_mapping"]["id_number"]["box"] == expected
        assert readings[0]["regions"][0]["box"] == expected
        assert seen[0][0].shape == (320, 640, 3)
        assert result["structured_fields"]["id_number"] == "1001205025331"
        assert not result["id_number_validation"]["format_valid"]
    finally:
        for reading in readings:
            for patch in reading["images"]:
                patch.close()
