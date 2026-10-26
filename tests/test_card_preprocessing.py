import numpy as np
from PIL import Image

from identification_demo.card_preprocessing import enhance_card, prepare_card


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


def test_preprocessing_keeps_full_image_and_export_artifacts_are_caller_owned():
    image = np.full((80, 128, 3), 255, dtype=np.uint8)
    artifacts = {}

    try:
        prepared, metadata = prepare_card(image, "off", artifacts)
        assert prepared is image and metadata["crop"]["status"] == "off"
        assert metadata["working_image_size"] == [128, 80]
        assert set(artifacts) == {"card", "prepared_card"}
        assert artifacts["card"].mode == "RGB" and artifacts["card"].size == (128, 80)
        np.testing.assert_array_equal(np.asarray(artifacts["card"]), image[:, :, ::-1])
    finally:
        for artifact in artifacts.values():
            artifact.close()


def test_pipeline_keeps_full_frame_line_boxes_and_denoise(monkeypatch):
    from pathlib import Path
    from identification_demo.namlhn_docu_pipeline import extract_pixels
    from test_namlhn_docu_pipeline import components

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / ".vendor/vietnam-cccd-ocr"))
    detector, recognizer, boxes, _patches, seen = components(monkeypatch, ["Số: 1001205025331"])

    pixels = np.clip(45 + np.random.default_rng(7).normal(0, 12, (500, 800)), 0, 255).astype(np.uint8)
    original = np.repeat(pixels[:, :, None], 3, axis=2)
    readings = []
    try:
        with Image.fromarray(original) as image:
            result = extract_pixels(detector, recognizer, image, "front", readings=readings, card_enhancement="adaptive")
            np.testing.assert_array_equal(np.asarray(image), original)
        assert result["text_lines"][0]["box"] == boxes[0]
        assert result["field_mapping"]["id_number"]["box"] == boxes[0]
        assert readings[0]["regions"][0]["box"] == boxes[0]
        assert seen[0][0].shape == (500, 800, 3)
        assert seen[0][2] == {"pad_top": .04, "pad_bot": .01, "pad_x": .02}
        assert result["image_preprocessing"]["crop"]["status"] == "off"
        assert result["image_preprocessing"]["working_image_size"] == [800, 500]
        assert result["image_preprocessing"]["enhancement"]["denoise"]
        assert result["structured_fields"]["id_number"] == "1001205025331"
        assert not result["id_number_validation"]["format_valid"]
    finally:
        for reading in readings:
            for patch in reading["images"]:
                patch.close()
