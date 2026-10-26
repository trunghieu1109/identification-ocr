from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from identification_demo.text_enhancement import enhance_text_crop, needs_enhancement, prefer_enhanced, read_text_crops


def test_denoising_reduces_flat_background_noise_without_modifying_the_input():
    rng = np.random.default_rng(42)
    pixels = np.clip(220 + rng.normal(0, 4, (64, 160, 3)), 0, 255).astype(np.uint8)
    pixels[15:49, 60:80] = 30
    with Image.fromarray(pixels) as original, enhance_text_crop(Image.fromarray(pixels)) as enhanced:
        assert enhanced.size == original.size and enhanced.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(original), pixels)
        assert np.asarray(enhanced)[:, :40].std() < pixels[:, :40].std()
        assert np.asarray(enhanced)[25:40, 65:75].mean() < 60


@pytest.mark.parametrize("size", [(1, 1), (40, 2), (200, 32)])
def test_uniform_crops_are_not_given_artificial_edges(size):
    with Image.new("RGB", size, "white") as image, enhance_text_crop(image) as enhanced:
        np.testing.assert_array_equal(np.asarray(enhanced), np.asarray(image))
        assert needs_enhancement(image) is False


def test_quality_gate_skips_clean_sharp_text_and_identifies_low_contrast():
    import cv2
    pixels = np.full((48, 400, 3), 255, dtype=np.uint8)
    cv2.putText(pixels, "VIET NAM 2026", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    with Image.fromarray(pixels) as clean:
        assert needs_enhancement(clean) is False
    with Image.fromarray((pixels.astype(np.float32) * .3 + 150).astype(np.uint8)) as faded:
        assert needs_enhancement(faded) is True


@pytest.mark.parametrize("raw,old,new,score,accepted", [
    ("Hà Ni", .7, "Hà Nội", .91, True),
    ("Hà Nội", .9, "Hà Ni", .8, False),
    ("Hà Nội", .85, "Hà Ni", .86, False),
    ("Hà Nội", .7, "Hà Nội", .99, False),
    ("", .1, "Có giá trị đến", .8, True),
    ("001099000001", .6, "001099000007", .99, False),
    ("25/10/2036", .6, "25/10/2038", .99, False),
    ("Họ và tên: NGUYỄN VĂN THỦY", .7, "THỦY", .99, False),
    ("Hà Nội", .7, "", .99, False),
    ("Hà Ni", .7, "Hà Nội", float("nan"), False),
])
def test_selection_needs_a_score_margin_and_preserves_observed_numbers(raw, old, new, score, accepted):
    assert prefer_enhanced(raw, old, new, score) is accepted


def test_adaptive_only_retries_uncertain_crops_and_preserves_order(monkeypatch):
    import identification_demo.text_enhancement as module
    enhanced_crops = []

    def enhance(image):
        patch = image.copy()
        enhanced_crops.append(patch)
        return patch

    monkeypatch.setattr(module, "enhance_text_crop", enhance)
    monkeypatch.setattr(module, "needs_enhancement", lambda image: True)
    predictor = SimpleNamespace(predict_batch=lambda images, return_prob: (["Hà Ni", "Việt Nam"], [.7, .99]),
        predict=lambda image, return_prob: ("Hà Nội", .95))
    with Image.new("RGB", (200, 32)) as first, Image.new("RGB", (200, 32)) as second:
        texts, scores, info, timings = read_text_crops(predictor, [first, second], "adaptive")
        assert texts == ["Hà Nội", "Việt Nam"] and scores == [.95, .99]
        assert info["attempted_regions"] == info["changed_regions"] == 1
        assert info["status"] == "applied"
        first.getpixel((0, 0))  # The caller still owns the original crops.
    assert len(enhanced_crops) == 1 and all(value >= 0 for value in timings.values())
    with pytest.raises(ValueError):
        enhanced_crops[0].getpixel((0, 0))


def test_confident_crops_and_disabled_mode_do_not_run_an_extra_inference(monkeypatch):
    import identification_demo.text_enhancement as module
    monkeypatch.setattr(module, "enhance_text_crop", lambda _: pytest.fail("No enhancement needed"))
    with Image.new("RGB", (100, 32)) as image:
        for mode, score in (("adaptive", .99), ("off", .5)):
            predictor = SimpleNamespace(predict=lambda patch, return_prob: ("Hà Nội", score))
            texts, scores, info, _ = read_text_crops(predictor, [image], mode)
            assert texts == ["Hà Nội"] and scores == [score]
            assert info["attempted_regions"] == 0 and info["status"] == "skipped"


def test_retry_work_is_bounded_for_a_page_with_many_uncertain_boxes(monkeypatch):
    import identification_demo.text_enhancement as module
    monkeypatch.setattr(module, "enhance_text_crop", lambda image: image.copy())
    monkeypatch.setattr(module, "needs_enhancement", lambda image: True)
    calls = []

    def predict(images, return_prob):
        calls.append(len(images))
        return (["Hà Ni"] * len(images), [.7] * len(images)) if len(calls) == 1 else (["Hà Nội"] * len(images), [.95] * len(images))

    images = [Image.new("RGB", (100, 32)) for _ in range(30)]
    try:
        texts, _scores, metadata, _timings = read_text_crops(SimpleNamespace(predict_batch=predict), images, "adaptive")
        assert calls == [30, 12] and len(texts) == 30
        assert metadata["attempted_regions"] == 12
        assert texts[:12] == ["Hà Nội"] * 12 and texts[12:] == ["Hà Ni"] * 18
    finally:
        for image in images:
            image.close()


@pytest.mark.parametrize("failure_stage", ["preprocess", "recognition"])
def test_optional_failure_preserves_original_text_and_closes_partial_crops(monkeypatch, failure_stage):
    import identification_demo.text_enhancement as module
    prepared = []

    def enhance(image):
        if prepared and failure_stage == "preprocess":
            raise RuntimeError("filter failed")
        prepared.append(image.copy())
        return prepared[-1]

    calls = []

    def predict(images, return_prob):
        calls.append(len(images))
        if len(calls) > 1:
            raise RuntimeError("optional inference failed")
        return ["Hà Nội", "Việt Nam"], [.7, .8]

    monkeypatch.setattr(module, "enhance_text_crop", enhance)
    monkeypatch.setattr(module, "needs_enhancement", lambda image: True)
    with Image.new("RGB", (200, 32)) as first, Image.new("RGB", (200, 32)) as second:
        texts, scores, info, _ = read_text_crops(SimpleNamespace(predict_batch=predict), [first, second], "adaptive")
        assert texts == ["Hà Nội", "Việt Nam"] and scores == [.7, .8]
        assert info["status"] == "error" and info["changed_regions"] == 0
        first.getpixel((0, 0))
    for image in prepared:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_optional_quality_check_failure_keeps_the_primary_reading(monkeypatch):
    import identification_demo.text_enhancement as module

    def failed(image):
        raise RuntimeError("quality check failed")

    monkeypatch.setattr(module, "needs_enhancement", failed)
    with Image.new("RGB", (200, 32)) as image:
        predictor = SimpleNamespace(predict=lambda image, return_prob: ("Hà Nội", .8))
        texts, scores, metadata, _ = read_text_crops(predictor, [image], "adaptive")
        assert texts == ["Hà Nội"] and scores == [.8]
        assert metadata["status"] == "error" and metadata["attempted_regions"] == 0
