"""Mild denoising and sharpening for OCR line crops, with an original-image fallback."""
import logging
import math
import re
import unicodedata
from time import perf_counter

from .engine import OCRUnavailable


ENHANCEMENT = "nlmeans_unsharp_v1"
MAX_ENHANCEMENT_CROPS = 12
RETRY_BELOW = .9
logger = logging.getLogger(__name__)


def needs_enhancement(image):
    """Avoid paying for another OCR pass on a sharp, clean line crop."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if min(gray.shape) < 3 or int(gray.max()) - int(gray.min()) < 8:
        return False
    low, high = np.percentile(gray, [1, 99])
    if high - low < 100:
        return True
    residual = np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32))
    noise = float(np.median(residual)) / .6745
    width = max(3, min(1024, round(gray.shape[1] * 32 / gray.shape[0])))
    scaled = cv2.resize(gray, (width, 32), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(scaled, cv2.CV_32F).var())
    return noise >= 3 or sharpness < 100


def enhance_text_crop(image):
    import cv2
    import numpy as np
    from PIL import Image

    pixels = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
    light = lab[:, :, 0]
    if int(light.max()) - int(light.min()) < 8:
        return image.copy()
    clean = cv2.fastNlMeansDenoising(light, None, h=5, templateWindowSize=7, searchWindowSize=15)
    low, high = np.percentile(clean, [5, 95])
    if high - low < 100:
        clean = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 2)).apply(clean)
    blurred = cv2.GaussianBlur(clean, (0, 0), .8)
    sharpened = cv2.addWeighted(clean, 1.5, blurred, -.5, 0)
    # Do not sharpen tiny intensity fluctuations in an otherwise flat region.
    mask = np.abs(clean.astype(np.int16) - blurred.astype(np.int16)) < 2
    sharpened[mask] = clean[mask]
    lab[:, :, 0] = sharpened
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def prefer_enhanced(original, original_score, enhanced, enhanced_score):
    """Confidence is a selection heuristic, not proof of a correct transcription."""
    original = unicodedata.normalize("NFC", str(original or "")).strip()
    enhanced = unicodedata.normalize("NFC", str(enhanced or "")).strip()
    if not enhanced or not math.isfinite(enhanced_score):
        return False
    if not math.isfinite(original_score):
        original_score = 0.
    if enhanced_score < max(.6, original_score + .015):
        return False
    if original and not .75 <= len(enhanced) / len(original) <= 1.35:
        return False
    # Enhancement must not silently change an already observed ID/date number.
    if original and re.findall(r"\d+", original) != re.findall(r"\d+", enhanced):
        return False
    return enhanced != original


def _predict(predictor, patches):
    if len(patches) == 1:
        text, score = predictor.predict(patches[0], return_prob=True)
        texts, scores = [text], [score]
    elif patches:
        texts, scores = predictor.predict_batch(patches, return_prob=True)
    else:
        texts, scores = [], []
    if len(texts) != len(patches) or len(scores) != len(patches):
        raise OCRUnavailable("Bộ đọc DocU trả thiếu kết quả nhận dạng dòng chữ.")
    return list(texts), [float(score) for score in scores]


def read_text_crops(predictor, patches, mode="off", readings=None):
    """Optionally retain exact model inputs/results; the caller owns image copies."""
    if mode not in {"off", "adaptive"}:
        raise ValueError("Text enhancement must be off or adaptive")
    started = perf_counter()
    texts, scores = _predict(predictor, patches)
    timings = {"docu_recognition": round((perf_counter() - started) * 1000, 2),
               "text_enhancement": 0., "enhanced_recognition": 0.}
    metadata = {"mode": mode, "method": ENHANCEMENT, "status": "skipped",
                "attempted_regions": 0, "changed_regions": 0}
    primary = None
    if readings is not None:
        primary = {"stage": "primary", "indexes": list(range(len(patches))),
                   "images": [patch.copy() for patch in patches],
                   "texts": list(texts), "scores": list(scores), "selected": [True] * len(patches)}
        readings.append(primary)
    if mode == "off":
        return texts, scores, metadata, timings
    enhanced = []
    stage = "text_enhancement"
    started = perf_counter()
    try:
        indexes = sorted((i for i, score in enumerate(scores)
                          if (not math.isfinite(score) or score < RETRY_BELOW) and needs_enhancement(patches[i])),
                         key=lambda i: scores[i] if math.isfinite(scores[i]) else -1)[:MAX_ENHANCEMENT_CROPS]
        indexes.sort()
        if not indexes:
            timings[stage] = round((perf_counter() - started) * 1000, 2)
            return texts, scores, metadata, timings
        for index in indexes:
            enhanced.append(enhance_text_crop(patches[index]))
        timings[stage] = round((perf_counter() - started) * 1000, 2)
        metadata["attempted_regions"] = len(enhanced)
        stage = "enhanced_recognition"
        started = perf_counter()
        new_texts, new_scores = _predict(predictor, enhanced)
        timings[stage] = round((perf_counter() - started) * 1000, 2)
        retry = None
        if readings is not None:
            retry = {"stage": "enhanced", "indexes": list(indexes),
                     "images": [patch.copy() for patch in enhanced],
                     "texts": list(new_texts), "scores": list(new_scores), "selected": [False] * len(indexes)}
            readings.append(retry)
        metadata["status"] = "applied"
        for position, (index, text, score) in enumerate(zip(indexes, new_texts, new_scores)):
            if prefer_enhanced(texts[index], scores[index], text, score):
                texts[index], scores[index] = text, score
                metadata["changed_regions"] += 1
                if primary is not None:
                    primary["selected"][index] = False
                    retry["selected"][position] = True
    except Exception as exc:
        timings[stage] = round((perf_counter() - started) * 1000, 2)
        metadata["status"] = "error"
        logger.warning("Optional OCR line enhancement failed (%s); preserving original reading", type(exc).__name__)
    finally:
        for patch in enhanced:
            patch.close()
    return texts, scores, metadata, timings
