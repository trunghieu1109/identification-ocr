"""Paddle text boxes, DocU line crops/recognition, and the namlhn parser."""
import math
import unicodedata
from time import perf_counter

from .authority_region import select_authority_region
from .cpu_pipeline import _parse
from .card_preprocessing import prepare_card, project_polygon
from .documents import detect_side
from .namlhn_address import address_fields
from .namlhn_layout import reading_order
from .text_enhancement import read_text_crops


STRATEGY = "namlhn_docu_v1"


def _fields(lines, side, with_mapping=False):
    lines = reading_order(lines)
    raw, fields = _parse(lines, side)
    mapping = raw.get("_field_mapping", {})
    if side == "front":
        fields.update({key: value for key, value in address_fields(lines).items() if key not in mapping})
    return (fields, mapping, raw.get("_id_number_validation", {})) if with_mapping else fields


def _detections(results, scale_x, scale_y):
    """Retain detector boxes without filtering them through Paddle recognition."""
    lines = []
    for result in results:
        payload = getattr(result, "json", result)
        payload = payload.get("res", payload)
        scores = payload.get("dt_scores", [])
        for index, polygon in enumerate(payload.get("dt_polys", [])):
            try:
                box = [[float(x) * scale_x, float(y) * scale_y] for x, y in polygon]
                score = float(scores[index]) if index < len(scores) else 0.0
            except (TypeError, ValueError):
                continue
            if (len(box) != 4 or not all(math.isfinite(n) for point in box for n in point)
                    or max(x for x, _ in box) - min(x for x, _ in box) <= 2
                    or max(y for _, y in box) - min(y for _, y in box) <= 2):
                continue
            lines.append({"box": box, "detection_score": score if math.isfinite(score) else 0.0})
    return lines


def extract_pixels(detector, recognizer, image, side, text_enhancement="off", readings=None,
                   card_preprocessor=None, card_enhancement="off", artifacts=None):
    """Optional readings retain caller-owned input crops, including blank OCR results."""
    import cv2
    import numpy as np
    from ocr.preprocess import prepare_image

    timings = {}
    started = perf_counter()
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    original_size = (frame.shape[1], frame.shape[0])
    frame, inverse_transform, image_preprocessing = prepare_card(frame, card_preprocessor, card_enhancement, artifacts)
    timings["card_preprocessing"] = round((perf_counter() - started) * 1000, 2)
    started = perf_counter()
    # Keep detector sharpening separate from the rectified card used for recognition.
    processed = prepare_image(frame, enhance=True, resize=True)
    timings["preprocess"] = round((perf_counter() - started) * 1000, 2)
    started = perf_counter()
    lines = _detections(detector.predict(processed), frame.shape[1] / processed.shape[1],
                        frame.shape[0] / processed.shape[0])
    timings["paddle_detection"] = round((perf_counter() - started) * 1000, 2)
    started = perf_counter()
    patches, selected = [], []
    try:
        for line in reading_order(lines):
            patch = recognizer.crop_oriented_box(
                frame, line["box"], pad_top=recognizer.pad_top,
                pad_bot=recognizer.pad_bot, pad_x=recognizer.pad_x,
            )
            if patch is not None:
                patches.append(patch)
                selected.append(line)
        timings["line_crops"] = round((perf_counter() - started) * 1000, 2)
        reading_start = len(readings) if readings is not None else 0
        capture = {"readings": readings} if readings is not None else {}
        texts, probabilities, enhancement, recognition_timings = read_text_crops(
            recognizer.predictor, patches, text_enhancement, **capture)
        timings.update(recognition_timings)
        for line, text, probability in zip(selected, texts, probabilities):
            probability = float(probability)
            line.update(text=unicodedata.normalize("NFC", str(text or "")).strip(),
                        score=probability if math.isfinite(probability) else 0.0,
                        confidence=probability if math.isfinite(probability) else 0.0,
                        recognition_source="docu_vietocr")
    finally:
        for patch in patches:
            patch.close()
    started = perf_counter()
    # Every retained text is from the DocU reader. No Paddle text can overwrite it.
    final = [line for line in selected if line["text"]]
    detected = detect_side([line["text"] for line in final])
    actual_side = detected if side == "auto" and detected != "unknown" else "front" if side == "auto" else side
    fields, mapping, id_validation = _fields(final, actual_side, with_mapping=True)
    result = {
        "structured_fields": fields, "document_side": detected, "text_lines": final,
        "strategy": STRATEGY, "coordinate_space": "original_image",
        "text_enhancement": enhancement,
        "field_mapping": mapping,
        "image_preprocessing": image_preprocessing,
        **({"id_number_validation": id_validation} if actual_side == "front" else {}),
        "timings_ms": timings,
        "work": {"primary_regions": len(lines), "recognized_regions": len(patches),
                 "recognition_batches": int(bool(patches)) + int(bool(enhancement["attempted_regions"])),
                 "enhanced_regions": enhancement["attempted_regions"],
                 "enhancement_changed_regions": enhancement["changed_regions"],
                 "reread_regions": enhancement["attempted_regions"],
                 "full_image_attempts": 1},
    }
    if actual_side == "back":
        result["initial_text_lines"] = final
        region = select_authority_region(final)
        if region.ambiguous:
            fields["issuing_authority"] = ""
            result["authority_region"] = {"status": "ambiguous", "method": "layout"}
        else:
            metadata = region.metadata((frame.shape[1], frame.shape[0]))
            if metadata:
                result["authority_region"] = metadata
    if not np.allclose(inverse_transform, np.eye(3)):
        for line in selected:
            line["working_box"] = line["box"]
            line["box"] = project_polygon(line["box"], inverse_transform, original_size)
        for item in mapping.values():
            item["box"] = project_polygon(item["box"], inverse_transform, original_size)
        region = result.get("authority_region", {})
        if region.get("bbox"):
            left, top, right, bottom = region["bbox"]
            points = project_polygon([[left, top], [right, top], [right, bottom], [left, bottom]], inverse_transform, original_size)
            region["bbox"] = [min(p[0] for p in points), min(p[1] for p in points),
                              max(p[0] for p in points), max(p[1] for p in points)]
            region["image_size"] = list(original_size)
    if readings is not None:
        for reading in readings[reading_start:]:
            reading["regions"] = [dict(selected[index]) for index in reading["indexes"]]
    timings["metadata"] = round((perf_counter() - started) * 1000, 2)
    return result
