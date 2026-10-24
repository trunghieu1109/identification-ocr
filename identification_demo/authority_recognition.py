"""Re-read authority lines with wider crops and conservative enhancement."""
from difflib import SequenceMatcher
from copy import copy
import logging
import re

from .authority import authority_rules, authority_word_score, compact, incomplete_authority
from .authority_region import select_authority_region
from .image_enhancement import enhance_text_patch, trim_text_band

logger = logging.getLogger(__name__)


def refine_authority(recognizer, image, items, province_names=()):
    if image is None or not authority_rules().get("image_enhancement", True):
        return None
    region = select_authority_region(items)
    indexes = list(region.body_indices)
    indexes = [index for index in indexes if getattr(items[index], "bbox", None) is not None][:6]
    if not indexes:
        return None
    boxes = [items[index].bbox.coordinates for index in region.indices if getattr(items[index], "bbox", None) is not None]
    left = min(float(point[0]) for box in boxes for point in box)
    right = max(float(point[0]) for box in boxes for point in box)
    config = recognizer.predictor.config["dataset"]
    original_width = config["image_max_width"]
    originals = {index: items[index].text for index in indexes}
    observed = compact(" ".join(region.parts))
    missing_head = (not re.match(r"^(?:cuc|cong an|bo|so|phong|ministry|department)\b", observed)
                    or (observed.startswith("cuc ") and incomplete_authority(region.text)))
    changed = 0
    hints = {compact(value) for value in authority_rules().get("line_hints", []) if isinstance(value, str)}
    patches, owners = [], []
    expanded_boxes = {}
    attempted = set()
    try:
        config["image_max_width"] = max(original_width, 1024)
        for index in indexes:
            item = items[index]
            # An exact known line with a strong first reading needs no repeat.
            if (compact(item.text) in hints and getattr(item, "confidence", 0) >= .9
                    and not incomplete_authority(region.text)):
                continue
            coordinates = item.bbox.coordinates
            start, end = min(float(p[0]) for p in coordinates), max(float(p[0]) for p in coordinates)
            width = max(1.0, end - start)
            padding = min(0.4, max(0.04, (start - left) / width + 0.03, (right - end) / width + 0.03))
            # When the first institutional line is clipped, include enough real
            # pixels to recover its beginning. Do not fill words from a template.
            if missing_head and index == indexes[0]:
                padding = max(padding, .7)
            patch = recognizer.crop_oriented_box(image, coordinates, pad_top=0.08, pad_bot=0.02, pad_x=padding)
            if patch is not None:
                patches.extend((patch, enhance_text_patch(patch)))
                owners.extend((index, index))
                attempted.add(index)
                # Keep metadata in the same pixel coordinates as the actual
                # wider read, so a recovered word is inside the returned ROI.
                y1, y2 = min(p[1] for p in coordinates), max(p[1] for p in coordinates)
                expanded_boxes[index] = [(max(0, start - width * padding), max(0, y1 - (y2 - y1) * .08)),
                                         (end + width * padding, max(0, y1 - (y2 - y1) * .08)),
                                         (end + width * padding, y2 + (y2 - y1) * .02),
                                         (max(0, start - width * padding), y2 + (y2 - y1) * .02)]
        # If the detector missed a province printed on the next line, read one
        # small band below "CÔNG AN TỈNH". Require an exact dictionary match;
        # never interpret a person's name or an arbitrary OCR result as a place.
        suffix = None
        known_provinces = {compact(value) for value in province_names}
        if len(indexes) < 6 and known_provinces and re.search(r"\b(?:tinh|thanh pho)$", compact(" ".join(region.parts))):
            last = items[indexes[-1]]
            box = last.bbox.coordinates
            top, bottom = min(p[1] for p in box), max(p[1] for p in box)
            height = max(1, bottom - top)
            coordinates = [(max(0, left - .4 * height), bottom + .02 * height),
                           (min(image.shape[1], right + .4 * height), bottom + .02 * height),
                           (min(image.shape[1], right + .4 * height), min(image.shape[0], bottom + 1.35 * height)),
                           (max(0, left - .4 * height), min(image.shape[0], bottom + 1.35 * height))]
            patch = recognizer.crop_oriented_box(image, coordinates, pad_top=.02, pad_bot=.02, pad_x=.02)
            if patch is not None:
                with trim_text_band(patch) as trimmed:
                    enhanced = enhance_text_patch(trimmed)
                patches.extend((patch, enhanced))
                owners.extend((-1, -1))
                attempted.add(-1)
                suffix = copy(last)
                suffix.bbox = copy(last.bbox)
                suffix.bbox.coordinates = coordinates
                suffix.text, suffix.confidence = "", 0
        scores = {index: (authority_word_score(items[index].text), getattr(items[index], "confidence", 0))
                  if index >= 0 else (-1, 0) for index in attempted}
        if patches:
            try:
                batch = getattr(recognizer.predictor, "predict_batch", None)
                if batch:
                    texts, probabilities = batch(patches, return_prob=True)
                    predictions = list(zip(texts, probabilities))
                else:
                    predictions = [recognizer.predictor.predict(patch, return_prob=True) for patch in patches]
                for index, (text, probability) in zip(owners, predictions):
                    item = items[index] if index >= 0 else suffix
                    text = text.strip() if isinstance(text, str) else ""
                    probability = float(probability) if probability is not None else 0.0
                    if index == -1:
                        if probability < .6 or compact(text) not in known_provinces:
                            continue
                    else:
                        original_words = set(compact(originals[index]).split())
                        extension = (incomplete_authority(originals[index]) and original_words
                                     and original_words <= set(compact(text).split()))
                        if (not text or probability < .6 or
                                (not extension and SequenceMatcher(None, compact(originals[index]), compact(text)).ratio() < .6)):
                            continue
                    candidate = (authority_word_score(text), probability)
                    if candidate > scores[index]:
                        item.text, scores[index] = text, candidate
            except Exception as exc:
                logger.warning("Authority reread failed (%s); preserving recognized text", type(exc).__name__)
        changed = sum(items[index].text != original for index, original in originals.items())
        for index, original in originals.items():
            if items[index].text != original and index in expanded_boxes:
                items[index].bbox = copy(items[index].bbox)
                items[index].bbox.coordinates = expanded_boxes[index]
        if suffix is not None and suffix.text:
            items.append(suffix)
            changed += 1
    finally:
        config["image_max_width"] = original_width
        for patch in patches:
            patch.close()
    return {"target": "issuing_authority", "reread_lines": len(attempted), "changed_lines": changed,
            "reread_regions": len(patches), "contrast_enhancement": bool(patches), "expanded_crop": bool(patches)}
