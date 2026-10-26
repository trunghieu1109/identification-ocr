"""Use Paddle's existing text first; re-read only uncertain requested fields."""
from time import perf_counter

from .documents import FRONT_FIELDS, BACK_FIELDS, detect_side, fold, parse_back
from .authority import authority_word_score
from .authority_region import select_authority_region
from .namlhn_identity import extract_id_number, extract_name
from .namlhn_layout import reading_order
from .namlhn_mapping import map_front_fields


def _parse(lines, side):
    from ocr.parser import parse_idcard_fields
    from ocr.utils import to_parser_results
    if side == "back":
        return {}, parse_back(lines)
    source_lines = lines
    mapping = map_front_fields(lines)
    lines = reading_order(lines)
    number, box, validation = extract_id_number(lines, with_details=True)
    raw = parse_idcard_fields(to_parser_results([(item["text"], item["box"]) for item in lines]), number, box)
    name = extract_name(lines)
    if name is not None:
        raw["Name"], raw["Name_box"] = name
    for key, _label, upstream in FRONT_FIELDS:
        if key in mapping:
            if key == "id_number" and not mapping[key]["value"] and number:
                # Spatial uncertainty must not erase an observed, imperfect ID.
                mapping[key] = {**mapping[key], "value": number, "box": box, "method": "observed_number",
                                "status": "found", "value_indices": [index for index, line in enumerate(source_lines)
                                                                         if _inside(line["box"], box)]}
            raw[upstream] = mapping[key]["value"]
            raw[upstream + "_box"] = mapping[key]["box"]
    raw["_field_mapping"] = mapping
    value = str(raw.get("ID_number") or "").strip()
    if value:
        validation = {**validation, "format_valid": value.isascii() and value.isdigit() and len(value) in (9, 12),
                      "observed_length": len(value)}
        validation["status"] = "ambiguous" if validation["ambiguous"] else "format_valid" if validation["format_valid"] else "unverified"
    raw["_id_number_validation"] = validation
    return raw, {key: str(raw.get(upstream) or "").strip() for key, _, upstream in FRONT_FIELDS}


def _inside(box, target):
    if not box or not target:
        return False
    x = sum(p[0] for p in box) / len(box)
    y = sum(p[1] for p in box) / len(box)
    return min(p[0] for p in target) <= x <= max(p[0] for p in target) and min(p[1] for p in target) <= y <= max(p[1] for p in target)


def reread_indexes(lines, raw, fields, side, needed, threshold=.9):
    """A missing unlocated field warrants one broad fallback; QR-filled fields do not."""
    selected = set()
    for key in needed:
        if key == "expiry_date":
            from .expiry import expiry_reread_indices
            selected.update(expiry_reread_indices(lines))
            continue
        if side == "front":
            upstream = next(upstream for name, _, upstream in FRONT_FIELDS if name == key)
            box = raw.get(upstream + "_box")
            indexes = [i for i, line in enumerate(lines) if _inside(line["box"], box)]
        elif key == "issuing_authority":
            region = select_authority_region(lines)
            indexes = list(region.body_indices)
            if region.ambiguous:
                continue
            # Paddle's Latin recognizer can damage Vietnamese title anchors.
            # Include nearby title fragments in the same authority block.
            if region.method == "text_order":
                adjacent = {neighbor for index in indexes for neighbor in (index - 1, index + 1)
                            if 0 <= neighbor < len(lines) and authority_word_score(lines[neighbor]["text"]) >= 2}
                indexes = sorted(set(indexes) | adjacent)
        else:
            indexes = [i for i, line in enumerate(lines) if any(word in fold(line["text"]) for word in
                       (("ngay", "date", "/", "thang") if key == "issue_date" else ("nhan dang", "not", "seo")))]
        if not fields.get(key) and not indexes:
            return list(range(len(lines)))
        free_text = key in {"name", "place_of_origin", "place_of_residence", "issuing_authority"}
        selected.update(i for i in indexes if free_text or not fields.get(key) or lines[i].get("score", 0) < threshold)
    return sorted(selected)


def extract_pixels(extractor, image, side, needed_fields=None):
    import cv2
    import numpy as np
    from PIL import Image
    from ocr.preprocess import prepare_image
    from ocr.utils import corners_from_box

    timings = {}
    started = perf_counter()
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    processed = prepare_image(frame, enhance=extractor.enhance, resize=extractor.resize)
    timings["preprocess"] = round((perf_counter() - started) * 1000, 2)
    started = perf_counter()
    lines = extractor.detect(processed)
    timings["paddle_detect_recognize"] = round((perf_counter() - started) * 1000, 2)
    started = perf_counter()
    detected = detect_side([line["text"] for line in lines])
    actual_side = detected if side == "auto" and detected != "unknown" else "front" if side == "auto" else side
    schema = BACK_FIELDS if actual_side == "back" else FRONT_FIELDS
    needed = list(needed_fields) if needed_fields is not None else [key for key, *_ in schema] + ["expiry_date"]
    raw, primary = _parse(lines, actual_side)
    selected = reread_indexes(lines, raw, primary, actual_side, needed)
    timings["metadata"] = round((perf_counter() - started) * 1000, 2)
    started = perf_counter()
    patches, indexes = [], []
    for index in selected:
        patch = extractor._warp_region(processed, *corners_from_box(lines[index]["box"]), padding=True)
        patches.append(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
        indexes.append(index)
    combined = [dict(line) for line in lines]
    try:
        if patches:
            texts, probabilities = extractor._vietocr.predict_batch(patches, return_prob=True)
            for index, text, probability in zip(indexes, texts, probabilities):
                cleaned = extractor._postprocess_text(text)
                if cleaned and probability >= .6:
                    combined[index]["text"] = cleaned
    finally:
        for patch in patches:
            patch.close()
    timings["vietocr_reread"] = round((perf_counter() - started) * 1000, 2)
    started = perf_counter()
    _secondary_raw, secondary = _parse(combined, actual_side) if selected else (raw, primary)
    # A weaker second reading must not erase a value the first pass found.
    fields = {key: secondary.get(key) or value for key, value in primary.items()}
    if actual_side == "back" and select_authority_region(combined).ambiguous:
        fields["issuing_authority"] = ""
    timings["metadata"] = round(timings["metadata"] + (perf_counter() - started) * 1000, 2)
    result = {"structured_fields": fields, "document_side": detect_side([line["text"] for line in combined]),
            "text_lines": combined, "timings_ms": timings,
            "work": {"primary_regions": len(lines), "reread_regions": len(patches),
                     "reread_batches": int(bool(patches)), "full_image_attempts": 1}}
    if actual_side == "front":
        result["id_number_validation"] = (raw if fields.get("id_number") and not secondary.get("id_number") else _secondary_raw).get("_id_number_validation", {})
        mapping = dict(_secondary_raw.get("_field_mapping", {}))
        for key, item in raw.get("_field_mapping", {}).items():
            if fields[key] and not secondary.get(key):
                mapping[key] = item
        result["field_mapping"] = mapping
    if actual_side == "back":
        result["initial_text_lines"] = lines
        region = select_authority_region(combined)
        metadata = region.metadata((processed.shape[1], processed.shape[0]))
        if region.ambiguous:
            result["authority_region"] = {"status": "ambiguous", "method": "layout"}
        elif metadata:
            result["authority_region"] = metadata
    return result
