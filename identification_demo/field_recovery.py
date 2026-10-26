"""Recover a missing date from one bounded, label-guided GPU region read."""
import logging
import re
from copy import deepcopy
from time import perf_counter

from .authority_region import _line
from .authority_recovery import _same_line
from .documents import ISSUE_LABEL, fold, parse_back
from .expiry import EXPIRY_LABEL, OTHER_LABEL, _anchors, _relation, _values, expiry_rows, parse_expiry

logger = logging.getLogger(__name__)
STRATEGY = "docu_regions_v2"


def expiry_values(rows):
    return {(item["expiry_type"], item["expiry_date"]) for item in parse_expiry(rows)}


def _caption_boxes(rows, side, label):
    groups = [group for group, _, _ in _anchors(expiry_rows(rows))] if side == "front" else [
        [record] for record in expiry_rows(rows) if label.search(fold(record["text"]))]
    boxes = []
    for group in groups:
        valid = [item["bbox"] for item in group if item["bbox"]]
        if valid:
            box = (min(b[0] for b in valid), min(b[1] for b in valid),
                   max(b[2] for b in valid), max(b[3] for b in valid))
            # Vietnamese/English captions in one small block are one anchor.
            for index, previous in enumerate(boxes):
                h = max(box[3]-box[1], previous[3]-previous[1])
                overlap = min(box[2], previous[2])-max(box[0], previous[0])
                if overlap > 0 and max(box[1], previous[1])-min(box[3], previous[3]) <= h:
                    boxes[index] = (min(box[0], previous[0]), min(box[1], previous[1]),
                                    max(box[2], previous[2]), max(box[3], previous[3]))
                    break
            else:
                boxes.append(box)
    return boxes


def _unassigned_dates(rows):
    records = list(map(_line, rows))
    date_owner = re.compile(r"ngay\s*sinh|date\s*of\s*birth|\bdob\b|ngay\s*cap|date\s*of\s*issue|ngay\s*,?\s*thang\s*,?\s*nam")
    return [box for text, box, _ in records if box and _values(fold(text))
            and re.fullmatch(r"[\d\s/.-]+", text) and
            not any(other_box and date_owner.search(fold(other)) and
                    _relation(other_box, box) is not None for other, other_box, _ in records)]


def recover_date(rows, image_size, side, read_region, needed_fields=None, expanded_context=False):
    """Return original text unless a local read resolves one unambiguous date.

    read_region returns text DTOs in the processed image's coordinates and the
    recognition-crop count. It reuses the caller's loaded detector/recognizer.
    """
    target = "expiry_date" if side == "front" else "issue_date"
    metadata = {"strategy": STRATEGY, "target": target, "status": "skipped",
                "reason": "not_requested", "reread_regions": 0, "elapsed_ms": 0}
    if needed_fields is not None and target not in needed_fields:
        return rows, metadata
    if side == "front":
        existing = expiry_values(rows)
        if existing:
            metadata["reason"] = "already_found" if len(existing) == 1 else "conflict"
            return rows, metadata
        label = EXPIRY_LABEL
    else:
        if parse_back(rows)["issue_date"]:
            metadata["reason"] = "already_found"
            return rows, metadata
        label = ISSUE_LABEL
    anchors = _caption_boxes(rows, side, label)
    anchor_kind = "caption"
    if not anchors and side == "front":
        anchors = _unassigned_dates(rows)
        anchor_kind = "unassigned_date"
    if len(anchors) != 1:
        metadata["reason"] = "label_not_found" if not anchors else (
            "ambiguous_date" if anchor_kind == "unassigned_date" else "ambiguous_label")
        return rows, metadata

    started = perf_counter()
    x1, y1, x2, y2 = anchors[0]
    height = max(12, y2 - y1)
    width, image_height = image_size
    if anchor_kind == "unassigned_date":
        margin = max(2*height, .6*(x2-x1))
        bounds = (max(0, int(x1-margin)), max(0, int(y1-4*height)),
                  min(width, int(x2+margin)), min(image_height, int(y2+height)))
    else:
        bounds = (max(0, int(x1 - .7 * height)), max(0, int(y1 - .3 * height)),
                  min(width, int(x2 + 2 * height)), min(image_height, int(y2 + 3 * height)))
    if expanded_context:
        # The caption may start far to the left of a date-only box, or its
        # date may be far to the right. Expand the local search, not the full
        # card detector's resolution. Keep vertical extent bounded as in v2.
        bounds = (0, bounds[1], width, bounds[3])
    metadata.update(bbox=list(bounds), anchor=anchor_kind, status="not_found", reason="no_valid_date")
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        metadata.update(status="skipped", reason="invalid_region")
        return rows, metadata
    try:
        recovered, count = read_region(bounds)
        metadata["reread_regions"] = count
        merged = deepcopy(rows)
        for item in recovered:
            text, box, confidence = _line(item)
            numeric_fragment = bool(re.fullmatch(r"[\d\s/.-]+", text))
            caption = bool(label.search(fold(text)))
            caption_fragment = bool(re.fullmatch(
                r"(?:co|gia|tri|den|toi|date|of|expiry|expiration|valid|until|ngay|het|han|cap|issue|thang|nam|[\s:/.,-])+",
                fold(text)))
            if not text or not box or confidence < .6 or not (
                    _values(fold(text)) or numeric_fragment or caption or caption_fragment):
                continue
            # A nearby birth date cannot become an expiry/issue date.
            if OTHER_LABEL.search(fold(text)) and not label.search(fold(text)):
                continue
            candidate = {"text": text, "box": [[box[0], box[1]], [box[2], box[1]],
                                                 [box[2], box[3]], [box[0], box[3]]],
                         "confidence": confidence}
            matches = [i for i, old in enumerate(merged) if _same_line(_line(old)[1], box)]
            if matches:
                if expanded_context and caption and _values(fold(text)):
                    existing_dates = {value for i in matches for value in _values(fold(_line(merged[i])[0]))}
                    if existing_dates and set(_values(fold(text))) == existing_dates:
                        # A wider read restores the caption around an already
                        # correct date. Preserve both original and new evidence.
                        merged.append(candidate)
                        continue
                # Correct an unreadable numeric line, preserving known dates
                # and captions from the first pass.
                if any(_values(fold(_line(merged[i])[0])) or
                       EXPIRY_LABEL.search(fold(_line(merged[i])[0])) or
                       OTHER_LABEL.search(fold(_line(merged[i])[0])) for i in matches):
                    # A second reading may recover a caption together with its
                    # value in the very same box. Preserve the caption.
                    if caption and _values(fold(text)) and not any(
                            _values(fold(_line(merged[i])[0])) for i in matches):
                        merged[matches[0]] = candidate
                    continue
                merged[matches[0]] = candidate
            else:
                merged.append(candidate)
        merged.sort(key=lambda item: ((_line(item)[1] or (0, 0, 0, 0))[1],
                                      (_line(item)[1] or (0, 0, 0, 0))[0]))
        if side == "front":
            values = expiry_values(merged)
            found = len(values) == 1
            if len(values) > 1:
                metadata["reason"] = "conflict"
        else:
            found = bool(parse_back(merged)["issue_date"])
        if found:
            metadata.update(status="recovered", reason="label_and_date")
            return merged, metadata
        return rows, metadata
    except Exception as exc:
        logger.warning("Local date recovery failed (%s); preserving initial OCR", type(exc).__name__)
        metadata.update(status="error", reason="local_read_failed")
        return rows, metadata
    finally:
        metadata["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
