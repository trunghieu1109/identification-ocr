"""One bounded GPU re-detection for a missing institutional continuation."""
from difflib import SequenceMatcher
from statistics import median
import math

from .authority import LEADER_TITLE, authority_rules, authority_word_score, compact, incomplete_authority
from .authority_region import _line, _union, select_authority_region


def recovery_bounds(image_size, items):
    from .documents import AUTHORITY_LABEL, fold

    region = select_authority_region(items)
    if region.ambiguous or (region.text and not incomplete_authority(region.text)):
        return None
    records = [_line(item) for item in items]
    boxes = [records[index][1] for index in region.indices if records[index][1]]
    if not boxes:
        anchors = [box for text, box, _ in records if box and
                   (LEADER_TITLE.match(fold(text)) or AUTHORITY_LABEL.search(fold(text)))]
        # Separate title anchors may belong to separate cards. Do not guess.
        if len(anchors) != 1:
            return None
        boxes = anchors
    width, height = image_size
    left, top, right, bottom = _union(boxes)
    line_height = min(height * .08, max(4, median(box[3] - box[1] for box in boxes)))
    settings = authority_rules().get("recovery_crop", {})
    settings = settings if isinstance(settings, dict) else {}

    def setting(name, default, low, high):
        value = settings.get(name, default)
        return max(low, min(high, value)) if type(value) in (float, int) and math.isfinite(value) else default

    span = max((right - left) * 1.6 + 2 * line_height,
               width * setting("min_width_ratio", .66, .4, 1))
    span = max(right - left, min(width, span))
    center = (left + right) / 2
    # Shift the region inside the image instead of shrinking it at an edge.
    x1 = max(0, min(width - span, center - span / 2))
    below = max(setting("below_lines", 6, 2, 10) * line_height,
                setting("min_below_height_ratio", .16, 0, .35) * height)
    return (int(x1), max(0, int(top - setting("above_lines", .75, 0, 2) * line_height)),
            min(width, int(x1 + span)), min(height, int(bottom + below)))


def _same_line(first, second):
    if not first or not second:
        return False
    overlap_x = min(first[2], second[2]) - max(first[0], second[0])
    overlap_y = min(first[3], second[3]) - max(first[1], second[1])
    return (overlap_x >= .6 * min(first[2] - first[0], second[2] - second[0])
            and overlap_y >= .6 * min(first[3] - first[1], second[3] - second[1]))


def recover_authority(image_size, items, read_region, search_bounds=None):
    """Read a small area once; retain the original unless its name improves.

    read_region returns DTOs in the processed card's pixel coordinates plus
    the actual number of recognition crops. No CPU model is used here.
    """
    current = select_authority_region(items)
    if current.ambiguous or (current.text and not incomplete_authority(current.text)):
        return None
    bounds = search_bounds if search_bounds is not None else recovery_bounds(image_size, items)
    if bounds is None:
        return None
    before = select_authority_region(items)
    recovered, count = read_region(bounds)
    merged = list(items)
    for item in recovered:
        text, box, confidence = _line(item)
        if not text or not box or confidence < .6:
            continue
        matches = [index for index, old in enumerate(merged) if _same_line(_line(old)[1], box)]
        if matches:
            index = max(matches, key=lambda index: authority_word_score(_line(merged[index])[0]))
            original = _line(merged[index])[0]
            old_words, new_words = set(compact(original).split()), set(compact(text).split())
            similar = (SequenceMatcher(None, compact(original), compact(text)).ratio() >= .6
                       or len(old_words & new_words) >= .75 * max(1, len(old_words)))
            if not similar or authority_word_score(text) < authority_word_score(original):
                continue
            # Remove duplicate boxes from the same row before parsing geometry.
            merged[index] = item
            merged = [old for other, old in enumerate(merged) if other == index or other not in matches]
        else:
            merged.append(item)
    after = select_authority_region(merged)
    improved = (not after.ambiguous and bool(after.text) and
                authority_word_score(after.text) > authority_word_score(before.text))
    if improved:
        # Replace only the selected institution lines. A false detection in the
        # crop must not alter the issue date, features, or unrelated columns.
        original_indices = set(before.indices)
        selected = [merged[index] for index in after.indices]
        items[:] = [item for index, item in enumerate(items) if index not in original_indices] + selected
    return {"recovery_attempted": True, "recovered_lines": len(after.body_indices) if improved else 0,
            "search_bbox": list(bounds),
            "reread_regions": count, "reread_lines": count,
            "changed_lines": len(after.body_indices) if improved else 0}
