"""Associate the printed expiry caption with its value, using OCR geometry."""
import math
import re
from datetime import datetime

from .documents import fold, text_rows

EXPIRY_LABEL = re.compile(
    r"\b(?:co\s*gia\s*tri(?:\s*(?:den|toi))?|gia\s*tri\s*(?:den|toi)|"
    r"ngay\s*(?:het\s*han|han\s*dung)|het\s*han|han\s*(?:su\s*dung|dung)|"
    r"thoi\s*han\s*su\s*dung|date\s*of\s*(?:expiry|expiration)|expiry\s*date|"
    r"expiration\s*date|valid\s*(?:until|till|through|to)|expires?(?:\s*on)?)\b"
)
OTHER_LABEL = re.compile(
    r"\b(?:ngay\s*sinh|date\s*of\s*birth|dob|ngay\s*cap|date\s*of\s*issue|"
    r"issued\s*on|ngay\s*,?\s*thang\s*,?\s*nam|date\s*,?\s*month\s*,?\s*year|"
    r"ho\s*(?:va\s*)?ten|full\s*name|gioi\s*tinh|sex|gender|quoc\s*tich|nationality|"
    r"que\s*quan|noi\s*(?:thuong\s*tru|cu\s*tru|cap)|co\s*quan\s*cap|"
    r"place\s*of\s*(?:birth|origin|residence|issue)|issuing\s*authority|"
    r"dac\s*diem\s*nhan\s*dang|personal\s*identification)\b"
)
UNLIMITED = re.compile(r"\b(?:khong\s*(?:co\s*)?thoi\s*han|vo\s*thoi\s*han|"
                       r"no\s*(?:expiry|expiration)|unlimited)\b")
DATE = re.compile(
    r"(?<!\d)(?:(?P<iso>\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2})|"
    r"(?P<dmy>\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{4})|"
    r"ngay\s*(?P<day>\d{1,2})\s*thang\s*(?P<month>\d{1,2})\s*nam\s*(?P<year>\d{4}))(?!\d)"
)


def expiry_rows(items):
    """Keep normalized text, original input indexes, and finite pixel bounds."""
    records = []
    for index, item in enumerate(items if isinstance(items, (list, tuple)) else []):
        texts = text_rows({"text_lines": [item]})
        if not texts:
            continue
        points = item.get("box", item.get("bbox")) if isinstance(item, dict) else None
        if isinstance(points, dict):
            points = points.get("coordinates")
        bounds = None
        try:
            if len(points) == 4 and all(isinstance(n, (int, float)) for n in points):
                bounds = tuple(float(n) for n in points)
            elif len(points) >= 4:
                xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
                bounds = (min(xs), min(ys), max(xs), max(ys))
            if bounds and (not all(math.isfinite(n) for n in bounds) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]):
                bounds = None
        except (ValueError, TypeError, IndexError):
            bounds = None
        records.append({"number": len(records) + 1, "text": texts[0], "bbox": bounds, "index": index})
    return records


def _values(text):
    values = [("unlimited", "")] if UNLIMITED.search(text) else []
    for match in DATE.finditer(text):
        if match["iso"]:
            year, month, day = re.split(r"\s*[-/.]\s*", match["iso"])
        elif match["dmy"]:
            day, month, year = re.split(r"\s*[/.-]\s*", match["dmy"])
        else:
            day, month, year = match["day"], match["month"], match["year"]
        try:
            datetime(int(year), int(month), int(day))
        except ValueError:
            continue
        value = f"{int(day):02d}/{int(month):02d}/{int(year):04d}"
        if ("date", value) not in values:
            values.append(("date", value))
    return values


def _union(records):
    boxes = [row["bbox"] for row in records]
    if not boxes or not all(boxes):
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _relation(anchor, value):
    """Rank values to the right on the same baseline, then below the caption."""
    height = max(anchor[3] - anchor[1], value[3] - value[1])
    overlap_y = min(anchor[3], value[3]) - max(anchor[1], value[1])
    gap_x = value[0] - anchor[2]
    if overlap_y >= .4 * min(anchor[3] - anchor[1], value[3] - value[1]):
        if -.5 * height <= gap_x <= max(12 * height, 1.5 * (anchor[2] - anchor[0])):
            return (0, max(0, gap_x) / height)
    gap_y = value[1] - anchor[3]
    overlap_x = min(anchor[2], value[2]) - max(anchor[0], value[0])
    aligned = overlap_x >= .25 * min(anchor[2] - anchor[0], value[2] - value[0])
    below_right = gap_y <= 2 * height and anchor[0] <= value[0] <= anchor[2] + 2 * height
    if -.3 * height <= gap_y <= 4 * height and (aligned or below_right or abs(value[0] - anchor[0]) <= 2 * height):
        return (1, max(0, gap_y) / height + abs(value[0] - anchor[0]) / max(height, anchor[2] - anchor[0]))
    return None


def _anchors(records):
    anchors = []
    for index, row in enumerate(records):
        # Form a fragmented caption in visual order when boxes are available.
        group = [row]
        neighbours = sorted((other for other in records if other["bbox"] and row["bbox"]
                             and other is not row and _relation(row["bbox"], other["bbox"]) is not None),
                            key=lambda other: (_relation(row["bbox"], other["bbox"]), other["bbox"][0]))
        for length in range(1, 4):
            text = fold(" ".join(item["text"] for item in group))
            label = EXPIRY_LABEL.search(text)
            if label and label.start() < len(fold(row["text"])):
                anchors.append((group, text, label))
                break
            if length == 3:
                break
            if row["bbox"]:
                if length > len(neighbours):
                    break
                following = neighbours[length - 1]
            elif index + length < len(records):
                following = records[index + length]
            else:
                break
            if OTHER_LABEL.search(fold(following["text"])):
                break
            group = [*group, following]
    return anchors


def _spatial_values(row, records):
    """A detector may split 25/10/ and 2036 into adjacent boxes."""
    found = _values(fold(row["text"]))
    if found:
        return found, [row]
    if not re.fullmatch(r"[\d\s/.-]+", row["text"]):
        return [], [row]
    group = [row]
    following = sorted((other for other in records if other is not row and other["bbox"]
                        and re.fullmatch(r"[\d\s/.-]+", other["text"]) and
                        (relation := _relation(row["bbox"], other["bbox"])) is not None and relation[0] == 0),
                       key=lambda other: other["bbox"][0])
    for other in following[:3]:
        previous = group[-1]["bbox"]
        if other["bbox"][0] - previous[2] > 2 * max(previous[3] - previous[1], other["bbox"][3] - other["bbox"][1]):
            break
        group.append(other)
        found = _values(" ".join(item["text"] for item in group))
        if found:
            return found, group
    return [], group


def expiry_reread_indices(items):
    """Select the expiry caption and nearby values for CPU re-reading."""
    records = expiry_rows(items)
    selected = set()
    for group, _, _ in _anchors(records):
        selected.update(row["index"] for row in group)
        bounds = _union(group)
        for row in records:
            if OTHER_LABEL.search(fold(row["text"])):
                continue
            if bounds and row["bbox"]:
                nearby = _relation(bounds, row["bbox"]) is not None
            else:
                nearby = 0 < row["number"] - group[-1]["number"] <= 3
            if nearby:
                selected.add(row["index"])
    return sorted(selected)[:12]


def has_expiry_label(items):
    return bool(_anchors(expiry_rows(items)))


def read_expiry(raw, side):
    """Resolve labelled dates from both OCR readings without guessing a value."""
    candidates = []
    for reading, key in (("primary", "text_lines"), ("backup", "paddle_text_lines")):
        if key == "paddle_text_lines" and text_rows(raw, key) == text_rows(raw):
            continue
        candidates.extend({**candidate, "reading": reading} for candidate in parse_expiry(raw.get(key, [])))
    values = {(candidate["expiry_type"], candidate["expiry_date"]) for candidate in candidates}
    kind, value = next(iter(values)) if len(values) == 1 else (None, "")
    label = has_expiry_label(raw.get("text_lines", [])) or has_expiry_label(raw.get("paddle_text_lines", []))
    return value, {"status": "conflict" if len(values) > 1 else "found" if values else "not_found",
                   "expiry_type": kind, "label_found": label, "side": side, "candidates": candidates}


def parse_expiry(items):
    records = expiry_rows(items)
    matches, seen = [], set()

    def add(values, evidence, method):
        evidence = [{"number": row["number"], "text": row["text"]} for row in evidence]
        for kind, value in values:
            signature = kind, value, tuple(row["number"] for row in evidence)
            if signature not in seen:
                seen.add(signature)
                matches.append({"expiry_date": value, "expiry_type": kind, "evidence": evidence, "method": method})

    for group, text, label in _anchors(records):
        tail = text[label.end():]
        boundary = OTHER_LABEL.search(tail)
        inline = _values(tail[:boundary.start()] if boundary else tail)
        if inline:
            add(inline, group, "inline")
            continue
        bounds = _union(group)
        spatial = []
        if bounds:
            for row in records:
                if row in group or not row["bbox"] or OTHER_LABEL.search(fold(row["text"])):
                    continue
                relation = _relation(bounds, row["bbox"])
                if relation is None:
                    continue
                # A date next to another field caption belongs to that field.
                if any(other["bbox"] and OTHER_LABEL.search(fold(other["text"]))
                       and (near := _relation(other["bbox"], row["bbox"])) is not None and near <= relation
                       for other in records):
                    continue
                values, value_rows = _spatial_values(row, records)
                if values:
                    spatial.append((relation, value_rows, values))
            if spatial:
                spatial.sort(key=lambda item: item[0])
                best = spatial[0][0]
                for rank, value_rows, values in spatial:
                    if rank[0] == best[0] and rank[1] <= best[1] + .5:
                        add(values, [*group, *value_rows], "layout")
                continue
        if boundary:
            continue
        # With pixel bounds, never override a failed spatial association by
        # borrowing an unrelated date merely because it follows in OCR order.
        following = records[group[-1]["number"]:group[-1]["number"] + 4]
        evidence, tail = list(group), text[label.end():]
        for row in following:
            if bounds and row["bbox"]:
                break
            next_text = fold(row["text"])
            if OTHER_LABEL.search(next_text):
                break
            evidence.append(row)
            tail += " " + next_text
            values = _values(tail)
            if values:
                add(values, evidence, "text_order")
                break

    for row in records:
        if UNLIMITED.fullmatch(fold(row["text"]).strip(" .:;/()-")):
            add([("unlimited", "")], [row], "inline")
    return matches
