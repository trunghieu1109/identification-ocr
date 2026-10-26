"""Assign front-card text to caption boxes before ordering each field's values."""
import re

from .documents import fold
from .expiry import DATE, EXPIRY_LABEL, OTHER_LABEL, _relation, _values, expiry_rows
from .namlhn_address import _caption as address_caption
from .namlhn_identity import DOB_LABEL, ID_LABEL, NAME_LABEL, _complete_numbers, _is_name, _number_part, extract_id_number
from .namlhn_layout import projected_bounds, reading_order, visual_rows


LABELS = {"name": NAME_LABEL, "date_of_birth": DOB_LABEL,
          "gender": re.compile(r"\b(?:gioi\s*tinh|sex|gender)\b"),
          "nationality": re.compile(r"\b(?:quoc\s*tich|nationality)\b")}
MULTILINE = {"name", "place_of_origin", "place_of_residence"}
TRIM = " .:;：/,|_-"
SPLIT_LABELS = {**LABELS,
                "place_of_origin": re.compile(r"\b(?:que\s*quan|place\s*of\s*origin)\b"),
                "place_of_residence": re.compile(r"\b(?:noi\s*(?:thuong|cu)\s*tru|place\s*of\s*residence)\b"),
                "boundary": EXPIRY_LABEL}


def _caption(text):
    text = text.strip(TRIM)
    address = address_caption(text)
    if address:
        return address
    match = ID_LABEL.match(fold(text))
    if match:
        # Keep the observed caption: the number reader already handles N0/No.
        return "id_number", text
    for key, pattern in LABELS.items():
        matches = list(pattern.finditer(fold(text)))
        if matches and matches[0].start() == 0:
            return key, text[matches[-1].end():].strip(TRIM)
    if EXPIRY_LABEL.match(fold(text)) or OTHER_LABEL.match(fold(text)):
        return "boundary", ""
    return None


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _polygon(bounds):
    left, top, right, bottom = bounds
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _distance(anchor, box):
    relation = _relation(anchor, box)
    if relation is None:
        return None
    # A far-away caption on the same row must not beat a nearby caption above.
    return relation[1] / 4 if relation[0] == 0 else 1 + relation[1]


def _caption_fragment(text, key):
    if key.startswith("place_of_"):
        caption = address_caption(text, context=key)
        return caption is not None and caption == (key, "")
    fragments = {"name": {"full", "name", "full name", "i", "1", "l"},
                 "date_of_birth": {"date", "date of", "dato of", "of birth", "birth"}}
    return fold(text).strip(TRIM) in fragments.get(key, set())


def _split_record(record):
    """One detector box can contain e.g. both gender and nationality captions."""
    caption = _caption(record["text"])
    if caption is None:
        return [record]
    boundaries = [match.start() for key, pattern in SPLIT_LABELS.items() if key != caption[0]
                  for match in pattern.finditer(fold(record["text"])) if match.start() > 0]
    if not boundaries:
        return [record]
    start = min(boundaries)
    left, top, right, bottom = record["bbox"]
    middle = left + (right - left) * start / len(record["text"])
    first = dict(record, text=record["text"][:start].rstrip(), bbox=(left, top, middle, bottom))
    following = dict(record, text=record["text"][start:].lstrip(), bbox=(middle, top, right, bottom))
    source = record["source_bbox"]
    source_middle = source[0] + (source[2] - source[0]) * start / len(record["text"])
    first["source_bbox"] = (source[0], source[1], source_middle, source[3])
    following["source_bbox"] = (source_middle, source[1], source[2], source[3])
    return [first, *_split_record(following)]


def _anchors(records):
    anchors, remaining = [], []
    for record in records:
        caption = _caption(record["text"])
        if caption is None:
            remaining.append(record)
            continue
        key, inline = caption
        anchors.append({"key": key, "bbox": record["bbox"], "indices": [record["index"]],
                        "inline": [dict(record, text=inline)] if inline else []})
    anchors.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    # Vietnamese/English copies of a caption identify the same field region.
    merged = []
    for anchor in anchors:
        neighbours = [other for other in merged if other["key"] == anchor["key"] != "boundary"
                      and (not other["inline"] or not anchor["inline"])
                      and (distance := _distance(other["bbox"], anchor["bbox"])) is not None and distance < 2.5]
        if neighbours:
            previous = min(neighbours, key=lambda item: _distance(item["bbox"], anchor["bbox"]))
            previous["bbox"] = _union([previous["bbox"], anchor["bbox"]])
            previous["indices"].extend(anchor["indices"])
            previous["inline"].extend(anchor["inline"])
        else:
            merged.append(anchor)
    # Caption fragments such as "Place of" / "rrgin" may themselves be boxes.
    for _ in range(3):
        pending = []
        for record in remaining:
            neighbours = [(distance, anchor) for anchor in merged
                          if _caption_fragment(record["text"], anchor["key"])
                          and (distance := _distance(anchor["bbox"], record["bbox"])) is not None and distance < 3]
            if not neighbours:
                pending.append(record)
                continue
            anchor = min(neighbours, key=lambda item: item[0])[1]
            anchor["bbox"] = _union([anchor["bbox"], record["bbox"]])
            anchor["indices"].append(record["index"])
        remaining = pending
    return merged, remaining


def _compatible(key, text):
    if key == "id_number":
        part = _number_part(text)
        return (part is not None and bool(part[0]) and len(part[0]) <= 32) or any(_complete_numbers(text))
    if key == "date_of_birth":
        return bool(re.fullmatch(r"[0-9\s/.-]+", text) and (re.search(r"[/.-]", text) or len(text.strip()) <= 4))
    if key == "gender":
        return fold(text).strip(TRIM) in {"nam", "nu", "male", "female"}
    if key == "name":
        return _is_name(text, continuation=True)
    if key == "nationality":
        return bool(text and all(c.isalpha() or c in " '-’" for c in text)
                    and not OTHER_LABEL.search(fold(text)) and not EXPIRY_LABEL.search(fold(text)))
    if key.startswith("place_of_"):
        return bool(text and not OTHER_LABEL.search(fold(text)) and not EXPIRY_LABEL.search(fold(text))
                    and not DATE.fullmatch(text) and not re.fullmatch(r"[0-9\s.-]{9,}", text)
                    and not re.search(r"\b(?:can\s*cuoc|cong\s*dan|citizen|identity)\b", fold(text)))
    return False


def _blocked(anchor, candidate, anchors):
    a, c = anchor["bbox"], candidate["bbox"]
    for other in anchors:
        if other is anchor or other["key"] == anchor["key"]:
            continue
        # A displaced birth caption can precede an alphabetic name value.
        if anchor["key"] == "name" and other["key"] == "date_of_birth":
            continue
        b = other["bbox"]
        height = min(a[3] - a[1], b[3] - b[1], c[3] - c[1])
        if a[1] + .5 * height < b[1] < c[1] + .3 * height:
            if _distance(b, c) is not None and abs(b[0] - a[0]) < 2 * height:
                return True
        overlap = min(a[3], b[3], c[3]) - max(a[1], b[1], c[1])
        if overlap > .4 * height and a[2] <= b[0] and b[2] <= c[0]:
            return True
    return False


def _value(key, records):
    lines = [{"text": record["text"], "box": _polygon(record["bbox"]), "index": record["index"]}
             for record in records]
    ordered = reading_order(lines)
    parts, indices = [], []
    for line in ordered:
        if not parts or fold(parts[-1]) != fold(line["text"]):
            parts.append(line["text"])
        indices.append(line["index"])
    text = " ".join(parts).strip(TRIM)
    if key == "id_number":
        value = extract_id_number(lines)[0]
    elif key == "date_of_birth":
        values = {value for kind, value in _values(fold(text)) if kind == "date"}
        value = next(iter(values)) if len(values) == 1 else ""
    elif key == "gender":
        values = {"Nữ" if fold(part).strip(TRIM) in {"nu", "female"} else "Nam" for part in parts
                  if fold(part).strip(TRIM) in {"nam", "nu", "male", "female"}}
        value = next(iter(values)) if len(values) == 1 else ""
    elif key == "nationality":
        value = "Việt Nam" if fold(text) in {"viet nam", "vietnam"} else text
    elif key == "name":
        value = text if _is_name(text, continuation=True) else ""
    else:
        value = text
    return value, indices


def map_front_fields(lines):
    """Use caption/value geometry; original array positions are evidence only."""
    records = expiry_rows(lines)
    if not records or any(record["bbox"] is None for record in records):
        return {}
    bounds = projected_bounds(lines)
    records = [dict(record, source_bbox=record["bbox"],
                    bbox=bounds[record["index"]] if bounds is not None else record["bbox"]) for record in records]
    records = [part for record in records for part in _split_record(record)]
    anchors, remaining = _anchors(records)
    owned = {id(anchor): list(anchor["inline"]) for anchor in anchors}
    for record in remaining:
        options = []
        for anchor in anchors:
            if not _compatible(anchor["key"], record["text"]):
                continue
            distance = _distance(anchor["bbox"], record["bbox"])
            if distance is None or _blocked(anchor, record, anchors):
                continue
            if anchor["inline"] and anchor["key"] not in MULTILINE:
                current = _value(anchor["key"], anchor["inline"])[0]
                if current and (anchor["key"] != "id_number" or len(current) == 12):
                    continue
            options.append((distance, anchor))
        options.sort(key=lambda item: item[0])
        if not options:
            continue
        if len(options) > 1 and options[0][1]["key"] != options[1][1]["key"] and options[1][0] - options[0][0] < .15:
            continue
        owned[id(options[0][1])].append(record)
    fields = {}
    for anchor in anchors:
        key = anchor["key"]
        if key == "boundary":
            continue
        values = owned[id(anchor)]
        if key == "nationality" and values and not anchor["inline"]:
            # A short scalar value must not absorb unrelated text below it.
            rows = visual_rows([{"box": _polygon(record["bbox"]), "record": record} for record in values])
            nearest = min(rows, key=lambda row: min(_distance(anchor["bbox"], item["record"]["bbox"]) for item in row))
            values = [item["record"] for item in nearest]
        value, indices = _value(key, values)
        item = {"value": value, "box": _polygon(_union([record["source_bbox"] for record in values])) if values else [],
                "label_indices": sorted(anchor["indices"]), "value_indices": indices,
                "method": "spatial", "status": "found" if value else "unresolved" if values else "not_found"}
        if key in fields and fields[key]["value"] != value:
            fields[key] = dict(item, value="", box=[], status="ambiguous")
        else:
            fields[key] = item
    return fields
