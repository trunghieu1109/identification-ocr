"""Associate observed identity values with their captions and spatial rows."""
import re
import unicodedata

from .documents import fold
from .expiry import DATE, EXPIRY_LABEL, OTHER_LABEL, _relation, expiry_rows
from .namlhn_layout import visual_rows


NAME_LABEL = re.compile(r"\b(?:ho\s*(?:va\s*)?ten|full\s*name|name)\b")
DOB_LABEL = re.compile(r"\b(?:ngay\s*sinh|dat[eo]\s*of\s*birth)\b")
ID_LABEL = re.compile(r"^(?:so(?:\s*(?:cccd|cmnd|dinh\s*danh|can\s*cuoc(?:\s*cong\s*dan)?))?|"
                      r"(?:id\s*)?(?:n[o0]|number)|cccd|cmnd)(?=\W|[0-9]|$)[ .:;/|,-]*")
SEPARATORS = " .:;/,|_-"
NUMBER_SPAN = re.compile(r"(?<![0-9])[0-9](?:[0-9 \t.-]*[0-9])?(?![0-9])")
WHOLE_NUMBER = re.compile(r"(?<![0-9])(?:[0-9]{12}|[0-9]{9})(?![0-9])")
OBSERVED_NUMBER = re.compile(r"(?<![A-Za-z0-9])[0-9OoIl](?:[0-9OoIl \t.-]*[0-9OoIl])?(?![A-Za-z0-9])")


def _box(lines):
    boxes = [row["bbox"] for row in expiry_rows(lines)]
    if not boxes or not all(boxes):
        return []
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _strip_number_caption(text):
    tail = unicodedata.normalize("NFC", text).strip(SEPARATORS)
    labelled = False
    while match := ID_LABEL.match(fold(tail)):
        labelled = True
        tail = tail[match.end():].strip(SEPARATORS)
    return tail, labelled


def _other_number_field(text):
    value = fold(text)
    return OTHER_LABEL.search(value) or DOB_LABEL.search(value) or EXPIRY_LABEL.search(value)


def _complete_numbers(text):
    """A whole number remains evidence even when its caption is misrecognized."""
    text = unicodedata.normalize("NFC", text)
    _tail, labelled = _strip_number_caption(text)
    for match in [*WHOLE_NUMBER.finditer(text), *NUMBER_SPAN.finditer(text)]:
        number = re.sub(r"\D", "", match[0])
        if (len(number) in (9, 12) and not DATE.search(match[0])
                and not _other_number_field(text[:match.start()])):
            yield number, labelled


def _number_part(text):
    tail, labelled = _strip_number_caption(text)
    if re.fullmatch(r"[0-9\s.-]+", tail) and not DATE.search(tail):
        return re.sub(r"\D", "", tail), labelled
    # A damaged English caption must not hide an adjacent numeric fragment.
    spans = list(NUMBER_SPAN.finditer(tail)) if labelled else []
    if len(spans) == 1 and not _other_number_field(tail[:spans[0].start()]) and not DATE.search(spans[0][0]):
        return re.sub(r"\D", "", spans[0][0]), labelled
    return ("", True) if labelled and not tail else None


def _observed_candidates(lines):
    """Retain imperfect ID readings without borrowing dates or address numbers."""
    candidates = []
    for row in [[line] for line in lines] + visual_rows(lines):
        text = " ".join(line.get("text", "") for line in row)
        tail, labelled = _strip_number_caption(text)
        if _other_number_field(text) and not labelled:
            continue
        dates = list(DATE.finditer(tail))
        for match in OBSERVED_NUMBER.finditer(tail):
            value = re.sub(r"[\s.-]", "", match[0])
            digit_count = sum(c.isdigit() for c in value)
            if (not digit_count or DATE.search(match[0]) or _other_number_field(tail[:match.start()])
                    or any(match.start() < date.end() and date.start() < match.end() for date in dates)
                    or not labelled and not 7 <= digit_count <= 20):
                continue
            # Geometry still matters when a caption and value occupy separate boxes.
            evidence = [line for line in row if any(c.isdigit() for c in line.get("text", ""))]
            if len(evidence) > 1:
                adjacent = all(
                    (b := _box([following])) and (a := _box([previous]))
                    and -.5 * max(a[2][1] - a[0][1], b[2][1] - b[0][1]) <= b[0][0] - a[1][0]
                    <= 3 * max(a[2][1] - a[0][1], b[2][1] - b[0][1])
                    for previous, following in zip(evidence, evidence[1:]))
                if not adjacent:
                    continue
            candidates.append((labelled, value, _box(evidence or row)))
    return candidates


def extract_id_number(lines, with_details=False):
    """Return observed ID text even when OCR produced an invalid-length reading."""
    candidates = []
    for line in lines:
        for number, labelled in _complete_numbers(line.get("text", "")):
            candidates.append((labelled, number, _box([line])))
    for row in visual_rows(lines):
        digits, evidence, labelled = "", [], False

        def retain():
            if len(digits) in (9, 12):
                candidates.append((labelled, digits, _box(evidence)))

        for line in row:
            part = _number_part(line.get("text", ""))
            previous, current = _box(evidence[-1:]), _box([line])
            adjacent = True
            if previous and current:
                height = max(previous[2][1] - previous[0][1], current[2][1] - current[0][1])
                gap = current[0][0] - previous[1][0]
                adjacent = -.5 * height <= gap <= 3 * height
            if part is None or not adjacent or len(digits) >= 12:
                retain()
                digits, evidence, labelled = "", [], False
            if part is not None:
                digits += part[0]
                labelled = labelled or part[1]
                evidence.append(line)
        retain()
    observed = _observed_candidates(lines)
    extensions = [item for item in observed if item[0] and len(item[1]) > 9
                  and any(old[0] and len(old[1]) == 9 and item[1].startswith(old[1]) for old in candidates)]
    # A value explicitly associated with Số/No. wins over an unrelated number.
    if extensions and not any(item[0] and len(item[1]) == 12 for item in candidates):
        candidates = extensions
    elif not any(item[0] for item in candidates) and any(item[0] for item in observed):
        candidates = [item for item in observed if item[0]]
    elif not candidates:
        candidates = observed
    if any(item[0] for item in candidates):
        candidates = [item for item in candidates if item[0]]
    # A 9-digit box can be the first fragment of a complete 12-digit CCCD.
    if any(len(item[1]) == 12 for item in candidates):
        candidates = [item for item in candidates if len(item[1]) == 12]
    # A failed format check is metadata, not a reason to erase recognized text.
    value, box = (candidates[0][1], candidates[0][2]) if candidates else ("", [])
    if not with_details:
        return value, box
    format_valid = bool(re.fullmatch(r"(?:[0-9]{9}|[0-9]{12})", value))
    ambiguous = len({item[1] for item in candidates}) > 1
    return value, box, {"status": "ambiguous" if ambiguous else "format_valid" if format_valid else "unverified" if value else "not_found",
                        "format_valid": format_valid, "observed_length": len(value), "ambiguous": ambiguous}


def _name_tail(text):
    text = text.strip(SEPARATORS)
    labels = list(NAME_LABEL.finditer(fold(text)))
    if not labels or labels[0].start() != 0:
        return None
    tail = text[labels[-1].end():].strip(SEPARATORS)
    # A separator or a second field caption is not part of an inline name.
    if fold(tail) in {"i", "1", "l", "full"}:
        return ""
    boundary = OTHER_LABEL.search(fold(tail)) or EXPIRY_LABEL.search(fold(tail))
    return tail[:boundary.start()].strip(SEPARATORS) if boundary else tail


def _is_name(text, continuation=False):
    value = fold(text)
    return (bool(text) and all(char.isalpha() or char in " '-’." for char in text)
            and not OTHER_LABEL.search(value) and not EXPIRY_LABEL.search(value)
            and value not in {"male", "female", "viet nam", "vietnam", "full"}
            and (value not in {"nam", "nu"} or (continuation and text.isupper()))
            and not re.search(r"\b(?:can\s*cuoc|cong\s*dan|cong\s*hoa|doc\s*lap|citizen|identity)\b", value))


def extract_name(lines):
    """Read inline, fragmented or wrapped names, skipping displaced DOB captions."""
    rows = []
    for group in visual_rows(lines):
        rows.append({"text": unicodedata.normalize("NFC", " ".join(line.get("text", "") for line in group)).strip(),
                     "box": _box(group)})
    records = expiry_rows(rows)
    found, saw_label = [], False
    for index, anchor in enumerate(records):
        inline = _name_tail(anchor["text"])
        if inline is None:
            continue
        saw_label = True
        if inline and not _is_name(inline):
            continue
        parts = [inline] if inline else []
        evidence = [rows[index]] if inline else []
        for following in records[index + 1:index + 7]:
            if anchor["bbox"] and following["bbox"] and _relation(anchor["bbox"], following["bbox"]) is None:
                continue
            text = following["text"]
            caption = _name_tail(text)
            if caption is not None:
                if parts:
                    break
                if not caption:
                    continue
                text = caption
            elif DOB_LABEL.search(fold(text)):
                if parts:
                    break
                continue
            if not _is_name(text, continuation=bool(parts)):
                if re.fullmatch(r"[0-9\s/.-]+", text) and not parts:
                    continue
                break
            parts.append(text)
            evidence.append(rows[following["index"]])
            if len(parts) == 3:
                break
        if parts:
            found.append((" ".join(parts), _box(evidence)))
    values = {value for value, _box_value in found}
    if not saw_label:
        return None
    return found[0] if len(values) == 1 else ("", [])
