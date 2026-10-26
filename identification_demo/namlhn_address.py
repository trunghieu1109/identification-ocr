"""Group address captions and neighbouring OCR lines without changing their text."""
import re
from difflib import SequenceMatcher

from .documents import fold
from .expiry import DATE, EXPIRY_LABEL, OTHER_LABEL, UNLIMITED, expiry_rows


VIETNAMESE_LABELS = {
    "place_of_origin": re.compile(r"^que\s*quan\b"),
    "place_of_residence": re.compile(r"^noi\s*(?:thuong|cu)\s*tru\b"),
}
ENGLISH_LABELS = {"place_of_origin": "origin", "place_of_residence": "residence"}
ENGLISH_PREFIX = re.compile(r"^place\s*(?:of\s*)?([a-z]+)\b")


def _trim(text):
    return text.lstrip(" :;/,|.-").rstrip()


def _english_kind(word):
    ranked = sorted(((SequenceMatcher(None, word, expected).ratio(), kind)
                     for kind, expected in ENGLISH_LABELS.items()), reverse=True)
    return ranked[0][1] if ranked[0][0] >= .65 and ranked[0][0] - ranked[1][0] >= .15 else None


def _caption(text, context=None):
    """Tolerate damaged English captions only at a caption's start/in context."""
    text = _trim(text)
    text = re.sub(r"^[1Il]\s+(?=place\b)", "", text, flags=re.IGNORECASE)
    for kind, pattern in VIETNAMESE_LABELS.items():
        match = pattern.match(fold(text))
        if match:
            tail = _trim(text[match.end():])
            tail = re.sub(r"^[1Il]\s+(?=place\b)", "", tail, flags=re.IGNORECASE)
            english = ENGLISH_PREFIX.match(fold(tail))
            if english:
                # The Vietnamese anchor identifies the field even if its
                # English translation has recognition errors such as "rrgin".
                tail = _trim(tail[english.end():])
            elif fold(tail) in {"place of", "place", "1", "i", "l"}:
                tail = ""
            return kind, tail
    english = ENGLISH_PREFIX.match(fold(text))
    if english:
        if english[1] in {"birth", "issue", "issuance"}:
            return None
        kind = _english_kind(english[1])
        if kind or context:
            return kind or context, _trim(text[english.end():])
    if context and (fold(text) in {"place", "place of", "of"}
                    or _english_kind(fold(text)) == context):
        return context, ""
    return None


def _near(previous, candidate):
    if previous is None or candidate is None:
        return True
    height = max(previous[3] - previous[1], candidate[3] - candidate[1])
    if candidate[1] < previous[1] - .6 * height or candidate[1] - previous[3] > 2.5 * height:
        return False
    overlap = min(previous[2], candidate[2]) - max(previous[0], candidate[0])
    if overlap >= .2 * min(previous[2] - previous[0], candidate[2] - candidate[0]):
        return True
    # A Vietnamese/English caption or value may occupy separate boxes on one row.
    return (abs(candidate[1] - previous[1]) <= .6 * height
            and 0 <= candidate[0] - previous[2] <= 4 * height)


def address_fields(lines):
    records = expiry_rows(lines)
    fields = {}
    for index, anchor in enumerate(records):
        caption = _caption(anchor["text"])
        if caption is None:
            continue
        kind, inline = caption
        parts = [inline] if inline else []
        previous = anchor["bbox"]
        for following in records[index + 1:index + 9]:
            if not _near(previous, following["bbox"]):
                if previous and following["bbox"] and following["bbox"][1] > previous[3] + 2.5 * max(
                        previous[3] - previous[1], following["bbox"][3] - following["bbox"][1]):
                    break
                continue
            text = following["text"]
            continuation = _caption(text, context=kind)
            if continuation is not None:
                if continuation[0] != kind or (parts and continuation[1]):
                    break
                if continuation[1]:
                    parts.append(continuation[1])
                previous = following["bbox"] or previous
                continue
            if (OTHER_LABEL.match(fold(text)) or EXPIRY_LABEL.match(fold(text))
                    or DATE.fullmatch(text) or UNLIMITED.fullmatch(fold(text))):
                break
            if not parts or fold(parts[-1]) != fold(text):
                parts.append(text)
            previous = following["bbox"] or previous
            if len(parts) >= 4:
                break
        value = " ".join(parts).strip()
        # A caption alone must clear the upstream's spurious English value.
        if not fields.get(kind):
            fields[kind] = value
    return fields
