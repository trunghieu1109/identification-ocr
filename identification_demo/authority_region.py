"""Select an issuing institution by its title anchor and neighbouring text boxes."""
import math
import re
from dataclasses import dataclass

from .authority import (LEADER_TITLE, authority_word_score, compact, has_institution_name,
                        normalize_authority, strip_authority_title)


@dataclass
class AuthorityRegion:
    indices: tuple = ()
    body_indices: tuple = ()
    parts: tuple = ()
    bbox: tuple | None = None
    method: str = "text_order"
    anchor: str = "institution"
    ambiguous: bool = False

    @property
    def text(self):
        value = normalize_authority(" ".join(self.parts))
        return value if has_institution_name(value) else ""

    def metadata(self, image_size):
        if not self.bbox or not self.text:
            return None
        width, height = image_size
        box = (max(0, min(width, self.bbox[0])), max(0, min(height, self.bbox[1])),
               max(0, min(width, self.bbox[2])), max(0, min(height, self.bbox[3])))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        return {"method": self.method, "bbox": [round(value, 2) for value in box],
                "image_size": list(image_size), "line_count": len(self.body_indices), "anchor": self.anchor}


def _line(item):
    if isinstance(item, str):
        return item.strip(), None, 0
    text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
    if not isinstance(text, str):
        return "", None, 0
    points = item.get("box", item.get("bbox")) if isinstance(item, dict) else getattr(item, "bbox", None)
    if isinstance(points, dict):
        points = points.get("coordinates")
    else:
        points = getattr(points, "coordinates", points)
    bounds = None
    try:
        if points is not None and len(points) >= 4:
            xs, ys = [float(point[0]) for point in points], [float(point[1]) for point in points]
            if all(math.isfinite(value) for value in xs + ys) and max(xs) > min(xs) and max(ys) > min(ys):
                bounds = (min(xs), min(ys), max(xs), max(ys))
    except (ValueError, TypeError, IndexError):
        pass
    confidence = item.get("score", item.get("confidence", 0)) if isinstance(item, dict) else getattr(item, "confidence", 0)
    confidence = float(confidence) if type(confidence) in (int, float) and math.isfinite(confidence) else 0
    return re.sub(r"\s+", " ", text).strip(), bounds, max(0, min(1, confidence))


def _union(boxes):
    return (min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes))


def _aligned(first, second):
    overlap = min(first[2], second[2]) - max(first[0], second[0])
    return overlap >= .35 * min(first[2] - first[0], second[2] - second[0])


def _joined(parts):
    joined = []
    for part in parts:
        words = part.split()
        if joined:
            previous = joined[-1].split()
            for overlap in range(min(len(previous), len(words)), 0, -1):
                if compact(" ".join(previous[-overlap:])) == compact(" ".join(words[:overlap])):
                    words = words[overlap:]
                    break
        if words:
            joined.append(" ".join(words))
    return tuple(joined)


def select_authority_region(items):
    # Imported at call time: documents also exposes this selector through its
    # historical authority_block() API and provides the text-only fallback.
    from .documents import (AUTHORITY_LABEL, AUTHORITY_CONTINUATION, DATE, FEATURE_LABEL,
                            FINGERPRINT, INSTITUTION, ISSUE_LABEL, WORD_DATE, _authority_text_block,
                            _is_mrz, fold)

    records = [_line(item) for item in items] if isinstance(items, (list, tuple)) else []
    texts = [record[0] for record in records]

    def content(text):
        labels = list(AUTHORITY_LABEL.finditer(fold(text)))
        if labels:
            text = text[labels[-1].end():].strip(" :;/,-")
        else:
            institution = INSTITUTION.search(fold(text))
            if institution and (DATE.search(text[:institution.start()]) or ISSUE_LABEL.search(fold(text[:institution.start()]))):
                text = text[institution.start():].strip()
        return strip_authority_title(text)

    def boundary(text):
        value = fold(text)
        return bool(ISSUE_LABEL.search(value) or FEATURE_LABEL.search(value) or FINGERPRINT.search(value)
                    or WORD_DATE.search(value) or DATE.search(text) or _is_mrz(text)
                    or re.search(r"^(?:nguoi\s*ky|chu\s*ky|signature|signed|ho\s*(?:va\s*)?ten|"
                                 r"(?:thieu|trung|thuong|dai)\s*(?:tuong|ta|uy))\b", value))

    if sum(record[1] is not None for record in records) < 2:
        indices, _parts = _authority_text_block(texts)
        body = tuple(index for index in indices if content(texts[index]))
        parts = _joined([content(texts[index]) for index in body])
        if parts and compact(" ".join(parts)) == "bo cong an" and not any(AUTHORITY_LABEL.search(fold(texts[i])) or LEADER_TITLE.match(fold(texts[i])) for i in indices):
            return AuthorityRegion()
        return AuthorityRegion(tuple(indices), body, parts,
                               _union([records[i][1] for i in body if records[i][1]]) if any(records[i][1] for i in body) else None)

    candidates = []
    for index, (text, anchor_box, _confidence) in enumerate(records):
        if not anchor_box or not text:
            continue
        value = fold(text)
        label = bool(AUTHORITY_LABEL.search(value))
        title = bool(LEADER_TITLE.match(value))
        inline_date = (DATE.search(text) or WORD_DATE.search(value) or ISSUE_LABEL.search(value)) and INSTITUTION.search(fold(content(text)))
        if boundary(text) and not label and not inline_date:
            continue
        agency = bool(INSTITUTION.search(fold(content(text))))
        if not (label or title or agency):
            continue
        if compact(content(text)) == "bo cong an" and not label:
            continue
        indexes = [index]
        # A Latin OCR pass can read "CỤC TRƯỞNG CỤC CẢNH SÁT" as
        # "CC TRUNG CC CÁNH SÁT". Recover that neighbouring box for the
        # Vietnamese re-reader without inventing its missing characters.
        if re.match(r"^(?:quan\s*ly|hanh\s*chinh|ve|trat\s*tu|xa\s*hoi|dkql|dlqg)\b", fold(content(text))):
            preceding = sorted((i for i, (previous, box, _) in enumerate(records) if box and i != index
                                and box[3] <= anchor_box[1] + .2 * (anchor_box[3] - anchor_box[1])
                                and _aligned(anchor_box, box)), key=lambda i: records[i][1][3], reverse=True)
            top = anchor_box
            for previous in preceding[:3]:
                previous_text, previous_box, _ = records[previous]
                height = max(top[3] - top[1], previous_box[3] - previous_box[1])
                if top[1] - previous_box[3] > 2.3 * height or boundary(previous_text):
                    break
                if not content(previous_text):
                    break
                if (authority_word_score(previous_text) < 2 or
                        not (INSTITUTION.search(fold(previous_text)) or re.search(r"canh\s*sat", fold(previous_text)))):
                    break
                indexes.insert(0, previous)
                top = previous_box
        last = anchor_box
        # Read by geometry, skipping other columns even when their lines are
        # interleaved in the detector's reading order.
        following = sorted((i for i, (_, box, _) in enumerate(records) if box is not None and i != index
                            and (box[1] + box[3]) / 2 >= (anchor_box[1] + anchor_box[3]) / 2 - .2 * (anchor_box[3] - anchor_box[1])),
                           key=lambda i: (records[i][1][1], records[i][1][0]))
        for other in following:
            next_text, next_box, _ = records[other]
            height = max(last[3] - last[1], next_box[3] - next_box[1])
            same_row = min(last[3], next_box[3]) - max(last[1], next_box[1]) >= .5 * min(last[3] - last[1], next_box[3] - next_box[1])
            nearby_fragment = same_row and 0 <= next_box[0] - last[2] <= 1.5 * height
            if not (_aligned(anchor_box, next_box) or nearby_fragment):
                continue
            if next_box[1] - last[3] > 2.3 * height:
                break
            if boundary(next_text) or AUTHORITY_LABEL.search(fold(next_text)) or LEADER_TITLE.match(fold(next_text)):
                break
            part = content(next_text)
            province = bool(indexes and re.search(r"(?:tinh|thanh\s*pho)\s*$", fold(content(texts[indexes[-1]]))))
            continuation = INSTITUTION.search(fold(part)) or AUTHORITY_CONTINUATION.search(fold(part))
            if not (continuation or province or (label and not content(text))):
                break
            indexes.append(other)
            last = next_box
            if len(indexes) >= 8:
                break
        body = tuple(i for i in indexes if content(texts[i]))
        parts = _joined([content(texts[i]) for i in body])
        if not parts or not any(INSTITUTION.search(fold(part)) for part in parts) and not label:
            continue
        bounds = _union([records[i][1] for i in body])
        # A nearby date supports the block only when it lies in this column.
        near_date = any(box and _aligned(anchor_box, box) and boundary(line)
                        and (DATE.search(line) or WORD_DATE.search(fold(line)))
                        and 0 <= anchor_box[1] - box[3] <= 5 * max(1, anchor_box[3] - anchor_box[1])
                        for line, box, _ in records)
        score = (8 if label else 6 if title else 3) + min(2, len(parts)) + int(near_date) + sum(records[i][2] for i in body) / len(body)
        candidates.append((score, AuthorityRegion(tuple(indexes), body, parts, bounds, "layout",
                                                  "label" if label else "title" if title else "institution")))
    if any(box and LEADER_TITLE.match(fold(text)) for text, box, _ in records) and not any(region.anchor == "title" for _, region in candidates):
        candidates = [(score, region) for score, region in candidates if region.anchor == "label"]
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if not candidates:
        return AuthorityRegion(method="layout")
    score, best = candidates[0]
    for other_score, other in candidates[1:]:
        if compact(other.text) == compact(best.text) or set(other.indices) & set(best.indices):
            continue
        if score - other_score < 1.5:
            return AuthorityRegion(method="layout", ambiguous=True)
    return best
