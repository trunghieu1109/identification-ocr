"""Card-side detection and labelled back-side extraction from actual OCR text."""
import re
import unicodedata
from datetime import datetime


FRONT_FIELDS = (
    ("id_number", "Số CCCD", "ID_number"),
    ("name", "Họ và tên", "Name"),
    ("date_of_birth", "Ngày sinh", "Date_of_birth"),
    ("gender", "Giới tính", "Gender"),
    ("nationality", "Quốc tịch", "Nationality"),
    ("place_of_origin", "Quê quán", "Place_of_origin"),
    ("place_of_residence", "Nơi thường trú", "Place_of_residence"),
)
BACK_FIELDS = (
    ("issue_date", "Ngày cấp", ""),
    ("issuing_authority", "Cơ quan cấp", ""),
    ("identifying_features", "Đặc điểm nhân dạng", ""),
)
ISSUE_LABEL = re.compile(
    r"ngay\s*(?:,\s*)?thang\s*(?:,\s*)?nam(?!\s*sinh)(?:\s*cap)?|ngay\s*cap|"
    r"date\s*,?\s*month\s*,?\s*year(?!\s*(?:of\s*)?birth)|date\s*(?:of\s*)?(?:issue|issuance)|issued\s*on"
)
FEATURE_LABEL = re.compile(
    r"dac\s*diem\s*nhan\s*dang|personal\s*identification(?:\s*features)?|"
    r"identifying\s*(?:features|characteristics)|distinctive\s*features"
)
AUTHORITY_LABEL = re.compile(r"noi\s*cap|co\s*quan\s*cap|issuing\s*authority|issued\s*by|place\s*of\s*issue")
INSTITUTION = re.compile(
    r"bo\s*cong\s*an|cuc\s*(?:truong|canh\s*sat)|quan\s*ly\s*hanh\s*chinh|"
    r"trat\s*tu\s*xa\s*hoi|ministry\s*of\s*public\s*security|police\s*department|"
    r"department\s*of|administrative|social\s*order|giam\s*doc\s*cong\s*an|"
    r"cong\s*an\s*(?:tinh|thanh\s*pho)|dkql\s*cu\s*tru|dlqg|du\s*lieu\s*quoc\s*gia|ve\s*dan\s*cu|\bqlhc\b|\bttxh\b|^(?:tinh|thanh\s*pho)\s"
)
FINGERPRINT = re.compile(r"van\s*tay|ngon\s*tro|left\s*index|right\s*index|fingerprint")
DATE = re.compile(r"(?<!\d)(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})(?!\d)")
WORD_DATE = re.compile(r"ngay\s*(\d{1,2})\s*thang\s*(\d{1,2})\s*nam\s*(\d{4})(?!\d)")


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
                   if unicodedata.category(c) != "Mn")


def text_rows(raw, key="text_lines"):
    rows = []
    source = raw.get(key, [])
    for row in source if isinstance(source, (list, tuple)) else []:
        text = row.get("text", "") if isinstance(row, dict) else row if isinstance(row, str) else ""
        if isinstance(text, str) and text.strip():
            rows.append(unicodedata.normalize("NFC", re.sub(r"[\t ]+", " ", text)).strip())
    return rows


def detect_side(rows):
    text = fold("\n".join(rows))
    if FEATURE_LABEL.search(text) or ISSUE_LABEL.search(text) or re.search(r"i[d<]\s*vnm", text):
        return "back"
    front_markers = (
        r"ho\s*(?:va\s*)?ten|full\s*name", r"ngay\s*sinh|date\s*of\s*birth",
        r"gioi\s*tinh|\bsex\b", r"que\s*quan|place\s*of\s*origin",
    )
    if sum(bool(re.search(pattern, text)) for pattern in front_markers) >= 2:
        return "front"
    if INSTITUTION.search(text) and (WORD_DATE.search(text) or DATE.search(text)):
        return "back"
    return "unknown"


def _is_mrz(text):
    compact = re.sub(r"\s+", "", text.upper())
    return bool(20 <= len(compact) <= 90 and re.fullmatch(r"[A-Z0-9<]+", compact)
                and ("<<" in compact or re.match(r"I[D<]VNM", compact)))


def _boundary(text):
    value = fold(text)
    return (ISSUE_LABEL.search(value) or FEATURE_LABEL.search(value) or AUTHORITY_LABEL.search(value)
            or INSTITUTION.search(value) or FINGERPRINT.search(value) or WORD_DATE.search(value) or _is_mrz(text))


def _labelled_value(rows, pattern, limit=3):
    for index, row in enumerate(rows):
        matches = list(pattern.finditer(fold(row)))
        if not matches:
            continue
        # Vietnamese and English labels may both occur on the same OCR line.
        tail = row[matches[-1].end():].strip(" :;/,-")
        parts = [tail] if tail else []
        for following in rows[index + 1:index + 1 + limit]:
            if _boundary(following) or DATE.fullmatch(following.strip()):
                break
            parts.append(following)
        return " ".join(parts).strip()
    return ""


def _date_value(text):
    match = DATE.search(text) or WORD_DATE.search(fold(text))
    if not match:
        return ""
    try:
        value = datetime(int(match[3]), int(match[2]), int(match[1]))
    except ValueError:
        return ""
    return value.strftime("%d/%m/%Y")


def _issue_date(rows):
    for index, row in enumerate(rows):
        if ISSUE_LABEL.search(fold(row)):
            for candidate in rows[index:index + 3]:
                date = _date_value(candidate)
                if date:
                    return date
                if candidate != row and (FEATURE_LABEL.search(fold(candidate)) or _is_mrz(candidate)):
                    break
    # Some backs print the date alone, adjacent to the issuing authority.
    if detect_side(rows) == "back":
        dates = {_date_value(row) for row in rows
                 if DATE.fullmatch(row.strip()) or WORD_DATE.fullmatch(fold(row).strip())
                 or re.fullmatch(r"ngay\s*\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{4}", fold(row))}
        dates.discard("")
        if len(dates) == 1:
            return dates.pop()
    return ""


def _authority(rows):
    labelled = _labelled_value(rows, AUTHORITY_LABEL, limit=2)
    if labelled:
        return labelled
    for index, row in enumerate(rows):
        if INSTITUTION.search(fold(row)):
            parts = [row]
            for following in rows[index + 1:index + 3]:
                if INSTITUTION.search(fold(following)) and not ISSUE_LABEL.search(fold(following)):
                    parts.append(following)
                else:
                    break
            return " ".join(parts)
    return ""


def parse_back(rows, backup_rows=()):
    return {
        "issue_date": _issue_date(rows) or _issue_date(backup_rows),
        "issuing_authority": _authority(rows) or _authority(backup_rows),
        "identifying_features": _labelled_value(rows, FEATURE_LABEL) or _labelled_value(backup_rows, FEATURE_LABEL),
    }
