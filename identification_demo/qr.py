"""Read QR pixels locally and parse the seven-part CCCD text format."""
import re
import unicodedata
from datetime import datetime


QR_FIELDS = (
    ("id_number", "Số căn cước"),
    ("old_id_number", "Số CMND cũ"),
    ("name", "Họ và tên"),
    ("date_of_birth", "Ngày sinh"),
    ("gender", "Giới tính"),
    ("place_of_residence", "Nơi cư trú"),
    ("issue_date", "Ngày cấp"),
)


class QRUnavailable(RuntimeError):
    pass


class QRBusy(RuntimeError):
    pass


def _date(value):
    if not re.fullmatch(r"[0-9]{8}|[0-9]{2}/[0-9]{2}/[0-9]{4}", value):
        return ""
    try:
        return datetime.strptime(value.replace("/", ""), "%d%m%Y").strftime("%d/%m/%Y")
    except ValueError:
        return ""


def parse_cccd_qr(text):
    # Keep empty components: the old CMND number is optional. Unknown layouts
    # are rejected instead of assigning guessed fields.
    parts = [unicodedata.normalize("NFC", part.strip())
             for part in text.lstrip("\ufeff").strip().split("|")]
    if len(parts) != 7 or not re.fullmatch(r"[0-9]{12}", parts[0]):
        return None
    if parts[1] and not re.fullmatch(r"[0-9]{9}|[0-9]{12}", parts[1]):
        return None
    fields = dict(zip((key for key, _ in QR_FIELDS), parts))
    fields["date_of_birth"] = _date(parts[3])
    fields["issue_date"] = _date(parts[6])
    fields["gender"] = {"nam": "Nam", "nữ": "Nữ", "khác": "Khác"}.get(parts[4].casefold(), "")
    return fields
