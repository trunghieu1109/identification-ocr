"""Conservative Vietnam TD1 adapter. Never repair digits to make checks pass."""
import re
from datetime import date


def check_digit(text):
    values = (int(c) if c.isdigit() else ord(c) - 55 if "A" <= c <= "Z" else 0 for c in text)
    return str(sum(value * (7, 3, 1)[i % 3] for i, value in enumerate(values)) % 10)


def _valid(text, digit):
    return digit.isdigit() and check_digit(text) == digit


def _date(text, year):
    try:
        return date(year, int(text[2:4]), int(text[4:6]))
    except (ValueError, TypeError):
        return None


def mrz_rank(result):
    return ({"verified": 5, "partial": 4, "invalid": 3, "unsupported": 2, "not_found": 1, "error": 0}.get(result["status"], 0),
            sum(bool(value) for value in result.get("checks", {}).values()), len(result.get("fields", {})))


def usable_mrz_fields(result):
    if result["status"] == "verified":
        return result["fields"]
    checks = result.get("checks", {})
    if all(checks.get(key) for key in ("id_number", "document_number", "composite")):
        return {key: value for key, value in result["fields"].items()
                if result.get("field_checks", {}).get(key) == "checksum"}
    return {}


def parse_vietnam_td1(raw_text, today=None):
    today = today or date.today()
    lines = [re.sub(r"\s", "", line.upper()) for line in raw_text.splitlines() if line.strip()]
    result = {"status": "not_found" if not lines else "invalid", "format": "TD1",
              "raw_text": raw_text, "fields": {}, "checks": {}, "field_checks": {}, "warnings": []}
    # Exact lengths preserve column positions. No padding, truncation or O→0.
    if len(lines) != 3 or any(len(line) != 30 or not re.fullmatch(r"[A-Z0-9<]{30}", line) for line in lines):
        return result
    first, second, names = lines
    if first[:5] not in {"IDVNM", "I<VNM"} or second[15:18] != "VNM":
        result["status"] = "unsupported"
        return result
    checks = result["checks"]
    checks["document_number"] = _valid(first[5:14], first[14])
    checks["date_of_birth"] = bool(re.fullmatch(r"[0-9]{6}", second[:6])) and _valid(second[:6], second[6])
    checks["expiry_date"] = bool(re.fullmatch(r"[0-9]{6}", second[8:14])) and _valid(second[8:14], second[14])
    checks["composite"] = _valid(first[5:] + second[:7] + second[8:15] + second[18:29], second[29])
    # Vietnam's 12-digit ID appears in optional data, separate from the
    # nine-character ICAO document number. Do not label the latter as old CMND.
    number = first[15:27]
    checks["id_number"] = (bool(re.fullmatch(r"[0-9]{12}", number)) and first[27:29] == "<<"
                           and _valid(number, first[29]))
    fields = result["fields"]
    if checks["id_number"] and checks["composite"]:
        fields["id_number"] = number
        result["field_checks"]["id_number"] = "checksum"
    # Resolve YY without a fixed pivot: retain ambiguity rather than changing
    # 1920 into 2020. Cross-source reconciliation can supply a full OCR/QR date.
    if checks["date_of_birth"] and checks["composite"]:
        births = [value for century in (1800, 1900, 2000)
                  if (value := _date(second[:6], century + int(second[:2]))) is not None
                  and 0 <= (today - value).days <= 130 * 366]
        if len(births) == 1:
            fields["date_of_birth"] = births[0].strftime("%d/%m/%Y")
            result["field_checks"]["date_of_birth"] = "checksum"
        else:
            result["birth_date_yymmdd"] = second[:6]
            if births:
                result["warnings"].append("MRZ chỉ ghi hai chữ số năm sinh; cần đối chiếu thế kỷ với QR hoặc chữ in.")
            else:
                checks["date_of_birth"] = False
    if checks["expiry_date"] and checks["composite"]:
        # No expiry century is assumed; use a bounded validity interval.
        expiries = [value for century in (1900, 2000, 2100)
                    if (value := _date(second[8:14], century + int(second[8:10]))) is not None
                    and today.year - 30 <= value.year <= today.year + 50]
        if len(expiries) == 1:
            fields["expiry_date"] = expiries[0].strftime("%d/%m/%Y")
            result["field_checks"]["expiry_date"] = "checksum"
        else:
            checks["expiry_date"] = False
    if checks["composite"]:
        # These positions are NOT protected by the composite checksum.
        if second[7] in "MF":
            fields["gender"] = {"M": "Nam", "F": "Nữ"}[second[7]]
        fields["nationality"] = "Việt Nam"
        if re.fullmatch(r"[A-Z]+(?:<[A-Z]+)*<<[A-Z]+(?:<[A-Z]+)*<*", names):
            fields["name"] = " ".join(names.replace("<", " ").split())
        for key in ("gender", "nationality", "name"):
            if key in fields:
                result["field_checks"][key] = "not_covered_by_checksum"
    result["status"] = "verified" if all(checks.values()) else "partial" if fields else "invalid"
    if result["status"] != "verified":
        result["warnings"].append("MRZ chưa vượt qua toàn bộ kiểm tra; chỉ dùng trường có checksum riêng hợp lệ khi số căn cước, số tài liệu và checksum tổng hợp đều khớp.")
    if fields.get("name"):
        result["warnings"].append("Tên MRZ không có dấu và không được checksum bảo vệ; cần đối chiếu thẻ.")
    return result
