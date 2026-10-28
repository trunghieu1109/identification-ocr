import unicodedata

import pytest

from identification_demo.qr import parse_cccd_qr


# Synthetic data only; leading zeroes and Vietnamese UTF-8 must survive decoding.
PAYLOAD = "001099000001|012345678|NGUYỄN VĂN THỬ|01021999|Nam|Phường Thử, Hà Nội|15032021"


def test_cccd_fields_preserve_identifiers_accents_and_format_dates():
    fields = parse_cccd_qr(PAYLOAD)
    assert fields == {
        "id_number": "001099000001", "old_id_number": "012345678",
        "name": "NGUYỄN VĂN THỬ", "date_of_birth": "01/02/1999",
        "gender": "Nam", "place_of_residence": "Phường Thử, Hà Nội",
        "issue_date": "15/03/2021",
    }


def test_optional_cmnd_and_unicode_normalization_do_not_shift_fields():
    text = unicodedata.normalize("NFD", PAYLOAD.replace("012345678", "").replace("Nam", "Nữ"))
    fields = parse_cccd_qr("\ufeff  " + text + "\n")
    assert fields["old_id_number"] == ""
    assert fields["name"] == "NGUYỄN VĂN THỬ"
    assert fields["gender"] == "Nữ"
    assert fields["issue_date"] == "15/03/2021"


@pytest.mark.parametrize("text", [
    "https://example.com", "", "001099000001|TEST",
    PAYLOAD.replace("001099000001", "1099000001"),
    PAYLOAD.replace("012345678", "invalid"),
    PAYLOAD + "|unknown extension",
])
def test_unknown_layouts_are_not_guessed(text):
    assert parse_cccd_qr(text) is None


def test_known_layout_with_missing_name_preserves_other_qr_fields():
    fields = parse_cccd_qr(PAYLOAD.replace("NGUYỄN VĂN THỬ", ""))
    assert fields["name"] == ""
    assert fields["id_number"] == "001099000001"
    assert fields["issue_date"] == "15/03/2021"


@pytest.mark.parametrize("value,expected", [
    ("29022000", "29/02/2000"), ("01/02/1999", "01/02/1999"),
    ("29021999", ""), ("31022000", ""), ("010120", ""), ("", ""),
])
def test_dates_are_validated(value, expected):
    assert parse_cccd_qr(PAYLOAD.replace("01021999", value))["date_of_birth"] == expected
