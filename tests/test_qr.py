import importlib
import types
import unicodedata

import pytest
from PIL import Image

from identification_demo.qr import QRBusy, QRReader, QRUnavailable, parse_cccd_qr


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


def test_reader_passes_pixels_to_qr_decoder_and_preserves_utf8(monkeypatch):
    calls = []

    def decode(image, **options):
        calls.append((image.mode, image.size, options))
        # The decoder's guessed text encoding must not corrupt valid UTF-8 bytes.
        return [types.SimpleNamespace(bytes=PAYLOAD.encode(), text="wrong encoding")] * 2

    module = types.SimpleNamespace(BarcodeFormat=types.SimpleNamespace(QRCode="QR"), read_barcodes=decode)
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    reader = QRReader()
    with Image.new("RGB", (200, 100)) as image:
        assert reader.read(image) == [PAYLOAD]
    assert calls == [("RGB", (200, 100), {"formats": "QR", "try_rotate": True, "try_downscale": True})]
    assert not reader._lock.locked()


def test_missing_decoder_is_actionable_and_can_be_retried(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    reader = QRReader()
    for _ in range(2):
        with pytest.raises(QRUnavailable, match="requirements.txt"):
            reader.read(None)
        assert not reader._lock.locked()


def test_parallel_decodes_are_rejected():
    reader = QRReader()
    reader._lock.acquire()
    try:
        with pytest.raises(QRBusy):
            reader.read(None)
    finally:
        reader._lock.release()
