"""Regression against the actual upstream parser, without loading model weights."""
import pytest

pytest.importorskip("cv2")
pytest.importorskip("vietocr")
from identification_demo.engine import NamLhnEngine

if not NamLhnEngine().source_available:
    pytest.skip("Upstream checkout is not configured", allow_module_level=True)

from ocr.parser import parse_idcard_fields


@pytest.mark.parametrize("label", [
    "Quốc tịch / Nationality: Việt Nam",
    "Nationality: Việt Nam",
    "Quốc tịch: Việt Nam",
])
def test_nationality_label_does_not_consume_the_following_address(label):
    box = [[0, 0], [100, 0], [100, 20], [0, 20]]
    result = parse_idcard_fields([
        [label, box],
        ["Quê quán / Place of origin: Hà Nội", box],
        ["Nơi thường trú: 123 Đường Kiểm Thử", box],
    ])
    assert result["Nationality"] == "Việt Nam"
