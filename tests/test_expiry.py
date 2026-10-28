"""Expiry parser and main API regressions, without model weights."""
import pytest
from fastapi.testclient import TestClient

from identification_demo.expiry import parse_expiry
from identification_demo.pipeline import CARD_FIELDS
from identification_demo.web import create_app
from test_pipeline import Engine, Reader, image_data


@pytest.mark.parametrize("rows,expected", [
    (["Có giá trị đến: 5/2/2036"], "05/02/2036"),
    (["Ngày hết hạn / Date of expiry", "25-10-2036"], "25/10/2036"),
    (["Có giá trị", "đến / Date of expiry:", "25.10.2036"], "25/10/2036"),
    (["VALID UNTIL: 2036-10-25"], "25/10/2036"),
    (["Date of expiration: 29/02/2024"], "29/02/2024"),
    (["Hạn sử dụng: ngày 25 tháng 10 năm 2036"], "25/10/2036"),
    (["Ngày sinh: 25/10/1990 Có giá trị đến: 25/10/2036 Ngày cấp: 25/10/2026"], "25/10/2036"),
])
def test_labelled_dates_are_normalized_and_keep_evidence(rows, expected):
    found = parse_expiry(rows)
    assert found and {item["expiry_date"] for item in found} == {expected}
    assert all(item["expiry_type"] == "date" and item["evidence"] for item in found)


@pytest.mark.parametrize("rows", [
    ["Ngày sinh: 25/10/1990", "Ngày cấp: 25/10/2026"],
    ["25/10/2036"],
    ["Có giá trị đến", "Ngày sinh: 25/10/1990"],
    ["Có giá trị đến", "Date of issue", "25/10/2026"],
    ["Có giá trị đến", "Ngày, tháng, năm", "25/10/2026"],
    ["Có giá trị đến: 31/02/2036"],
    ["Date of expiry: 2036-13-25"],
    ["Số CCCD: 001099000001"],
])
def test_birth_issue_unlabelled_and_invalid_dates_are_not_expiry(rows):
    assert parse_expiry(rows) == []


@pytest.mark.parametrize("text", ["Có giá trị đến: Không thời hạn", "KHÔNG THỜI HẠN", "Valid until: No expiry"])
def test_explicit_unlimited_is_distinct_from_missing(text):
    found = parse_expiry([text])
    assert len(found) == 1 and found[0]["expiry_type"] == "unlimited" and found[0]["expiry_date"] == ""


def test_main_api_also_returns_expiry():
    client = TestClient(create_app(Engine(), Reader()))
    result = client.post("/api/cccd/extract", files={
        "front_image": ("front.png", image_data(), "image/png"),
        "back_image": ("back.png", image_data("back"), "image/png"),
    }).json()
    assert set(result["fields"]) == set(CARD_FIELDS) and len(result["fields"]) == 12
    assert result["fields"]["expiry_date"] == "25/10/2036" and result["total_fields"] == 12
