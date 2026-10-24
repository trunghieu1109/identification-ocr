import pytest

from identification_demo.documents import detect_side, parse_back


@pytest.mark.parametrize("printed,expected", [
    ("Ngày cấp / Date of issue: 20/04/2021", "20/04/2021"),
    ("Ngày, tháng, năm / Date, month, year: 7-3-2020", "07/03/2020"),
    ("Ngày 15 tháng 11 năm 2018", "15/11/2018"),
    ("Ngày 16/04/2021", "16/04/2021"),
    ("Ngày cấp: 31/02/2021", ""),
])
def test_issue_dates_on_ordinary_cccd_backs(printed, expected):
    rows = ["Đặc điểm nhân dạng", "Sẹo nhỏ dưới lông mày trái", printed,
            "CỤC TRƯỞNG CỤC CẢNH SÁT", "QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"]
    assert parse_back(rows)["issue_date"] == expected


def test_multiline_authority_excludes_the_signer_and_fingerprint_labels():
    fields = parse_back([
        "Đặc điểm nhân dạng / Personal identification:",
        "Nốt ruồi 0,2 cm dưới đuôi lông mày trái",
        "Ngày, tháng, năm / Date, month, year", "20/04/2021",
        "CỤC TRƯỞNG CỤC CẢNH SÁT",
        "QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI",
        "NGUYỄN VĂN MẪU", "Ngón trỏ trái / Left index",
    ])
    assert fields["issue_date"] == "20/04/2021"
    assert fields["issuing_authority"] == "CỤC TRƯỞNG CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"
    assert fields["identifying_features"] == "Nốt ruồi 0,2 cm dưới đuôi lông mày trái"


@pytest.mark.parametrize("rows,expected", [
    (["CỤC TRƯỞNG", "CỤC CẢNH SÁT ĐKQL CƯ TRÚ", "VÀ DLQG VỀ DÂN CƯ", "Người ký"],
     "CỤC TRƯỞNG CỤC CẢNH SÁT ĐKQL CƯ TRÚ VÀ DLQG VỀ DÂN CƯ"),
    (["Nơi cấp / Place of issue: Công an tỉnh Hà Nam"], "Công an tỉnh Hà Nam"),
    (["Cơ quan cấp:", "CỤC CẢNH SÁT QLHC VỀ TTXH"], "CỤC CẢNH SÁT QLHC VỀ TTXH"),
])
def test_authority_variants_are_read_from_text(rows, expected):
    assert parse_back(rows)["issuing_authority"] == expected


def test_date_and_authority_are_not_inferred_when_missing():
    assert parse_back(["MỘT DÒNG CHỮ BẤT KỲ"])["issuing_authority"] == ""
    fields = parse_back(["Họ và tên", "NGUYỄN VĂN MẪU", "Ngày sinh: 01/01/2000", "Giới tính: Nam"])
    assert fields["issue_date"] == ""
    assert fields["issuing_authority"] == ""


def test_word_date_is_not_appended_to_identifying_features():
    fields = parse_back(["Đặc điểm nhân dạng:", "Nốt ruồi má trái", "Ngày 14 tháng 8 năm 2019", "CỤC TRƯỞNG CỤC CẢNH SÁT"])
    assert fields["identifying_features"] == "Nốt ruồi má trái"
    assert fields["issue_date"] == "14/08/2019"


@pytest.mark.parametrize("rows,side", [
    (["Đặc điểm nhân dạng", "Ngày cấp: 20/04/2021"], "back"),
    (["Ngày, tháng, năm / Date, month, year", "20/04/2021"], "back"),
    (["CỤC TRƯỞNG CỤC CẢNH SÁT", "Ngày 15 tháng 11 năm 2018"], "back"),
    (["CỤC TRƯỞNG CỤC CẢNH SÁT", "15/11/2018"], "back"),
    (["Họ và tên", "Ngày, tháng, năm sinh: 01/01/2000", "Giới tính: Nam"], "front"),
    (["Full name", "Date, month, year of birth: 01/01/2000", "Sex: Male"], "front"),
    (["Some unrelated text"], "unknown"),
])
def test_side_detection_does_not_confuse_birth_dates_with_issue_dates(rows, side):
    assert detect_side(rows) == side


def test_paddle_text_can_recover_an_unreadable_vietocr_field():
    fields = parse_back(["đọc chưa rõ"], ["Ngày cấp: 20/04/2021", "Nơi cấp: Công an tỉnh Hà Nam"])
    assert fields["issue_date"] == "20/04/2021"
    assert fields["issuing_authority"] == "Công an tỉnh Hà Nam"
