#!/usr/bin/env python3
"""Create clearly synthetic two-sided cards for an OCR timing smoke test."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(args.font), 32)
    title = ImageFont.truetype(str(args.font), 38)
    for side in ("front", "back"):
        for expiry in (False, True):
            image = Image.new("RGB", (1200, 760), "#f5f8f2")
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((6, 6, 1194, 754), radius=26, outline="#597970", width=4)
            draw.text((50, 36), "CĂN CƯỚC - DỮ LIỆU GIẢ LẬP", fill="#243b35", font=title)
            lines = (["Số / No.: 001099000001", "Họ và tên / Full name: NGUYỄN VĂN THỬ",
                      "Ngày sinh / Date of birth: 01/02/1999", "Giới tính / Sex: Nam",
                      "Quốc tịch / Nationality: Việt Nam", "Quê quán / Place of origin: Hà Nội",
                      "Nơi thường trú / Place of residence: Hà Nội"] if side == "front" else
                     ["Đặc điểm nhân dạng: Nốt ruồi má trái", "Ngày cấp / Date of issue: 15/03/2021",
                      "Cơ quan cấp: CÔNG AN TỈNH HÀ NAM"])
            if expiry:
                lines.append("Có giá trị đến / Date of expiry:")
            for index, text in enumerate(lines):
                draw.text((50, 118 + index * 70), text, fill="#151b18", font=font)
            if expiry:
                y = 118 + (len(lines) - 1) * 70
                draw.text((50, y + 52), "25/10/2036", fill="#151b18", font=font)
            image.save(args.output / f"{side}-expiry-{int(expiry)}.png")
    expected = {"front": {"id_number": "001099000001", "name": "NGUYỄN VĂN THỬ", "date_of_birth": "01/02/1999",
                          "gender": "Nam", "nationality": "Việt Nam", "place_of_origin": "Hà Nội", "place_of_residence": "Hà Nội"},
                "back": {"issue_date": "15/03/2021", "issuing_authority": "CÔNG AN TỈNH HÀ NAM"}}
    manifest = {"synthetic": True, "pairs": [
        {"name": "expiry-front", "front": "front-expiry-1.png", "back": "back-expiry-0.png", "expected": expected},
        {"name": "expiry-back", "front": "front-expiry-0.png", "back": "back-expiry-1.png", "expected": expected},
    ]}
    (args.output / "dataset.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(args.output / "dataset.json")


if __name__ == "__main__":
    main()
