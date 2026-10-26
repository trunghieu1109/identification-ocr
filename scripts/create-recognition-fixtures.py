#!/usr/bin/env python3
"""Generate labelled synthetic Vietnamese line crops with a separate holdout split."""
import argparse
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


DEVELOPMENT = [
    "CĂN CƯỚC CÔNG DÂN", "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc",
    "Số / No.: 001099000001", "Họ và tên / Full name: NGUYỄN VĂN THỬ", "NGUYỄN VĂN THỬ", "HOÀNG THÀNH VINH",
    "ĐẶNG THỊ THUỲ DƯƠNG", "TRẦN NGỌC ÁNH", "ĐỖ THỊ HỒNG NHUNG", "PHẠM HỮU PHƯỚC",
    "Ngày sinh / Date of birth: 01/02/1999", "Giới tính / Sex: Nam", "Giới tính / Sex: Nữ",
    "Quốc tịch / Nationality: Việt Nam", "Quê quán / Place of origin: Hà Nội",
    "Nơi thường trú / Place of residence: Phường Minh Khai, Quận Bắc Từ Liêm, Thành phố Hà Nội",
    "Tổ 12, Phường Trưng Vương, Thành phố Việt Trì, Tỉnh Phú Thọ",
    "Số 128/45 Nguyễn Đình Chiểu, Phường Đa Kao, Quận 1, Thành phố Hồ Chí Minh",
    "Xã Đức Hoà Thượng, Huyện Đức Hoà, Tỉnh Long An", "Phường Tân Phú, Thành phố Thủ Đức, Thành phố Hồ Chí Minh",
    "Có giá trị đến / Date of expiry:", "25/10/2036", "Không thời hạn", "Ngày cấp / Date of issue: 15/03/2021",
    "CỤC TRƯỞNG CỤC CẢNH SÁT", "QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", "CÔNG AN TỈNH HÀ NAM",
    "001099000007", "079203001256", "31/12/2000", "15/03/2021",
]
HOLDOUT = [
    "LÊ THỊ QUỲNH NHƯ", "VÕ HOÀNG PHÚC", "BÙI ĐỨC THẮNG", "HUỲNH TẤN TÀI", "TRƯƠNG NGỌC DIỄM",
    "ĐINH THỊ MỸ LINH", "ĐẶNG HỮU THUẬN", "NGUYỄN THỊ THUỶ TIÊN", "Nguyễn Văn Hưởng", "Trần Thị Bích Ngọc",
    "Nữ", "Nam", "Quốc tịch: Việt Nam", "Ngày sinh: 29/02/2000", "Có giá trị đến: 31/12/2040",
    "Ngày cấp: 07/08/2023", "079203001256", "012345678901",
    "Thôn Đông, Xã Vĩnh Thịnh, Huyện Vĩnh Tường, Tỉnh Vĩnh Phúc",
    "Số 45/2 Đường Lý Thường Kiệt, Phường Hoà Thuận Đông, Quận Hải Châu, Thành phố Đà Nẵng",
    "Khu phố Bình Phước A, Phường Bình Chuẩn, Thành phố Thuận An, Tỉnh Bình Dương",
    "Phường Nguyễn Trãi, Quận Hà Đông, Thành phố Hà Nội",
    "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", "CÔNG AN THÀNH PHỐ HỒ CHÍ MINH",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--font-root", type=Path, default=Path("/usr/share/fonts/truetype"))
    args = parser.parse_args()
    if (args.output / "dataset.json").exists():
        parser.error("Use a new output directory to preserve the previous dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    held_out = args.split == "holdout"
    fonts = [("liberation/LiberationSans-Regular.ttf", 30), ("ubuntu/Ubuntu-R.ttf", 26)] if held_out else [
        ("dejavu/DejaVuSans.ttf", 30), ("dejavu/DejaVuSans.ttf", 22)]
    rows = []
    for index, text in enumerate(HOLDOUT if held_out else DEVELOPMENT):
        for font_index, (font_name, size) in enumerate(fonts):
            font_path = args.font_root / font_name
            if not font_path.is_file():
                parser.error(f"Missing fixture font: {font_path}")
            font = ImageFont.truetype(str(font_path), size)
            variants = ("clear", "degraded") if held_out else ("clear" if font_index == 0 else "small-faded",)
            for variant in variants:
                box = font.getbbox(text)
                image = Image.new("RGB", (box[2] - box[0] + 12, box[3] - box[1] + 12),
                                  (240, 244, 236) if held_out else "white")
                ImageDraw.Draw(image).text((6 - box[0], 6 - box[1]), text, font=font, fill=(25, 25, 25))
                if variant != "clear":
                    image = ImageEnhance.Contrast(image).enhance(.6 if held_out else .65)
                    image = image.filter(ImageFilter.GaussianBlur(.5 if held_out else .35))
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=65 if held_out else 72)
                    with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
                        image = decoded.convert("RGB")
                name = f"{index:02d}-{font_index}-{variant}.png" if held_out else f"{index:02d}-{font_index}.png"
                image.save(args.output / name)
                image.close()
                rows.append({"name": name, "image": name, "text": text, "variant": variant, "font": font_path.name})
    (args.output / "dataset.json").write_text(json.dumps({"synthetic": True, "split": args.split, "samples": rows},
                                                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output / 'dataset.json'}: {len(rows)} synthetic crops")


if __name__ == "__main__":
    main()
