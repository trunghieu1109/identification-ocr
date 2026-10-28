#!/usr/bin/env python3
"""Export a front-card image as exact namlhn GPU recognition crops and OCR text."""
import argparse
import hashlib
import html
import json
import math
import os
import subprocess
import sys
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def normalized_text(value):
    return unicodedata.normalize("NFC", str(value or "")).strip()


def close_readings(readings):
    for reading in readings:
        for image in reading["images"]:
            image.close()


def write_export(image, readings, raw, output, source, model, enhancement, artifacts=None):
    """Save actual input pixels, with the chosen reading paired to each crop."""
    from PIL import ImageDraw

    output = Path(output)
    (output / "crops").mkdir(parents=True)
    image.save(output / "source.png")
    for name, artifact in (artifacts or {}).items():
        if name not in {"card", "prepared_card"}:
            raise ValueError("Unknown preprocessing artifact")
        artifact.save(output / f"{name}.png")
    rows = {}
    for reading in readings:
        for position, index in enumerate(reading["indexes"]):
            row = rows.setdefault(index, {"number": index + 1, "readings": {}})
            row["box"] = reading["regions"][position]["box"]
            row["readings"][reading["stage"]] = {
                "text": reading["texts"][position], "confidence": reading["scores"][position],
                "selected": reading["selected"][position], "_image": reading["images"][position],
            }

    line_number = 0
    for row in rows.values():
        stem = f'{row["number"]:04d}'
        selected = next((item for item in row["readings"].values() if item["selected"]), None)
        if selected is None:
            raise ValueError(f"Crop {stem} has no selected OCR reading")
        row["text"] = normalized_text(selected["text"])
        row["confidence"] = selected["confidence"]
        row["selected_reading"] = next(stage for stage, item in row["readings"].items() if item["selected"])
        row["crop"] = f"crops/{stem}.png"
        row["text_file"] = f"crops/{stem}.txt"
        selected["_image"].save(output / row["crop"])
        (output / row["text_file"]).write_text(row["text"] + "\n", encoding="utf-8")
        if row["text"]:
            line_number += 1
        row["line_number"] = line_number if row["text"] else None
        for stage, item in row["readings"].items():
            patch = item.pop("_image")
            item["crop"] = row["crop"]
            if len(row["readings"]) > 1:
                (output / "attempts").mkdir(exist_ok=True)
                item["crop"] = f"attempts/{stem}.{stage}.png"
                patch.save(output / item["crop"])

    rows = list(rows.values())
    expected = [normalized_text(line["text"]) for line in raw.get("text_lines", [])]
    if [row["text"] for row in rows if row["text"]] != expected:
        raise ValueError("Captured crops do not match the pipeline's final OCR lines")
    with image.copy() as annotated:
        draw = ImageDraw.Draw(annotated)
        width = max(2, min(image.size) // 250)
        for row in rows:
            box = [tuple(point) for point in row["box"]]
            color = "#18794e" if row["text"] else "#c27800"
            draw.line(box + [box[0]], fill=color, width=width)
            x, y = min(p[0] for p in box), max(0, min(p[1] for p in box) - 13)
            label = str(row["number"])
            draw.rectangle((x, y, x + len(label) * 8 + 6, y + 13), fill=color)
            draw.text((x + 3, y), label, fill="white")
        annotated.save(output / "boxes.png")

    report = clean_json({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source, "model": model, "pipeline": raw.get("strategy"),
        "side": "front", "detected_side": raw.get("document_side"), "image_size": list(image.size),
        "text_enhancement": enhancement, "crop_count": len(rows), "line_count": line_number,
        "preprocessing_images": {name: f"{name}.png" for name in (artifacts or {})},
        "crop_description": "Exact RGB crops passed to the line recognizer, before tensor resize/normalization; "
                            "includes perspective correction/padding and any selected enhancement.",
        "rows": rows, "ocr_result": raw,
    })
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = "\n".join(row["text"] for row in rows if row["text"])
    (output / "text.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
    write_html(report, output)
    with zipfile.ZipFile(output / "export.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != output / "export.zip":
                archive.write(path, path.relative_to(output))
    return report


def write_html(report, output):
    escape = html.escape
    cards = []
    for row in report["rows"]:
        attempts = []
        if len(row["readings"]) > 1:
            for stage, reading in row["readings"].items():
                label = "Lần đầu" if stage == "primary" else "Sau tăng cường ảnh"
                if reading["selected"]:
                    label += " · được chọn"
                attempts.append(f'<figure><figcaption>{label}</figcaption><img src="{reading["crop"]}" alt="{label}">'
                                f'<pre>{escape(str(reading["text"] or ""))}</pre></figure>')
        text = escape(row["text"]) if row["text"] else "(Không đọc được chữ)"
        score = row["confidence"]
        score_label = f"{score:.4f}" if score is not None else "Không có"
        ui_line = str(row["line_number"]) if row["line_number"] else "Bị bỏ qua vì trống"
        cards.append(f'<article><h2>Vùng {row["number"]:04d}</h2><p>Dòng trên UI: {ui_line} · '
                     f'Độ tin cậy: {score_label} · Lượt chọn: {row["selected_reading"]}</p>'
                     f'<img src="{row["crop"]}" alt="Ảnh crop vùng {row["number"]}"><pre>{text}</pre>'
                     f'<a href="{row["crop"]}">Ảnh PNG</a> · <a href="{row["text_file"]}">Text UTF-8</a>'
                     + (f'<details><summary>Các lượt nhận dạng</summary>{"".join(attempts)}</details>' if attempts else "")
                     + '</article>')
    body = "\n".join(cards) or "<p>Không phát hiện được vùng chữ nào có thể cắt để nhận dạng.</p>"
    previews = "".join(f'<figure><figcaption>{label}</figcaption><img src="{filename}" alt="{label}"></figure>'
                       for name, filename in report.get("preprocessing_images", {}).items()
                       for label in ["Toàn ảnh trước hiệu chỉnh" if name == "card" else "Toàn ảnh sau hiệu chỉnh ánh sáng và nhiễu"])
    page = f'''<!doctype html>
<html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ảnh crop và chữ OCR mặt trước</title>
<style>body{{font:16px system-ui,sans-serif;margin:24px auto;padding:0 20px;max-width:1100px;background:#f7f8fa;color:#172b4d}}
article{{background:white;border:1px solid #dce2eb;border-radius:10px;padding:18px;margin:16px 0}}
img{{max-width:100%;height:auto;border:1px solid #dce2eb}}.overview{{max-height:650px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:17px system-ui,sans-serif;padding:12px;background:#f0f3f8}}
p,figcaption{{color:#465570}}a{{color:#175cd3}}details{{margin-top:16px}}figure{{margin:16px 0}}</style>
<h1>Ảnh crop và chữ OCR mặt trước</h1>
<p>Model: {escape(report["model"])} · {report["crop_count"]} vùng · {report["line_count"]} dòng có chữ.</p>
<p>Ảnh crop giữ đúng pixel đưa vào bộ đọc chữ, gồm căn chỉnh và phần đệm; ở trước bước resize/chuẩn hóa tensor.
Vùng được đọc lại có cả ảnh và kết quả từng lượt. Độ tin cậy là điểm model trả về.</p>
<p><a href="export.zip">Tải toàn bộ ZIP</a> · <a href="result.json">JSON</a> · <a href="text.txt">Toàn bộ text</a></p>
<img class="overview" src="boxes.png" alt="Ảnh gốc với vùng chữ được đánh số">
{previews}
{body}</html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def run_local(args, output):
    sys.path.insert(0, str(ROOT))
    if args.model:
        os.environ["NAMLHN_GPU_MODEL"] = args.model
    if args.card_enhancement:
        os.environ["NAMLHN_GPU_CARD_ENHANCEMENT"] = args.card_enhancement
    from identification_demo.namlhn_docu_pipeline import extract_pixels
    from identification_demo.namlhn_gpu_worker import NamLhnDocUCudaEngine
    from identification_demo.processing import load_image

    data = args.image.read_bytes()
    readings = []
    artifacts = {}
    engine = NamLhnDocUCudaEngine()
    enhancement = args.enhancement or engine.text_enhancement
    try:
        with load_image(data) as image:
            def run():
                engine._initialize()
                return extract_pixels(engine._detector, engine._recognizer, image, "front",
                                      text_enhancement=enhancement, readings=readings,
                                      card_enhancement=engine.card_enhancement, artifacts=artifacts)

            raw, timings = engine._queue.run(run)
            raw.setdefault("timings_ms", {}).update(timings)
            if raw.get("document_side") == "back":
                raise ValueError("Ảnh được nhận diện là mặt sau; hãy chọn ảnh mặt trước.")
            report = write_export(image, readings, raw, output,
                                  {"name": args.image.name, "sha256": hashlib.sha256(data).hexdigest()},
                                  engine.model, enhancement, artifacts=artifacts)
            print(f'Đã xuất {report["crop_count"]} vùng, {report["line_count"]} dòng có chữ.')
    finally:
        close_readings(readings)
        for artifact in artifacts.values():
            artifact.close()
        engine.close()


def run_docker(args, output):
    command = ["docker", "compose", "-f", str(ROOT / "compose.yml"), "run", "--rm", "--no-deps", "-T",
               "--entrypoint", "python", "--volume", f"{ROOT}:/ocr-export:ro",
               "--volume", f"{args.image}:/ocr-input/{args.image.name}:ro",
               "--volume", f"{output}:/ocr-output", "namlhn-gpu",
               "/ocr-export/scripts/export-front-ocr.py", f"/ocr-input/{args.image.name}", "--output", "/ocr-output"]
    if args.model:
        command.extend(["--model", args.model])
    if args.recognizer:
        command.extend(["--recognizer", args.recognizer])
    if args.enhancement:
        command.extend(["--enhancement", args.enhancement])
    if args.card_enhancement:
        command.extend(["--card-enhancement", args.card_enhancement])
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Ảnh mặt trước, JPG hoặc PNG")
    parser.add_argument("--output", type=Path, help="Thư mục xuất mới hoặc đang trống")
    parser.add_argument("--docker", action="store_true", help="Dùng môi trường GPU trong Docker Compose")
    parser.add_argument("--recognizer", choices=("vietocr",), default="vietocr", help="Bộ đọc chữ: VietOCR")
    parser.add_argument("--model", choices=("vgg_seq2seq",), help="Bộ đọc DocU Seq2Seq")
    parser.add_argument("--enhancement", choices=("off", "adaptive"), help="Mặc định theo cấu hình worker")
    parser.add_argument("--card-enhancement", choices=("adaptive", "off"), help="Hiệu chỉnh sáng/tương phản/nhiễu trước OCR")
    args = parser.parse_args(argv)
    args.image = args.image.resolve()
    if not args.image.is_file():
        parser.error(f"Không tìm thấy ảnh: {args.image}")
    if args.image.stat().st_size > 10 * 1024 * 1024:
        parser.error("Mỗi ảnh tối đa 10 MB, cùng giới hạn của dịch vụ OCR.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output or ROOT / "exports" / "ocr-front" / f"{args.image.stem}-{timestamp}").resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Thư mục xuất phải trống để giữ các kết quả trước: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        (run_docker if args.docker else run_local)(args, output)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Xuất OCR thất bại: {exc}\n")
    print(f"HTML: {output / 'index.html'}")
    print(f"ZIP:  {output / 'export.zip'}")


if __name__ == "__main__":
    main()
