import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from identification_demo.namlhn_docu_pipeline import extract_pixels
from test_namlhn_docu_pipeline import components


spec = importlib.util.spec_from_file_location("export_front_ocr", Path(__file__).resolve().parents[1] / "scripts/export-front-ocr.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.fixture(autouse=True)
def namlhn_parser(monkeypatch):
    source = Path(__file__).resolve().parents[1] / ".vendor/vietnam-cccd-ocr"
    if not (source / "ocr").is_dir():
        pytest.skip("The namlhn parser checkout is required")
    monkeypatch.syspath_prepend(str(source))


def test_export_pairs_exact_selected_pixels_with_text_including_empty_reads(monkeypatch, tmp_path):
    import identification_demo.text_enhancement as enhancement

    detector, recognizer, boxes, *_ = components(monkeypatch, ["Hà Ni", "", "Nam"], [.7, .99, .99])
    monkeypatch.setattr(enhancement, "needs_enhancement", lambda _image: True)
    monkeypatch.setattr(enhancement, "enhance_text_crop", lambda image: Image.new("RGB", image.size, (12, 34, 56)))

    def reread(image, return_prob):
        assert return_prob and image.getpixel((0, 0)) == (12, 34, 56)
        return "Hà Nội", .95

    recognizer.predictor.predict = reread
    readings = []
    try:
        with Image.new("RGB", (800, 500), (13, 71, 119)) as image:
            raw = extract_pixels(detector, recognizer, image, "front", "adaptive", readings=readings)
            report = exporter.write_export(image, readings, raw, tmp_path, {"name": "front.png"}, "vgg_seq2seq", "adaptive")
        assert [row["text"] for row in report["rows"]] == ["Hà Nội", "", "Nam"]
        assert [row["line_number"] for row in report["rows"]] == [1, None, 2]
        assert [row["box"] for row in report["rows"]] == boxes
        assert report["rows"][0]["selected_reading"] == "enhanced"
        assert report["rows"][0]["readings"]["primary"]["text"] == "Hà Ni"
        assert not report["rows"][0]["readings"]["primary"]["selected"]
        with Image.open(tmp_path / "crops/0001.png") as patch:
            assert patch.size == (100, 32) and patch.getpixel((0, 0)) == (12, 34, 56)
        with Image.open(tmp_path / "attempts/0001.primary.png") as patch:
            assert patch.getpixel((0, 0)) == (255, 255, 255)
        assert (tmp_path / "crops/0001.txt").read_text() == "Hà Nội\n"
        assert (tmp_path / "crops/0002.txt").read_text() == "\n"
        assert (tmp_path / "text.txt").read_text() == "Hà Nội\nNam\n"
        assert json.loads((tmp_path / "result.json").read_text())["rows"] == report["rows"]
        with zipfile.ZipFile(tmp_path / "export.zip") as archive:
            assert archive.testzip() is None
            assert "index.html" in archive.namelist() and "export.zip" not in archive.namelist()
            assert archive.read("crops/0001.txt").decode() == "Hà Nội\n"
    finally:
        exporter.close_readings(readings)


def test_export_preserves_raw_model_text_and_escapes_it_in_offline_html(monkeypatch, tmp_path):
    detector, recognizer, *_ = components(monkeypatch, ["  <script>Đe\u0323p</script>  "], [float("nan")])
    readings = []
    try:
        with Image.new("RGB", (800, 500)) as image:
            raw = extract_pixels(detector, recognizer, image, "front", "off", readings=readings)
            report = exporter.write_export(image, readings, raw, tmp_path, {"name": "front.png"}, "vgg_seq2seq", "off")
        assert report["rows"][0]["readings"]["primary"]["text"] == "  <script>Đe\u0323p</script>  "
        assert report["rows"][0]["text"] == "<script>Đẹp</script>"
        assert report["rows"][0]["confidence"] is None
        page = (tmp_path / "index.html").read_text()
        assert "<script>" not in page and "&lt;script&gt;Đẹp&lt;/script&gt;" in page
        json.loads((tmp_path / "result.json").read_text())
    finally:
        exporter.close_readings(readings)


def test_empty_detection_still_produces_an_openable_report(monkeypatch, tmp_path):
    detector, recognizer, *_ = components(monkeypatch, [])
    readings = []
    try:
        with Image.new("RGB", (800, 500)) as image:
            raw = extract_pixels(detector, recognizer, image, "front", readings=readings)
            report = exporter.write_export(image, readings, raw, tmp_path, {"name": "front.png"}, "vgg_seq2seq", "off")
        assert report["crop_count"] == report["line_count"] == 0
        assert (tmp_path / "export.zip").is_file() and (tmp_path / "text.txt").read_text() == ""
    finally:
        exporter.close_readings(readings)


def test_existing_exports_are_not_overwritten(tmp_path):
    image = tmp_path / "front.png"
    with Image.new("RGB", (10, 10)) as pixels:
        pixels.save(image)
    output = tmp_path / "result"
    output.mkdir()
    (output / "index.html").write_text("Existing report")
    with pytest.raises(SystemExit) as error:
        exporter.main([str(image), "--output", str(output)])
    assert error.value.code == 2
    assert (output / "index.html").read_text() == "Existing report"


def test_export_includes_the_actual_card_before_and_after_preprocessing(monkeypatch, tmp_path):
    detector, recognizer, *_ = components(monkeypatch, ["Số: 1001205025331"])
    readings, artifacts = [], {}
    try:
        with Image.new("RGB", (800, 500), (13, 71, 119)) as image:
            raw = extract_pixels(detector, recognizer, image, "front", readings=readings, artifacts=artifacts)
            report = exporter.write_export(image, readings, raw, tmp_path, {"name": "front.png"},
                                            "vgg_seq2seq", "off", artifacts=artifacts)
        assert report["preprocessing_images"] == {"card": "card.png", "prepared_card": "prepared_card.png"}
        with Image.open(tmp_path / "prepared_card.png") as prepared:
            assert prepared.getpixel((0, 0)) == (13, 71, 119)
        assert report["ocr_result"]["structured_fields"]["id_number"] == "1001205025331"
        assert not report["ocr_result"]["id_number_validation"]["format_valid"]
        with zipfile.ZipFile(tmp_path / "export.zip") as archive:
            assert {"card.png", "prepared_card.png"} <= set(archive.namelist())
    finally:
        exporter.close_readings(readings)
        for artifact in artifacts.values():
            artifact.close()
