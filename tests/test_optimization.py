from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from identification_demo.cpu_pipeline import reread_indexes
from identification_demo.gpu_engine import DocUGPUEngine, normalize_result
from identification_demo.pipeline import CARD_FIELDS, process_card
from test_pipeline import FRONT, BACK, PAYLOAD, Reader, image_data
from test_regions import FULL_BACK, box, fake_engine, readable_back


def test_combined_reuses_decoded_images_and_only_requests_qr_gaps(monkeypatch):
    import identification_demo.pipeline as pipeline
    original = pipeline.load_image
    decoded, sent = [], []
    def load(data):
        image = original(data)
        decoded.append(image)
        return image
    class Pixels:
        def extract_image(self, image, side, needed_fields, source_data):
            assert image is decoded[0 if side == "front" else 1]
            assert image.getpixel((0, 0)) == (255, 255, 255)
            sent.append((side, needed_fields))
            assert source_data == image_data(side)
            return FRONT if side == "front" else BACK
    monkeypatch.setattr(pipeline, "load_image", load)
    result = process_card({"front": image_data(), "back": image_data("back")}, Pixels(), Reader())
    assert len(decoded) == 2
    assert sent == [("front", ["nationality", "place_of_origin"]), ("back", ["issuing_authority"])]
    assert result["fields"]["issue_date"] == "15/03/2021"
    assert list(result["fields"]) == list(CARD_FIELDS)
    assert result["fields"]["identifying_features"] == ""
    assert result["status"] == "complete"
    timings = result["timings_ms"]
    assert set(timings) == {"image_decode", "qr_front", "qr_back", "ocr_front", "ocr_back", "merge", "total"}
    assert sum(value for key, value in timings.items() if key != "total") == pytest.approx(timings["total"], abs=.03)
    with pytest.raises(ValueError):
        decoded[0].getpixel((0, 0))


def test_cpu_rereads_only_requested_free_text_even_with_confident_paddle():
    def line(text, y, score=.99):
        return {"text": text, "box": box(0, y, 100, y + 15), "score": score}
    lines = [line("Họ và tên: TÊN KHÁC QR", 0, .3), line("Quốc tịch: Việt Nam", 30),
             line("Quê quán", 60), line("Hà Ni", 90), line("Nơi thường trú: Địa chỉ khác QR", 120, .2)]
    raw = {"Nationality_box": box(0, 30, 100, 45), "Place_of_origin_box": box(0, 60, 100, 105)}
    selected = reread_indexes(lines, raw, {"nationality": "Việt Nam", "place_of_origin": "Hà Ni"},
                              "front", ["nationality", "place_of_origin"])
    assert selected == [2, 3]


def test_cpu_missing_unlocated_field_keeps_one_broad_fallback():
    lines = [{"text": "damaged", "box": box(0, 0, 100, 20), "score": .3}]
    assert reread_indexes(lines, {}, {"name": ""}, "front", ["name"]) == [0]


def test_authority_reread_does_not_include_qr_date_or_signer():
    rows = ["Ngày cấp: 20/04/2021", "CỤC TRƯỞNG CỤC CẢNH SÁT",
            "QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", "Thiếu tướng Nguyễn Văn Mẫu"]
    lines = [{"text": text, "score": .99} for text in rows]
    selected = reread_indexes(lines, {}, {"issuing_authority": " ".join(rows[1:3])},
                              "back", ["issuing_authority"])
    assert selected == [1, 2]


def test_cpu_batch_corrects_requested_vietnamese_field_with_real_namlhn_parser():
    import numpy as np
    from identification_demo.cpu_pipeline import extract_pixels
    from identification_demo.engine import NamLhnEngine
    if not NamLhnEngine().source_available:
        pytest.skip("Upstream checkout is required")
    rows = ["Họ và tên: NGUYỄN VĂN MẪU", "Ngày sinh: 01/01/2000", "Quốc tịch: Việt Nam",
            "Quê quán / Place of origin: Hà Ni", "Nơi thường trú: Địa chỉ đã có từ QR"]
    lines = [{"text": text, "score": .99, "box": box(0, i * 30, 500, i * 30 + 20)} for i, text in enumerate(rows)]
    batches = []
    def recognize(patches, return_prob):
        batches.append(len(patches))
        return ["Quê quán / Place of origin: Hà Nội"] * len(patches), [.99] * len(patches)
    extractor = SimpleNamespace(enhance=False, resize=False, detect=lambda _image: lines,
                                _warp_region=lambda *_args, **_kwargs: np.full((20, 200, 3), 255, dtype=np.uint8),
                                _postprocess_text=lambda text: text,
                                _vietocr=SimpleNamespace(predict_batch=recognize))
    with Image.new("RGB", (600, 300)) as image:
        result = extract_pixels(extractor, image, "front", ["place_of_origin"])
    assert batches == [1]
    assert result["structured_fields"]["place_of_origin"] == "Hà Nội"
    assert result["work"]["primary_regions"] == 5 and result["work"]["reread_regions"] == 1


def test_gpu_does_not_retry_rotation_for_ignored_or_qr_filled_fields():
    doc = readable_back("one", box(0, 0, 100, 60), complete=False)
    doc.recognized_texts = [SimpleNamespace(text="Đặc điểm nhân dạng"), SimpleNamespace(text="Ngày cấp"),
                            SimpleNamespace(text="Cơ quan cấp: Công an tỉnh Hà Nam")]
    doc.extracted_fields = {}
    engine = fake_engine([doc])
    engine._read_full_image = lambda *_args: pytest.fail("No redundant full-image OCR")
    result = engine.extract_document("unused.png", "back", ["issuing_authority"])
    assert "recovery" not in result
    assert result["work"]["full_image_attempts"] == 1
    engine.close()


def test_gpu_reuses_identical_original_instead_of_recognizing_it_twice():
    import numpy as np
    frame = np.full((100, 160, 3), 255, dtype=np.uint8)
    doc = readable_back("one", box(0, 0, 160, 100), complete=False)
    doc.cropped_image = frame.copy()
    doc.recognized_texts = [SimpleNamespace(text="Đặc điểm nhân dạng"), SimpleNamespace(text="Ngày cấp"),
                            SimpleNamespace(text="Cơ quan cấp: Công an tỉnh Hà Nam")]
    doc.extracted_fields = {}
    engine = fake_engine([doc])
    rotations = []
    def recover(_frame, _side, rotation):
        rotations.append(rotation)
        return dict(FULL_BACK)
    engine._read_full_image = recover
    # Exercise the same array path as the actual worker's decoded upload.
    result = engine._extract_document(frame, "back", ["issue_date", "issuing_authority"])
    assert rotations == [180]
    assert result["work"]["skipped_duplicate_passes"] == 1
    assert result["work"]["full_image_attempts"] == 2
    assert result["recovery"]["attempted_rotations"] == [0, 180]
    engine.close()


def test_gpu_http_reuses_client_for_two_sides_and_forwards_only_needed_fields():
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path.endswith("ready"):
            return httpx.Response(200, json={"pipeline_ready": True, "device": "cuda:0", "cuda_available": True})
        assert request.url.params["fields"] == "issuing_authority"
        return httpx.Response(200, json={"success": True, "doc_type": "cccd_back", "extracted_fields": {},
                                        "text_lines": ["Cơ quan cấp: Công an tỉnh Hà Nam"]})
    engine = DocUGPUEngine("http://gpu.test", transport=httpx.MockTransport(handler))
    first_client = engine._client()
    with Image.new("RGB", (20, 20)) as image:
        for _ in range(2):
            engine.extract_image(image, "back", ["issuing_authority"], image_data("back"))
    assert engine._client() is first_client
    assert [request.url.path for request in requests] == ["/api/v1/health/ready", "/api/v1/extract", "/api/v1/extract"]
    engine.close()


def test_gpu_timing_metadata_cannot_leak_arbitrary_values():
    raw = normalize_result({"success": True, "extracted_fields": {},
                            "timings_ms": {"recognition": 42, "raw_text": "private", "detection": float("inf")},
                            "work": {"primary_regions": 2, "person": "private", "reread_regions": -1}})
    assert raw["timings_ms"] == {"recognition": 42}
    assert raw["work"] == {"primary_regions": 2}
