from threading import Event, get_ident
import types

import pytest
from PIL import Image

import identification_demo.qr_parallel as parallel_module
from identification_demo.qr_parallel import MAX_ENHANCED_PIXELS, MAX_ENHANCED_SIDE, WeChatParallel, enhanced_views, mapped_regions


PAYLOAD = "001099000001|012345678|NGUYỄN VĂN THỬ|01021999|Nam|Phường Thử, Hà Nội|15032021"


def branch(texts, variant="original"):
    return {"texts": texts, "regions": [{"points": [[1, 2]], "decoded": True}] if texts else [],
            "attempts": 1, "variant": variant}


@pytest.fixture
def reader(monkeypatch):
    decoders = []
    def create():
        decoder = types.SimpleNamespace()
        decoders.append(decoder)
        return decoder
    monkeypatch.setattr(parallel_module, "create_wechat_decoder", create)
    instance = WeChatParallel()
    assert len(decoders) == 2 and decoders[0] is not decoders[1]
    return instance


def test_both_branches_overlap_and_original_wins_when_processed_finishes_first(reader, monkeypatch):
    original_started, processed_finished = Event(), Event()
    threads = {}
    def original(image):
        threads["original"] = get_ident()
        original_started.set()
        assert processed_finished.wait(timeout=3), "Branches did not run concurrently"
        return branch([PAYLOAD])
    def processed(image):
        threads["processed"] = get_ident()
        assert original_started.wait(timeout=3)
        processed_finished.set()
        return branch(["different content"], "enhanced_full")
    monkeypatch.setattr(reader, "_original", original)
    monkeypatch.setattr(reader, "_processed", processed)
    result = reader.read(None)
    assert threads["original"] != threads["processed"]
    assert result["texts"] == [PAYLOAD] and result["selected_branch"] == "original"
    assert result["branches"]["original"]["status"] == "decoded"
    assert result["branches"]["processed"]["status"] == "decoded"
    assert "different content" not in str(result)


@pytest.mark.parametrize("original,processed,selected,expected", [
    ([], [PAYLOAD], "processed", [PAYLOAD]),
    ([PAYLOAD], [], "original", [PAYLOAD]),
    ([], [], None, []),
    (["https://example.com"], [PAYLOAD], "original", ["https://example.com"]),
    ([PAYLOAD, "second code"], ["third code"], "original", [PAYLOAD, "second code"]),
])
def test_branch_selection_never_merges_or_overrides_original(reader, monkeypatch, original, processed, selected, expected):
    monkeypatch.setattr(reader, "_original", lambda _: branch(original))
    monkeypatch.setattr(reader, "_processed", lambda _: branch(processed, "enhanced_full"))
    result = reader.read(None)
    assert result["selected_branch"] == selected
    assert result["texts"] == expected
    assert "texts" not in result["branches"]["original"]
    assert "texts" not in result["branches"]["processed"]


@pytest.mark.parametrize("failing,other_texts,selected", [
    ("original", [PAYLOAD], "processed"), ("processed", [PAYLOAD], "original"),
    ("original", [], None), ("processed", [], None),
])
def test_branch_error_does_not_discard_other_branch(reader, monkeypatch, failing, other_texts, selected):
    def fail(_):
        raise RuntimeError("private decoder failure")
    other = "processed" if failing == "original" else "original"
    monkeypatch.setattr(reader, "_" + failing, fail)
    monkeypatch.setattr(reader, "_" + other, lambda _: branch(other_texts))
    result = reader.read(None)
    assert result["selected_branch"] == selected
    assert result["texts"] == other_texts
    assert result["branches"][failing]["status"] == "error"
    assert "private decoder failure" not in str(result)


def test_both_failed_branches_raise_error(reader, monkeypatch):
    def fail(_):
        raise RuntimeError("cannot decode")
    monkeypatch.setattr(reader, "_original", fail)
    monkeypatch.setattr(reader, "_processed", fail)
    with pytest.raises(RuntimeError, match="cannot decode"):
        reader.read(None)


def test_worker_images_are_released_before_read_returns(reader, monkeypatch):
    completed = []
    def read_image(image):
        assert image.getpixel((0, 0)) == (10, 20, 30)
        completed.append(get_ident())
        return branch([PAYLOAD])
    monkeypatch.setattr(reader, "_original", read_image)
    monkeypatch.setattr(reader, "_processed", read_image)
    with Image.new("RGB", (50, 50), (10, 20, 30)) as image:
        reader.read(image)
    assert len(completed) == 2


def test_enhancement_does_not_mutate_source_and_keeps_each_view_bounded():
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    with Image.new("RGB", (2601, 1903), (15, 100, 200)) as image:
        before = image.tobytes()
        count = 0
        for pixels, view in enhanced_views(image):
            count += 1
            assert pixels.ndim == 2 and pixels.dtype.name == "uint8"
            height, width = pixels.shape
            assert height * width <= MAX_ENHANCED_PIXELS
            assert max(height, width) <= MAX_ENHANCED_SIDE
            assert view["scale_x"] > 0 and view["scale_y"] > 0
            assert 0 <= view["left"] < image.width and 0 <= view["top"] < image.height
        assert count == 5
        assert image.tobytes() == before


def test_small_images_have_single_enhanced_view():
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    with Image.new("RGB", (100, 90), "white") as image:
        views = list(enhanced_views(image))
    assert len(views) == 1
    assert views[0][0].shape == (180, 200)


def test_processed_regions_map_back_to_original_coordinates():
    np = pytest.importorskip("numpy")
    points = [np.array([[0, 0], [201, 0], [201, 303], [0, 303]], dtype=np.float32)]
    regions = mapped_regions(points, {"scale_x": 2.01, "scale_y": 3.03, "left": 40, "top": 75})
    assert np.allclose(regions[0]["points"], [[40, 75], [140, 75], [140, 175], [40, 175]])
    assert regions[0]["decoded"]


def test_processed_branch_stops_at_first_success_and_maps_tile(reader, monkeypatch):
    np = pytest.importorskip("numpy")
    tried = []
    def views(_):
        for i in range(5):
            tried.append(i)
            yield np.full((100, 100), i, dtype=np.uint8), {"name": f"view_{i}", "scale_x": 2,
                "scale_y": 2, "left": i * 20, "top": 30}
    def decode(pixels):
        return ((PAYLOAD, PAYLOAD), [np.array([[0, 0], [20, 0], [20, 20], [0, 20]])]) if pixels[0, 0] == 1 else ((), ())
    monkeypatch.setattr(parallel_module, "enhanced_views", views)
    reader.processed_decoder.detectAndDecode = decode
    result = reader._processed(None)
    assert tried == [0, 1] and result["attempts"] == 2
    assert result["texts"] == [PAYLOAD] and result["variant"] == "view_1"
    assert result["regions"][0]["points"] == [[20., 30.], [30., 30.], [30., 40.], [20., 40.]]


def test_real_wechat_recovers_distant_synthetic_qr():
    from identification_demo.wechat_qr import WeChatQRReader
    if not WeChatQRReader().describe()["available"]:
        pytest.skip("WeChatQRCode dependencies and models are required")
    qrcode = pytest.importorskip("qrcode")
    symbol = qrcode.make(PAYLOAD).convert("RGB").resize((180, 180), Image.Resampling.LANCZOS)
    with Image.new("RGB", (4000, 2600), "white") as image:
        image.paste(symbol, (3100, 350))
        result = WeChatParallel().read(image)
    assert result["branches"]["original"]["status"] == "no_qr"
    assert result["branches"]["processed"]["status"] == "decoded"
    assert result["selected_branch"] == "processed"
    assert result["texts"] == [PAYLOAD]
    points = result["regions"][0]["points"]
    assert all(3090 <= x <= 3290 and 340 <= y <= 540 for x, y in points)
