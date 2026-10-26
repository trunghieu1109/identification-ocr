from types import SimpleNamespace

import numpy as np
import pytest

from identification_demo.expiry import parse_expiry
from identification_demo.expiry_context import context_strips
from identification_demo.field_recovery import recover_date
from identification_demo.namlhn_text_processing import _tight_text_bounds, read_text_region


def row(text, box=(700, 600, 930, 630), confidence=.98):
    left, top, right, bottom = box
    return {'text': text, 'box': [[left, top], [right, top], [right, bottom], [left, bottom]],
            'confidence': confidence}


def test_expanded_search_includes_caption_far_to_left_of_date_only_box():
    original = [row('20/10/2028')]
    calls = []
    def read(bounds):
        calls.append(bounds)
        return [row('Có giá trị đến: 20/10/2028', (250, 600, 930, 630))], 1
    result, meta = recover_date(original, (1200, 760), 'front', read, expanded_context=True)
    assert calls[0][0] == 0 and calls[0][2] == 1200
    assert meta['status'] == 'recovered' and original[0] in result
    assert parse_expiry(result)[0]['expiry_date'] == '20/10/2028'


def test_original_v2_keeps_its_narrow_search_bounds():
    observed = []
    recover_date([row('20/10/2028')], (1200, 760), 'front',
                 lambda bounds: (observed.append(bounds) or [], 0))
    assert observed[0][0] > 0 and observed[0][2] < 1200


@pytest.mark.parametrize('recovered', [
    row('20/10/2028'), row('Có giá trị đến: 20/10/2029'),
    row('Ngày sinh: 20/10/2028'), row('Có giá trị đến: 20/10/2028', confidence=.4),
])
def test_expanded_inline_read_cannot_change_known_date_or_invent_label(recovered):
    original = [row('20/10/2028')]
    result, meta = recover_date(original, (1200, 760), 'front', lambda _: ([recovered], 1),
                               expanded_context=True)
    assert result is original and meta['status'] == 'not_found'


@pytest.mark.parametrize('original', [
    [row('Có giá trị đến: 20/10/2028')],
    [row('Ngày sinh', (700, 560, 930, 590)), row('20/10/2028')],
    [row('20/10/2028'), row('20/10/2029', (700, 400, 930, 430))],
])
def test_resolved_owned_or_ambiguous_dates_do_not_add_gpu_work(original):
    result, meta = recover_date(original, (1200, 760), 'front',
        lambda _: pytest.fail('Unexpected inference'), expanded_context=True)
    assert result is original and meta['status'] == 'skipped'


def test_strip_bounds_are_clipped_to_card_and_separate_above_rows():
    boxes = context_strips((700, 600, 930, 630), (1200, 760))
    assert len(boxes) == 3 and boxes[1][3] < 600 and boxes[2][3] <= boxes[1][1]
    boxes = context_strips((10, 8, 98, 32), (100, 60))
    assert all(0 <= left < right <= 100 and 0 <= top < bottom <= 60
               for left, top, right, bottom in boxes)


def test_uniform_margin_is_trimmed_while_caption_and_date_pixels_are_retained():
    image = np.full((100, 1000, 3), 245, dtype=np.uint8)
    image[40:50, 350:450] = 50
    image[35:60, 700:900] = 20
    tight = _tight_text_bounds(image, (0, 20, 950, 80))
    assert tight[0] < 350 and tight[2] > 900 and tight[1] < 35 and tight[3] > 60
    assert tight[0] > 300 and tight[1] > 20
    assert _tight_text_bounds(image, (0, 0, 300, 30)) is None


def test_expanded_rows_share_one_local_recognition_batch_and_do_not_add_detector_passes(monkeypatch):
    image = np.full((100, 1000, 3), 245, dtype=np.uint8)
    image[35:55, 300:450] = 30
    image[35:60, 700:900] = 20
    calls = []
    config = {'dataset': {'image_max_width': 512}}
    def detect(frame):
        calls.append(('detect', detector.imgsz))
        return []
    detector = SimpleNamespace(imgsz=640, conf_threshold=.25, detect_single_image=detect)
    def crop(frame, coords, **kwargs):
        from PIL import Image
        xs, ys = zip(*coords)
        return Image.fromarray(frame[int(min(ys)):int(max(ys)), int(min(xs)):int(max(xs))])
    def predict(patches, **kwargs):
        calls.append(('recognize', len(patches)))
        assert patches[0].width > 550 and patches[0].height < 40
        assert config['dataset']['image_max_width'] == 1024
        return ['Có giá trị đến: 20/10/2028'], [.98]
    recognizer = SimpleNamespace(WHOLE_CARD_CLASSES=set(), crop_oriented_box=crop,
        predictor=SimpleNamespace(config=config, predict_batch=predict))
    monkeypatch.setattr('identification_demo.namlhn_text_processing.prepare_text_image', lambda img: img.copy())
    rows, count = read_text_region(detector, recognizer, image, (0, 0, 1000, 100),
                                   extra_boxes=[(0, 20, 950, 80)])
    assert calls == [('detect', 960), ('recognize', 1)] and count == 1
    assert rows[0].text.startswith('Có giá trị đến') and rows[0].bbox.coordinates[0][0] < 300
    assert detector.imgsz == 640 and detector.conf_threshold == .25
    assert config['dataset']['image_max_width'] == 512
