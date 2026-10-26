"""Port namlhn's image enhancement and absolute crop margins to DocU models.

Based on ocr/preprocess.py and IDCardExtractor._padding at namlhn revision
2d951ac603a43677572ff39ed6a21bdebe622263. No additional OCR model is loaded.
"""
from copy import copy
import math

from .authority_region import _line
from .authority_recovery import _same_line


def _tight_text_bounds(image, bounds):
    """Trim uniform margins of an expanded line, retaining all visible text.

    This is pixel thresholding, not another text detector. It keeps a long
    blank margin from shrinking a small caption when VietOCR resizes a row.
    Textured crops retain their requested bounds rather than guessing rows.
    """
    import cv2
    import numpy as np

    left, top, right, bottom = bounds
    gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    if not gray.size or float(gray.max())-float(gray.min()) < 20:
        return None
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask = gray < min(220, max(threshold, float(np.percentile(gray, 90))-40))
    coverage = float(mask.mean())
    if coverage < .002:
        return None
    if coverage > .4:
        return bounds
    ys, xs = np.nonzero(mask)
    return (left+max(0, int(xs.min())-2), top+max(0, int(ys.min())-2),
            left+min(gray.shape[1], int(xs.max())+3), top+min(gray.shape[0], int(ys.max())+3))


def prepare_text_image(image):
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    # Bound elongated local crops; upscaling a narrow band to a 960px short
    # side without a long-side cap would create needlessly huge images.
    scale = min(max(1, 960/min(h, w)), 2560/max(h, w))
    processed = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))),
                           interpolation=cv2.INTER_CUBIC) if abs(scale-1) > .001 else image.copy()
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = gray
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    processed = cv2.filter2D(processed, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32))
    return processed


def read_text_region(detector, recognizer, image, bounds, seeds=(), max_regions=12, extra_boxes=()):
    left, top, right, bottom = bounds
    processed = prepare_text_image(image[top:bottom, left:right])
    sx, sy = processed.shape[1]/(right-left), processed.shape[0]/(bottom-top)
    settings = detector.imgsz, detector.conf_threshold
    try:
        detector.imgsz = max(960, detector.imgsz)
        detector.conf_threshold = min(.15, detector.conf_threshold)
        detected = detector.detect_single_image(processed)
    finally:
        detector.imgsz, detector.conf_threshold = settings
    candidates = []
    for item in detected:
        if item.field_key in recognizer.WHOLE_CARD_CLASSES:
            continue
        candidate = copy(item)
        candidate.bbox = copy(item.bbox)
        candidate.bbox.coordinates = [(x/sx+left, y/sy+top) for x, y in item.bbox.coordinates]
        candidates.append(candidate)
    # Keep primary boxes that a second detector pass missed, as namlhn keeps
    # its first pass as evidence for targeted re-reading.
    for item in seeds:
        if getattr(item, 'field_key', None) in recognizer.WHOLE_CARD_CLASSES:
            continue
        _text, box, _confidence = _line(item)
        if not box or not (left <= (box[0]+box[2])/2 <= right and top <= (box[1]+box[3])/2 <= bottom):
            continue
        if any(_same_line(_line(other)[1], box) for other in candidates):
            continue
        candidate = copy(item)
        candidate.bbox = copy(item.bbox)
        candidates.append(candidate)
    candidates.sort(key=lambda item: ((_line(item)[1] or (0,0,0,0))[1], (_line(item)[1] or (0,0,0,0))[0]))
    extras = []
    if extra_boxes:
        from types import SimpleNamespace
        for x1, y1, x2, y2 in list(extra_boxes)[:3]:
            local = (max(0, int((x1-left)*sx)), max(0, int((y1-top)*sy)),
                     min(processed.shape[1], math.ceil((x2-left)*sx)),
                     min(processed.shape[0], math.ceil((y2-top)*sy)))
            if local[2]-local[0] <= 3 or local[3]-local[1] <= 3:
                continue
            tight = _tight_text_bounds(processed, local)
            if tight is None:
                continue
            x1, y1, x2, y2 = tight
            coords = [(x1/sx+left, y1/sy+top), (x2/sx+left, y1/sy+top),
                      (x2/sx+left, y2/sy+top), (x1/sx+left, y2/sy+top)]
            extras.append(SimpleNamespace(text='', confidence=1., field_key='text_line',
                expanded_context=True, bbox=SimpleNamespace(coordinates=coords)))
    # All local boxes, including direct expanded rows, share one recognition
    # batch. The original v2 path has no extra boxes and remains unchanged.
    candidates = candidates[:max(0, max_regions-len(extras))]+extras
    patches, selected = [], []
    config = recognizer.predictor.config['dataset']
    original_width = config['image_max_width']
    try:
        config['image_max_width'] = max(1024, original_width)
        pad = max(int(4*processed.shape[0]/640), 2)
        for item in candidates:
            coords = [((x-left)*sx, (y-top)*sy) for x, y in item.bbox.coordinates]
            xs, ys = [p[0] for p in coords], [p[1] for p in coords]
            w, h = max(1, max(xs)-min(xs)), max(1, max(ys)-min(ys))
            patch = recognizer.crop_oriented_box(processed, coords,
                pad_top=pad/h, pad_bot=pad/h, pad_x=pad/w)
            if patch is not None:
                patches.append(patch)
                selected.append(item)
        if not patches:
            return [], 0
        texts, probabilities = recognizer.predictor.predict_batch(patches, return_prob=True)
        for item, text, probability in zip(selected, texts, probabilities):
            item.text = text.strip() if isinstance(text, str) else ''
            item.confidence = float(probability) if probability is not None and math.isfinite(float(probability)) else 0
        return selected[:min(len(texts), len(probabilities))], len(patches)
    finally:
        config['image_max_width'] = original_width
        for patch in patches:
            patch.close()
