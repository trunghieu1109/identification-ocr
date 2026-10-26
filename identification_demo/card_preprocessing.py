"""Localize the main card, retain its coordinate transform, and adjust poor lighting."""
import logging
from pathlib import Path

from .engine import OCRMultipleDocuments


logger = logging.getLogger(__name__)
SEGMENTATION_WEIGHTS = "segmentation/yolo26m_v2/yolo26m_joint_training/weights/best.pt"


def project_polygon(points, transform, size):
    import cv2
    import numpy as np

    if points is None or len(points) == 0:
        return []
    projected = cv2.perspectiveTransform(np.asarray(points, dtype=np.float32).reshape(1, -1, 2), transform)[0]
    projected[:, 0] = np.clip(projected[:, 0], 0, size[0] - 1)
    projected[:, 1] = np.clip(projected[:, 1], 0, size[1] - 1)
    return projected.astype(float).tolist()


def _ordered_quad(points):
    import cv2
    import numpy as np

    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    approximate = cv2.approxPolyDP(hull, .02 * cv2.arcLength(hull, True), True)
    quad = approximate.reshape(-1, 2) if len(approximate) == 4 else cv2.boxPoints(cv2.minAreaRect(hull))
    center = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0]))]
    quad = np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)
    return quad.astype(np.float32)


def crop_card(frame, predictions):
    """Ignore weak/implausible masks; preserve full image if already framed closely."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    identity = np.eye(3, dtype=np.float64)
    metadata = {"method": "docu_yolo_segmentation", "status": "not_found", "image_size": [width, height]}
    candidates = []
    for points, confidence in predictions:
        points = np.asarray(points, dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or not .65 <= confidence <= 1):
            continue
        quad = _ordered_quad(points)
        edges = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        short, long = min(edges), max(edges)
        area = abs(float(cv2.contourArea(quad)))
        fraction = area / (width * height)
        if short < 64 or not 1.2 <= long / short <= 2.2 or not .02 <= fraction <= 1.05:
            continue
        bounds = cv2.boundingRect(quad)
        candidates.append((area, quad, float(confidence), bounds))
    candidates.sort(key=lambda item: item[0], reverse=True)
    unique = []
    for candidate in candidates:
        bounds = candidate[3]
        duplicate = False
        for _area, _quad, _confidence, previous in unique:
            x, y, w, h = bounds
            a, b, c, d = previous
            overlap = max(0, min(x + w, a + c) - max(x, a)) * max(0, min(y + h, b + d) - max(y, b))
            if overlap / min(w * h, c * d) > .8:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    candidates = unique
    metadata["detected_regions"] = len(candidates)
    if not candidates:
        return frame, identity, metadata
    if len(candidates) > 1 and candidates[0][0] < 4 * candidates[1][0]:
        raise OCRMultipleDocuments(len(candidates))
    area, quad, confidence, _bounds = candidates[0]
    metadata.update(confidence=confidence, polygon=quad.astype(float).tolist(), area_fraction=round(area / (width * height), 4))
    if area / (width * height) >= .85:
        metadata["status"] = "already_framed"
        return frame, identity, metadata
    # Add a small margin so a slightly conservative mask cannot clip edge text.
    center = quad.mean(axis=0)
    quad = center + (quad - center) * 1.05
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    horizontal = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    vertical = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    scale = min(1., 4096 / max(horizontal, vertical))
    output_width, output_height = max(4, round(horizontal * scale)), max(4, round(vertical * scale))
    destination = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1],
                            [0, output_height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(quad, destination)
    image = cv2.warpPerspective(frame, transform, (output_width, output_height),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    metadata.update(status="cropped", polygon=quad.astype(float).tolist(), crop_size=[output_width, output_height],
                    inverse_transform=np.linalg.inv(transform).tolist())
    return image, np.linalg.inv(transform), metadata


class CardCropper:
    def __init__(self, model_root="/app/model", device="cuda:0"):
        from ultralytics import YOLO

        weights = Path(model_root) / SEGMENTATION_WEIGHTS
        if not weights.is_file():
            raise FileNotFoundError("Thiếu checkpoint cục bộ để tách vùng thẻ.")
        self.model, self.device = YOLO(str(weights)), device
        import numpy as np
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device=self.device,
                           conf=.65, imgsz=640, verbose=False)

    def __call__(self, frame):
        results = self.model.predict(frame, device=self.device, conf=.65, imgsz=640, verbose=False)
        predictions = []
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            for polygon, score in zip(result.masks.xy, result.boxes.conf.cpu().tolist()):
                predictions.append((polygon, float(score)))
        return crop_card(frame, predictions)


def enhance_card(frame, mode="adaptive"):
    import cv2
    import numpy as np

    if mode not in {"off", "adaptive"}:
        raise ValueError("Card enhancement must be off or adaptive")
    metadata = {"mode": mode, "brightness": False, "contrast": False, "denoise": False}
    if mode == "off":
        return frame, metadata
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    low, middle, high = np.percentile(light, [10, 50, 90])
    noise = float(np.median(np.abs(light.astype(np.float32) - cv2.medianBlur(light, 3)))) / .6745
    if noise >= 5 or (middle < 100 and noise >= 2):
        clean = cv2.bilateralFilter(frame, 5, 20, 5)
        lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
        light = lab[:, :, 0]
        metadata["denoise"] = True
    if middle < 100 and high < 225:
        table = np.round(255 * (np.arange(256) / 255.) ** .7).astype(np.uint8)
        light = cv2.LUT(light, table)
        metadata["brightness"] = True
    if high - low < 80 and high < 230 and float(light.std()) > 4:
        light = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(light)
        metadata["contrast"] = True
    if not any(metadata[key] for key in ("brightness", "contrast", "denoise")):
        return frame, metadata
    lab[:, :, 0] = light
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), metadata


def prepare_card(frame, cropper=None, enhancement="off", artifacts=None):
    import cv2
    import numpy as np
    from PIL import Image

    inverse = np.eye(3, dtype=np.float64)
    crop = frame
    metadata = {"crop": {"status": "off", "image_size": [frame.shape[1], frame.shape[0]]}}
    if cropper is not None:
        try:
            crop, inverse, metadata["crop"] = cropper(frame)
        except OCRMultipleDocuments:
            raise
        except Exception:
            logger.exception("Card localization failed; reading the original image")
            metadata["crop"]["status"] = "error"
    prepared, metadata["enhancement"] = enhance_card(crop, enhancement)
    metadata["working_image_size"] = [prepared.shape[1], prepared.shape[0]]
    if artifacts is not None:
        artifacts["card"] = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        artifacts["prepared_card"] = Image.fromarray(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))
    return prepared, inverse, metadata
