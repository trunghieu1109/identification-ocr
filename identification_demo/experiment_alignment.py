"""DocAligner LCNet050 ONNX inference, isolated from both production engines.

Pre/postprocessing follows DocsaidLab/DocAligner point_reg/infer.py at
3275b0f07f8e99d8c01cb0774dea2549be1416b6 (Apache-2.0). Direct ONNXRuntime
avoids loading unrelated Capybara training/visualization dependencies.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def rectify(image, points, confidence):
    """Reject unsafe quads and preserve the full-resolution pixels for warping."""
    finite_confidence = bool(np.isfinite(confidence))
    metadata = {"status": "original", "confidence": round(float(confidence), 4) if finite_confidence else None, "corners": [],
                "reason": "Không tìm được bốn góc đủ tin cậy; giữ ảnh gốc."}
    points = np.asarray(points, dtype=np.float32)
    width, height = image.size
    if not finite_confidence or confidence < .75 or points.size != 8 or not np.isfinite(points).all():
        return image.copy(), metadata
    points = points.reshape(4, 2)
    metadata["candidate_corners"] = points.round(2).tolist()
    if ((points < 0).any() or (points[:, 0] > width - 1).any() or (points[:, 1] > height - 1).any()):
        return image.copy(), metadata
    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < width * height * .08:
        return image.copy(), metadata
    # Convex-hull order cannot duplicate a corner on diamond-shaped documents.
    ordered = np.roll(hull, -np.argmin(hull.sum(axis=1)), axis=0)
    if np.cross(ordered[1] - ordered[0], ordered[2] - ordered[1]) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    top, right, bottom, left = [float(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])) for i in range(4)]
    out_width, out_height = round(max(top, bottom)), round(max(left, right))
    ratio = max(out_width, out_height) / max(1, min(out_width, out_height))
    if min(top, right, bottom, left) < 40 or not 1.15 <= ratio <= 2.5:
        return image.copy(), metadata
    scale = min(1.0, 2560 / max(out_width, out_height))
    out_width, out_height = round(out_width * scale), round(out_height * scale)
    destination = np.float32([[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1], [0, out_height - 1]])
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    aligned = cv2.warpPerspective(np.asarray(image), matrix, (out_width, out_height), flags=cv2.INTER_CUBIC)
    return Image.fromarray(aligned), {"status": "aligned", "confidence": metadata["confidence"],
                                    "corners": ordered.round(2).tolist(), "output_size": [out_width, out_height]}


class DocAligner:
    def __init__(self, model_dir):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(Path(model_dir) / "docaligner-lcnet050.onnx"),
                                            sess_options=options, providers=["CPUExecutionProvider"])

    def align(self, image):
        # Upstream point model uses BGR, NCHW, [0, 1], no mean/std transform.
        pixels = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        tensor = cv2.resize(pixels, (256, 256)).transpose(2, 0, 1)[None].astype("float32") / 255
        outputs = dict(zip([item.name for item in self.session.get_outputs()],
                           self.session.run(None, {self.session.get_inputs()[0].name: tensor})))
        confidence = float(np.asarray(outputs["has_obj"]).reshape(-1)[0])
        points = np.asarray(outputs["points"]).reshape(4, 2) * np.array(image.size)
        return rectify(image, points, confidence)
