"""Adjust lighting and noise while preserving the full uploaded image geometry."""


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


def prepare_card(frame, enhancement="off", artifacts=None):
    import cv2
    from PIL import Image

    metadata = {"crop": {"status": "off", "image_size": [frame.shape[1], frame.shape[0]]}}
    prepared, metadata["enhancement"] = enhance_card(frame, enhancement)
    metadata["working_image_size"] = [prepared.shape[1], prepared.shape[0]]
    if artifacts is not None:
        artifacts["card"] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        artifacts["prepared_card"] = Image.fromarray(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))
    return prepared, metadata
