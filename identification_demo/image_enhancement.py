"""Conservative text-patch enhancement; never generates new image content."""
from PIL import Image, ImageFilter, ImageOps


def trim_text_band(image):
    """Trim empty vertical margins from a narrowly selected text-line band."""
    import numpy as np
    gray = np.asarray(image.convert("L"))
    threshold = min(160, float(np.median(gray)) - 30)
    counts = (gray < threshold).sum(axis=1)
    rows = np.flatnonzero(counts >= max(2, image.width * .02))
    if not len(rows):
        return image.copy()
    bands = np.split(rows, np.flatnonzero(np.diff(rows) > 3) + 1)
    bands = [band for band in bands if len(band) >= 4]
    if not bands:
        return image.copy()
    band = max(bands, key=lambda value: counts[value].sum())
    margin = max(2, round((band[-1] - band[0] + 1) * .15))
    return image.crop((0, max(0, int(band[0]) - margin), image.width,
                       min(image.height, int(band[-1]) + margin + 1)))


def enhance_text_patch(image):
    patch = ImageOps.autocontrast(image.convert("L"), cutoff=0.5)
    scale = min(2.0, max(1.0, 64 / patch.height), 2200 / patch.width)
    if scale != 1:
        patch = patch.resize((max(1, round(patch.width * scale)), max(1, round(patch.height * scale))), Image.Resampling.LANCZOS)
    return patch.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3)).convert("RGB")
