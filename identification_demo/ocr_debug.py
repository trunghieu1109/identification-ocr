"""Display the actual back-side OCR lines before metadata extraction."""
import math

from .authority_region import _line, select_authority_region


def back_ocr_debug(initial, final, search_crop=None):
    def snapshot(items):
        items = list(items) if isinstance(items, (list, tuple)) else []
        region = select_authority_region(items)
        selected = set(region.body_indices)
        rows = [{"number": index + 1, "text": _line(item)[0], "in_authority": index in selected}
                for index, item in enumerate(items)]
        return rows, region

    first, _ = snapshot(initial)
    last, region = snapshot(final)
    result = {"initial_lines": first, "final_lines": last,
              "authority_candidate": " ".join(region.parts), "ambiguous": region.ambiguous}
    if isinstance(search_crop, dict):
        preview, bbox, size = search_crop.get("image"), search_crop.get("bbox"), search_crop.get("image_size")
        if (isinstance(preview, str) and preview.startswith("data:image/jpeg;base64,") and len(preview) <= 1_000_000
                and isinstance(bbox, (list, tuple)) and len(bbox) == 4
                and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in bbox)
                and isinstance(size, (list, tuple)) and len(size) == 2
                and all(type(value) is int and value > 0 for value in size)
                and bbox[0] < bbox[2] <= size[0] and bbox[1] < bbox[3] <= size[1]):
            result["search_crop"] = {"image": preview, "bbox": list(bbox), "image_size": list(size)}
    return result


def authority_crop_preview(image, bbox):
    """Encode the exact searched pixels; no interpolation of missing content."""
    import base64
    import io
    from PIL import Image

    left, top, right, bottom = bbox
    with Image.fromarray(image[top:bottom, left:right, ::-1]).convert("RGB") as patch:
        patch.thumbnail((1000, 700))
        buffer = io.BytesIO()
        patch.save(buffer, format="JPEG", quality=85)
    return {"image": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "bbox": list(bbox), "image_size": [int(image.shape[1]), int(image.shape[0])]}
