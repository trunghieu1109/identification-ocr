"""Comparison-only WeChat branches: original pixels and classical enhancement."""
import logging
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from time import perf_counter

from .wechat_qr import create_wechat_decoder, decode_wechat


logger = logging.getLogger(__name__)
MAX_ENHANCED_PIXELS = 2_000_000
MAX_ENHANCED_SIDE = 2000


def enhanced_views(image):
    """Yield a full view and four overlapping tiles, bounded one view at a time.

    Tiles let small symbols occupy more of WeChat's internal detector input.
    Enhancement uses grayscale, CLAHE, cubic scaling and mild unsharp masking;
    it loads no additional model and never changes the caller's pixels.
    """
    import cv2
    import numpy as np
    width, height = image.size
    boxes = [(0, 0, width, height)]
    if min(width, height) >= 240:
        tile_width, tile_height = round(width * .6), round(height * .6)
        boxes.extend((x, y, x + tile_width, y + tile_height)
                     for y in (0, height - tile_height) for x in (0, width - tile_width))
    for index, (left, top, right, bottom) in enumerate(boxes):
        with image.crop((left, top, right, bottom)) as crop:
            gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        crop_height, crop_width = gray.shape
        scale = min(2., sqrt(MAX_ENHANCED_PIXELS / (crop_width * crop_height)),
                    MAX_ENHANCED_SIDE / max(crop_width, crop_height))
        output_width, output_height = max(1, int(crop_width * scale)), max(1, int(crop_height * scale))
        gray = cv2.resize(gray, (output_width, output_height),
                          interpolation=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA)
        contrast = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray)
        blurred = cv2.GaussianBlur(contrast, (0, 0), .8)
        sharpened = cv2.addWeighted(contrast, 1.6, blurred, -.6, 0)
        # Coordinate mapping uses the actual rounded size, not the requested scale.
        yield sharpened, {"name": "enhanced_full" if index == 0 else f"enhanced_tile_{index}",
                          "left": left, "top": top, "scale_x": output_width / crop_width,
                          "scale_y": output_height / crop_height}


def mapped_regions(points, view=None):
    regions = []
    for polygon in points:
        coordinates = polygon.reshape(-1, 2).tolist()
        if view is not None:
            coordinates = [[x / view["scale_x"] + view["left"],
                            y / view["scale_y"] + view["top"]] for x, y in coordinates]
        regions.append({"points": coordinates, "decoded": True})
    return regions


class WeChatParallel:
    def __init__(self):
        # OpenCV DNN contexts must not be shared by simultaneous inference calls.
        self.original_decoder = create_wechat_decoder()
        self.processed_decoder = create_wechat_decoder()

    def _original(self, image):
        texts, points = decode_wechat(self.original_decoder, image)
        return {"texts": list(dict.fromkeys(texts)), "regions": mapped_regions(points),
                "attempts": 1, "variant": "original"}

    def _processed(self, image):
        attempts = 0
        for pixels, view in enhanced_views(image):
            attempts += 1
            texts, points = self.processed_decoder.detectAndDecode(pixels)
            if texts:
                return {"texts": list(dict.fromkeys(texts)), "regions": mapped_regions(points, view),
                        "attempts": attempts, "variant": view["name"]}
        return {"texts": [], "regions": [], "attempts": attempts, "variant": None}

    def _run_branch(self, name, image):
        started = perf_counter()
        try:
            output = getattr(self, f"_{name}")(image)
            summary = {"status": "decoded" if output["texts"] else "no_qr",
                       "decoded_count": len(output["texts"]), "attempts": output["attempts"],
                       "variant": output["variant"]}
        except Exception as exc:
            # Keep internal failures and any user data out of the public report.
            logger.warning("QR experiment branch %s failed (%s)", name, type(exc).__name__)
            output = {"texts": [], "regions": [], "error": exc}
            summary = {"status": "error", "decoded_count": 0, "attempts": None, "variant": None,
                       "message": "Nhánh này gặp lỗi xử lý; kết quả nhánh còn lại vẫn được giữ."}
        summary["processing_time_ms"] = round((perf_counter() - started) * 1000, 2)
        output["summary"] = summary
        return output

    def read(self, image):
        # Await both in this comparison tool so all branch timings are reviewable
        # and no worker can retain image pixels after the request is complete.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qr-experiment") as workers:
            original_future = workers.submit(self._run_branch, "original", image)
            processed_future = workers.submit(self._run_branch, "processed", image)
            original, processed = original_future.result(), processed_future.result()
        if "error" in original and "error" in processed:
            raise original["error"]
        # Completion order never overrides preference for any decoded original
        # content, including unsupported or multiple QR payloads.
        selected = "original" if original["texts"] else "processed" if processed["texts"] else None
        chosen = original if selected == "original" else processed
        return {"texts": chosen["texts"], "regions": chosen["regions"], "selected_branch": selected,
                "branches": {"original": original["summary"], "processed": processed["summary"]}}
