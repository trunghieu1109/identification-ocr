"""Bounded FastMRZ attempts; OCR stays available when MRZ is unreadable."""
from time import perf_counter

from .experiment_mrz import mrz_rank, parse_vietnam_td1, usable_mrz_fields


class MRZReader:
    def __init__(self, directory):
        from fastmrz import FastMRZ
        self.directory = str(directory)
        self.reader = FastMRZ(tessdata_path=self.directory)

    def _fast(self, pixels):
        return self.reader.get_details(pixels, input_type="numpy", ignore_parse=True)

    def _bottom(self, pixels):
        import cv2
        import pytesseract
        gray = cv2.cvtColor(pixels[round(pixels.shape[0] * .55):], cv2.COLOR_BGR2GRAY)
        factor = max(1.0, min(3.0, 1500 / gray.shape[1]))
        gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        return pytesseract.image_to_string(gray, lang="mrz", timeout=2,
            config=f'--tessdata-dir "{self.directory}" --oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')

    def read(self, image, *, fallback_image=None, variant="original", stop_event=None,
             max_attempts=3, budget_ms=900):
        import cv2
        import numpy as np
        started = perf_counter()
        best = parse_vietnam_td1("")
        attempts = []
        base_rotation = 0 if image.width >= image.height else 90
        plans = [("fastmrz", image, variant, base_rotation), ("bottom_region", image, variant, base_rotation)]
        if fallback_image is not None:
            plans.append(("fastmrz", fallback_image, "original", 0 if fallback_image.width >= fallback_image.height else 90))
        else:
            plans.append(("fastmrz", image, variant, (base_rotation + 180) % 360))
        stop_reason = "attempt_limit"
        for index in range(min(len(plans), max(1, max_attempts))):
            method, candidate, image_variant, rotation = plans[index]
            if attempts:
                if usable_mrz_fields(best):
                    stop_reason = "usable_fields"
                    break
                if stop_event is not None and stop_event.is_set():
                    stop_reason = "covered_by_qr_ocr"
                    break
                if (perf_counter() - started) * 1000 >= budget_ms:
                    stop_reason = "time_budget"
                    break
            before = perf_counter()
            try:
                pixels = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2BGR)
                rotated = np.rot90(pixels, -(rotation // 90)).copy()
                raw = self._fast(rotated) if method == "fastmrz" else self._bottom(rotated)
                result = parse_vietnam_td1(raw)
            except Exception:
                # Preserve a partial result when a later Tesseract attempt
                # times out. No OCR exception can suppress the printed text job.
                result = {**parse_vietnam_td1(""), "status": "error"}
            result.update(rotation_clockwise=rotation, image_variant=image_variant)
            attempts.append({"method": method, "variant": image_variant, "rotation_clockwise": rotation,
                             "status": result["status"], "elapsed_ms": round((perf_counter() - before) * 1000, 2)})
            if len(attempts) == 1 or mrz_rank(result) > mrz_rank(best):
                best = result
            if index == 0 and result["status"] == "not_found":
                # No region in this orientation: use the remaining reads on
                # the original image or opposite orientation, including its
                # bottom band, rather than OCRing the same empty band first.
                alternate = (fallback_image, "original", 0 if fallback_image.width >= fallback_image.height else 90) if fallback_image is not None else (image, variant, (base_rotation + 180) % 360)
                plans[1:] = [("fastmrz", *alternate), ("bottom_region", *alternate)]
            if usable_mrz_fields(best):
                stop_reason = "usable_fields"
                break
        best.update(attempts=attempts, stop_reason=stop_reason, budget_ms=budget_ms,
                    elapsed_ms=round((perf_counter() - started) * 1000, 2))
        return best
