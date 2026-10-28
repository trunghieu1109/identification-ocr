"""Display the actual back-side OCR lines before metadata extraction."""
from .authority_region import _line, select_authority_region


def back_ocr_debug(initial, final):
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
    return result
