"""Experimental DocU worker: field-guided recovery with the existing GPU models."""
from .documents import BACK_FIELDS, FRONT_FIELDS, detect_side, text_rows
from .field_recovery import STRATEGY, recover_date
from .gpu_worker import DocULocalEngine, create_gpu_app
from .namlhn_text_processing import read_text_region
from .ocr_debug import authority_crop_preview
from .authority_region import _line
from .expiry_context import STRATEGY as EXPANDED_STRATEGY, PROCESSING as CONTEXT_PROCESSING, context_strips
from .field_recovery import _unassigned_dates


class DocURegionLocalEngine(DocULocalEngine):
    def __init__(self):
        super().__init__()
        self._expanded_active = False

    def extract_expanded_image(self, image, side, needed_fields=None):
        result, timings = self._queue.run(self._extract_expanded_pixels, image, side, needed_fields)
        result.setdefault('timings_ms', {}).update(timings)
        return result

    def _extract_expanded_pixels(self, image, side, needed_fields):
        self._expanded_active = True
        try:
            return self._extract_pixels(image, side, needed_fields)
        finally:
            self._expanded_active = False

    def describe(self):
        return {**super().describe(), "strategy": STRATEGY}

    def _extract_document(self, image_path, side, needed_fields=None):
        self._date_recovery_attempted = False
        self._region_reads = []
        self._region_attempts = set()
        needed = list(needed_fields) if needed_fields is not None else [
            key for key, *_ in (BACK_FIELDS if side == "back" else FRONT_FIELDS)]
        if side == "front" and "expiry_date" not in needed:
            needed.append("expiry_date")
        self._field_needed = needed
        # Expiry gets one local read before evaluating the candidate. Missing
        # expiry alone must not trigger the base worker's full-image rotations.
        return super()._extract_document(image_path, side, [key for key in needed if key != "expiry_date"])

    def _document_payload(self, doc, side, force_extract=False):
        self._region_seeds = getattr(doc, "recognized_texts", [])
        self._region_target = "issuing_authority"
        payload = super()._document_payload(doc, side, force_extract=force_extract)
        payload["region_debug"] = list(getattr(self, "_region_reads", []))
        if doc.cropped_image is None or detect_side(text_rows(payload)) != side:
            return payload
        if getattr(self, "_date_recovery_attempted", False):
            return payload
        # The experiment uses the same Seq2Seq model for general text and local
        # recovery, and the same bounded region reader as authority recovery.
        self._region_target = "expiry_date" if side == "front" else "issue_date"
        rows, metadata = recover_date(
            payload["text_lines"], (doc.cropped_image.shape[1], doc.cropped_image.shape[0]), side,
            lambda bounds: self._read_authority_region(doc.cropped_image, bounds), self._field_needed,
            expanded_context=self._expanded_active,
        )
        if metadata["status"] != "skipped":
            self._date_recovery_attempted = True
        payload["text_lines"] = rows
        if self._expanded_active:
            metadata["strategy"] = EXPANDED_STRATEGY
        payload["field_recovery"] = metadata
        self._stage_timings["field_refinement"] = self._stage_timings.get("field_refinement", 0) + metadata["elapsed_ms"]
        self._work_counts["reread_regions"] += metadata["reread_regions"]
        payload["region_debug"] = list(getattr(self, "_region_reads", []))
        return payload

    def _read_authority_region(self, image, bounds):
        # A region already read for this image/target is not processed twice by
        # the base authority refinement and recovery stages.
        key = (self._region_target, tuple(bounds), image.shape)
        if key in self._region_attempts or len(self._region_attempts) >= 3:
            return [], 0
        self._region_attempts.add(key)
        extras = []
        if self._expanded_active and self._region_target == 'expiry_date':
            dates = _unassigned_dates(self._region_seeds)
            if len(dates) == 1:
                extras = context_strips(dates[0], (image.shape[1], image.shape[0]))
        items, count = read_text_region(self._pipeline.detector, self._pipeline.recognizer,
                                       image, bounds, getattr(self, "_region_seeds", ()), extra_boxes=extras)
        self._region_reads.append({
            "target": self._region_target, "processing": CONTEXT_PROCESSING if extras else "namlhn_preprocess_and_padding",
            "expanded_boxes": [list(box) for box in extras],
            "search_crop": authority_crop_preview(image, bounds),
            "text_lines": [{"text": _line(item)[0], "confidence": _line(item)[2],
                            "bbox": list(_line(item)[1]) if _line(item)[1] else None} for item in items],
        })
        return items, count


class ExpandedView:
    """Route v3 to the same loaded models; set its mode only inside the FIFO."""
    def __init__(self,engine):
        self.engine = engine
        self._queue = engine._queue

    def initialize(self):
        self.engine.initialize()

    def close(self):
        pass  # The root application owns the shared engine.

    def describe(self):
        return {**self.engine.describe(),'strategy':EXPANDED_STRATEGY}

    def extract_image(self,image,side,needed_fields=None):
        return self.engine.extract_expanded_image(image,side,needed_fields)


_engine = DocURegionLocalEngine()
app = create_gpu_app(_engine, extra_front_fields=("expiry_date",))
app.mount('/expanded',create_gpu_app(ExpandedView(_engine),extra_front_fields=('expiry_date',)))
