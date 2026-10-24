"""Stateless GPU worker using the upstream DocU OCR and metadata extractors."""
import logging
import os
import secrets
import sys
import tempfile
from itertools import combinations
from contextlib import asynccontextmanager, closing
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, File, Header, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .engine import OCRBusy, OCRUnavailable, OCRMultipleDocuments
from .gpu_engine import PROVIDER, normalize_result
from .documents import FRONT_FIELDS, BACK_FIELDS, detect_side, text_rows
from .processing import MAX_IMAGE_BYTES, InvalidImage, load_image
from .web import failure
from .regions import deduplicate_documents, nested_regions
from .authority_recognition import refine_authority
from .authority_region import select_authority_region
from .authority_recovery import recover_authority, recovery_bounds
from .authority import authority_word_score, incomplete_authority
from .ocr_debug import authority_crop_preview


logger = logging.getLogger(__name__)
DOCU_REF = "7b98e5ef984f55c9f23f896e67c93c1be0aabddc"
MODEL_PATHS = {
    "segmentation": "segmentation/yolo26m_v2/yolo26m_joint_training/weights/best.pt",
    "rotation": "classification/best_rotation_model.pth",
    "detection": "detection/yolo26s_detect_v2/weights/best.pt",
    "recognition": "recognition/vgg_seq2seq.pth",
    "authority_recognition": "recognition/vgg_seq2seq.pth",
}


class DocULocalEngine:
    def __init__(self):
        self._pipeline = None
        self._authority_recognizer = None
        self._lock = Lock()
        from .work_queue import ModelQueue
        self._queue = ModelQueue("ocr-gpu")
        self._stage_timings = {}
        self._work_counts = {}
        self._needed_fields = None
        self.cuda_available = False
        self.reason = "Đang khởi tạo OCR GPU."
        self.device = "cuda:0"
        self.recognition_model = os.getenv("GPU_OCR_RECOGNITION_MODEL", "vgg_seq2seq")
        self.authority_recognition_model = os.getenv("GPU_OCR_AUTHORITY_MODEL", self.recognition_model)
        for setting, model in (("GPU_OCR_RECOGNITION_MODEL", self.recognition_model),
                               ("GPU_OCR_AUTHORITY_MODEL", self.authority_recognition_model)):
            if model not in {"vgg_transformer", "vgg_seq2seq"}:
                raise ValueError(f"{setting} must be vgg_transformer or vgg_seq2seq")
        root = Path(os.getenv("DOCU_MODEL_DIR", "/app/model"))
        self.models = {key: root / path for key, path in MODEL_PATHS.items()}
        self.models["recognition"] = root / "recognition" / f"{self.recognition_model}.pth"
        self.models["authority_recognition"] = root / "recognition" / f"{self.authority_recognition_model}.pth"

    def describe(self):
        return {"provider": PROVIDER, "revision": DOCU_REF, "device": self.device,
                "recognition_model": self.recognition_model,
                "authority_recognition_model": self.authority_recognition_model,
                "pipeline_ready": self._pipeline is not None, "cuda_available": self.cuda_available,
                "models_loaded": {key: path.is_file() and self._pipeline is not None for key, path in self.models.items()},
                "reason": self.reason, "queue": self._queue.describe()}

    def initialize(self):
        self._queue.run(self._initialize)

    def close(self):
        self._queue.close()

    def _initialize(self):
        import torch

        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            self.reason = "CUDA không truy cập được GPU. Hãy kiểm tra driver và quyền GPU của container."
            raise OCRUnavailable(self.reason)
        missing = [key for key, path in self.models.items() if not path.is_file()]
        if missing:
            self.reason = "Thiếu checkpoint đã huấn luyện: " + ", ".join(missing)
            raise OCRUnavailable(self.reason)
        # Do not let upstream substitute general-purpose YOLO or random rotation
        # weights, or silently select CPU when the caller selected GPU.
        source = Path(os.getenv("DOCU_SOURCE_DIR", "/opt/docu"))
        sys.path.insert(0, str(source))
        from core.pipeline import DocUOCRPipeline

        pipeline = DocUOCRPipeline(
            device=self.device, enable_iqa=True, iqa_strict=False,
            enable_segmentation=True, seg_model_path=str(self.models["segmentation"]),
            enable_rotation=True, rot_model_path=str(self.models["rotation"]),
            det_model_path=str(self.models["detection"]),
            rec_config_name=self.recognition_model, rec_weights_path=str(self.models["recognition"]),
        )
        pipeline.detector.model.to(self.device)
        pipeline.segmentor.strategy.model.to(self.device)
        if self.authority_recognition_model == self.recognition_model:
            authority_recognizer = pipeline.recognizer
        else:
            from core.recognition.vietocr_recognizer import VietOCRRecognizer
            authority_recognizer = VietOCRRecognizer(
                config_name=self.authority_recognition_model,
                weights_path=str(self.models["authority_recognition"]), device=self.device,
            )
        # Execute a CUDA kernel so readiness also catches incompatible runtimes.
        torch.ones(1, device=self.device).add_(1)
        torch.cuda.synchronize()
        self._pipeline = pipeline
        self._authority_recognizer = authority_recognizer
        self.reason = ""

    def _document_payload(self, doc, side, force_extract=False):
        from copy import deepcopy

        extractor = self._pipeline.extractor.extractors[f"cccd_{side}"]
        ordered = extractor.sort_reading_order(doc.recognized_texts, y_tolerance=12.0)
        initial_lines = [{"text": line.text,
                          "box": deepcopy(getattr(getattr(line, "bbox", None), "coordinates", None)),
                          "confidence": getattr(line, "confidence", 0)} for line in ordered] if side == "back" else None
        detected = detect_side([line.text for line in ordered])
        enhancement = None
        search_bbox = None
        if (side == "back" and detected == "back" and
                (self._needed_fields is None or "issuing_authority" in self._needed_fields) and
                getattr(self._pipeline, "recognizer", None) is not None):
            started = perf_counter()
            from core.postprocessing.administrative_corrector import AdministrativeCorrector
            baseline = deepcopy(ordered)
            before = select_authority_region(baseline)
            try:
                original_bounds = recovery_bounds((doc.cropped_image.shape[1], doc.cropped_image.shape[0]), ordered) if doc.cropped_image is not None else None
                enhancement = refine_authority(self._authority_recognizer or self._pipeline.recognizer, doc.cropped_image, ordered,
                                               province_names=AdministrativeCorrector.PROVINCES)
                if doc.cropped_image is not None:
                    recovery = recover_authority(
                        (doc.cropped_image.shape[1], doc.cropped_image.shape[0]), ordered,
                        lambda bounds: self._read_authority_region(doc.cropped_image, bounds),
                        original_bounds,
                    )
                    if recovery:
                        search_bbox = recovery.get("search_bbox")
                        enhancement = enhancement or {"target": "issuing_authority", "reread_lines": 0,
                                                       "changed_lines": 0, "reread_regions": 0}
                        enhancement["recovery_attempted"] = True
                        enhancement["recovered_lines"] = recovery["recovered_lines"]
                        for key in ("reread_regions", "reread_lines", "changed_lines"):
                            enhancement[key] += recovery[key]
            except Exception as exc:
                logger.warning("Authority enhancement failed (%s); preserving initial OCR", type(exc).__name__)
                ordered = baseline
                if enhancement:
                    enhancement["changed_lines"] = enhancement["recovered_lines"] = 0
            after = select_authority_region(ordered)
            rank = lambda value: (bool(value), not incomplete_authority(value), authority_word_score(value))
            if before.text and rank(after.text) < rank(before.text):
                ordered = baseline
                if enhancement:
                    enhancement["changed_lines"] = enhancement["recovered_lines"] = 0
            self._stage_timings["authority_refinement"] = self._stage_timings.get("authority_refinement", 0) + (perf_counter() - started) * 1000
            if enhancement:
                self._work_counts["reread_regions"] = self._work_counts.get("reread_regions", 0) + enhancement.get("reread_regions", 0)
        if force_extract or doc.doc_type in {"unknown", "card"} or (detected == side and doc.doc_type != f"cccd_{side}") or (enhancement and enhancement["changed_lines"]):
            started = perf_counter()
            doc.extracted_fields = extractor.extract(ordered, doc.cropped_image)
            self._stage_timings["metadata"] = self._stage_timings.get("metadata", 0) + (perf_counter() - started) * 1000
        doc_type = f"cccd_{detected}" if detected != "unknown" else "unknown" if force_extract else doc.doc_type
        payload = {
            "success": True, "doc_type": doc_type, "device": self.device,
            "extracted_fields": {key: field.normalized_value for key, field in doc.extracted_fields.items()},
            "field_details": {key: {"is_valid": field.is_valid} for key, field in doc.extracted_fields.items()},
            "text_lines": [{"text": line.text,
                            "box": getattr(getattr(line, "bbox", None), "coordinates", None),
                            "confidence": getattr(line, "confidence", 0)} for line in ordered],
        }
        if side == "back":
            payload["initial_text_lines"] = initial_lines
            if search_bbox:
                payload["authority_search_crop"] = authority_crop_preview(doc.cropped_image, search_bbox)
            region = select_authority_region(payload["text_lines"])
            if region.ambiguous:
                payload["authority_region"] = {"status": "ambiguous", "method": "layout"}
            elif doc.cropped_image is not None:
                metadata = region.metadata((doc.cropped_image.shape[1], doc.cropped_image.shape[0]))
                if metadata:
                    payload["authority_region"] = metadata
        if enhancement:
            payload["enhancement"] = enhancement
        return payload

    def _read_authority_region(self, image, bounds):
        """Reuse the GPU models for one higher-resolution local detection/batch."""
        import math
        from copy import copy

        detector = self._pipeline.detector
        recognizer = self._authority_recognizer or self._pipeline.recognizer
        left, top, right, bottom = bounds
        settings = detector.imgsz, detector.conf_threshold
        try:
            detector.imgsz = max(detector.imgsz, 960)
            detector.conf_threshold = min(detector.conf_threshold, .15)
            detections = detector.detect_single_image(image[top:bottom, left:right])
        except Exception as exc:
            logger.warning("Authority region detection failed (%s); preserving recognized text", type(exc).__name__)
            return [], 0
        finally:
            detector.imgsz, detector.conf_threshold = settings
        patches, items = [], []
        config = recognizer.predictor.config["dataset"]
        original_width = config["image_max_width"]
        try:
            config["image_max_width"] = max(original_width, 1024)
            # A single institution is a few lines. Bound noisy detections and
            # keep whole-card boxes out of text recognition.
            for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
                if detection.field_key in recognizer.WHOLE_CARD_CLASSES:
                    continue
                item = copy(detection)
                item.bbox = copy(detection.bbox)
                item.bbox.coordinates = [(float(x) + left, float(y) + top) for x, y in detection.bbox.coordinates]
                start = min(x for x, y in item.bbox.coordinates)
                end = max(x for x, y in item.bbox.coordinates)
                width = max(1, end - start)
                # The search region is intentionally wide, but each recognition
                # patch remains one tightly cropped line. Padding a short line
                # to the entire search width shrinks its letters at model input.
                padding = .06
                patch = recognizer.crop_oriented_box(image, item.bbox.coordinates, pad_top=.08, pad_bot=.02, pad_x=padding)
                if patch is not None:
                    y1 = min(y for x, y in item.bbox.coordinates)
                    y2 = max(y for x, y in item.bbox.coordinates)
                    item.bbox.coordinates = [(max(0, start - width * padding), max(0, y1 - (y2 - y1) * .08)),
                                             (min(image.shape[1], end + width * padding), max(0, y1 - (y2 - y1) * .08)),
                                             (min(image.shape[1], end + width * padding), min(image.shape[0], y2 + (y2 - y1) * .02)),
                                             (max(0, start - width * padding), min(image.shape[0], y2 + (y2 - y1) * .02))]
                    patches.append(patch)
                    items.append(item)
                if len(patches) >= 12:
                    break
            if patches:
                texts, probabilities = recognizer.predictor.predict_batch(patches, return_prob=True)
                for item, text, probability in zip(items, texts, probabilities):
                    item.text = text.strip() if isinstance(text, str) else ""
                    probability = float(probability)
                    item.confidence = probability if math.isfinite(probability) else 0
                items = items[:min(len(texts), len(probabilities))]
            return items, len(patches)
        except Exception as exc:
            logger.warning("Authority region recognition failed (%s); preserving recognized text", type(exc).__name__)
            return [], len(patches)
        finally:
            config["image_max_width"] = original_width
            for patch in patches:
                patch.close()

    @staticmethod
    def _quality(payload, side, needed_fields=None):
        try:
            fields = normalize_result(payload)["structured_fields"]
        except OCRUnavailable:
            return (-1, 0, 0)
        rows = text_rows(payload)
        detected = detect_side(rows)
        if detected not in {side, "unknown"}:
            return (-1, 0, 0)
        schema = BACK_FIELDS if side == "back" else FRONT_FIELDS
        keys = needed_fields if needed_fields is not None else [key for key, *_ in schema]
        count = sum(bool(fields[key]) and (key != "issuing_authority" or not incomplete_authority(fields[key])) for key in keys)
        # A partial institutional name is still observed data. Prefer keeping
        # it over an otherwise equal crop/rotation with more unrelated rows.
        observed = sum(bool(fields[key]) for key in keys)
        return (int(detected == side), count, observed, len(rows))

    @staticmethod
    def _conflicting_cards(first, second, side):
        first = normalize_result(first)["structured_fields"]
        second = normalize_result(second)["structured_fields"]
        keys = ("issue_date", "identifying_features") if side == "back" else ("id_number", "name", "date_of_birth")
        return any(first[key] and second[key] and first[key] != second[key] for key in keys)

    def _read_full_image(self, image_path, side, degrees):
        import cv2
        from core.datatypes import DocumentInstanceDTO, PipelineContextDTO

        image = cv2.imread(str(image_path)) if isinstance(image_path, (str, Path)) else image_path
        if image is None:
            raise InvalidImage("Không đọc được ảnh cho OCR GPU.")
        rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if degrees:
            image = cv2.rotate(image, rotations[degrees])
        doc = DocumentInstanceDTO(doc_type=f"cccd_{side}", cropped_image=image)
        context = PipelineContextDTO()
        # Reuse the same CUDA detector and recognizer. Bypass card segmentation
        # and the orientation classifier; neither gets to discard this attempt.
        started = perf_counter()
        docs = self._pipeline.detector.process([doc], context)
        self._stage_timings["detection"] = self._stage_timings.get("detection", 0) + (perf_counter() - started) * 1000
        started = perf_counter()
        docs = self._pipeline.recognizer.process(docs, context)
        self._stage_timings["recognition"] = self._stage_timings.get("recognition", 0) + (perf_counter() - started) * 1000
        self._work_counts["primary_regions"] += sum(len(doc.recognized_texts) for doc in docs)
        return self._document_payload(docs[0], side, force_extract=True) if docs else None

    def _collect_metrics(self, context):
        names = {"ImageQualityAssessmentFilter": "iqa", "DocumentSegmentationFilter": "segmentation",
                 "RotationCorrectionFilter": "rotation", "YOLO26OBBDetector": "detection",
                 "VietOCRRecognizer": "recognition", "IDCardRuleExtractor": "metadata"}
        for name, item in getattr(context, "execution_metrics", {}).items():
            if name in names:
                key = names[name]
                self._stage_timings[key] = self._stage_timings.get(key, 0) + item.get("duration_ms", 0)

    def extract_document(self, image_path, side, needed_fields=None):
        result, timings = self._queue.run(self._extract_document, str(image_path), side, needed_fields)
        result.setdefault("timings_ms", {}).update(timings)
        return result

    def extract_image(self, image, side, needed_fields=None, source_data=None):
        result, timings = self._queue.run(self._extract_pixels, image, side, needed_fields)
        result.setdefault("timings_ms", {}).update(timings)
        return result

    def _extract_pixels(self, image, side, needed_fields):
        import cv2
        import numpy as np
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return self._extract_document(frame, side, needed_fields)

    def _extract_document(self, image_path, side, needed_fields=None):
        if not self._lock.acquire(blocking=False):
            raise OCRBusy("Dịch vụ OCR GPU đang xử lý ảnh khác. Hãy thử lại sau ít giây.")
        try:
            if self._pipeline is None:
                raise OCRUnavailable(self.reason)
            self._stage_timings = {}
            self._work_counts = {"primary_regions": 0, "reread_regions": 0, "full_image_attempts": 1}
            self._needed_fields = needed_fields
            result = self._pipeline.process_image(image_path, doc_type_hint=f"cccd_{side}")
            self._collect_metrics(getattr(result, "context", None))
            if not result.success:
                raise OCRUnavailable("Pipeline OCR GPU chưa xử lý được ảnh.")
            documents = deduplicate_documents(result.documents)
            self._work_counts["primary_regions"] = sum(len(doc.recognized_texts) for doc in result.documents)
            candidates = [(doc, self._document_payload(doc, side)) for doc in documents]
            # Two masks alone are not proof of two cards. Require independent
            # regions with readable card-side labels and multiple valid fields.
            minimum = len(BACK_FIELDS) if side == "back" else 4
            credible = [(doc, payload) for doc, payload in candidates if self._quality(payload, side)[:2] >= (1, minimum)]
            if any(not nested_regions(first[0], second[0]) and self._conflicting_cards(first[1], second[1], side)
                   for first, second in combinations(credible, 2)):
                logger.warning("GPU found distinct readable cards; side=%s detected_regions=%d", side, len(credible))
                raise OCRMultipleDocuments(len(credible))
            empty = {"success": True, "doc_type": "unknown", "extracted_fields": {}, "text_lines": []}
            grade = lambda payload: self._quality(payload, side, needed_fields)
            best = max((payload for _doc, payload in candidates), key=grade, default=empty)
            maximum = len(needed_fields) if needed_fields is not None else len(BACK_FIELDS if side == "back" else FRONT_FIELDS)
            quality = grade(best)
            if len(documents) == 1 and (quality[:2] == (1, maximum) or quality[0] == -1
                                        or (quality[0] == 1 and best.get("authority_region", {}).get("status") == "ambiguous")):
                return self._profiled(best)
            attempted = []
            method, rotation = "segmented", None
            original_payload = None
            if not isinstance(image_path, (str, Path)):
                import numpy as np
                original_payload = next((payload for doc, payload in candidates
                                         if getattr(doc.cropped_image, "shape", None) == image_path.shape
                                         and np.array_equal(doc.cropped_image, image_path)), None)
            started = perf_counter()
            for degrees in (0, 180, 90, 270):
                attempted.append(degrees)
                if degrees == 0 and original_payload is not None:
                    payload = original_payload
                    self._work_counts["skipped_duplicate_passes"] = 1
                else:
                    self._work_counts["full_image_attempts"] += 1
                    payload = self._read_full_image(image_path, side, degrees)
                if payload is not None and (grade(payload) > grade(best)
                                            or (method == "segmented" and grade(payload) == grade(best))):
                    best, method, rotation = payload, "full_image", degrees
                if grade(best)[:2] == (1, maximum):
                    break
                if side == "back" and payload and payload.get("enhancement", {}).get("recovery_attempted"):
                    requested = needed_fields if needed_fields is not None else [key for key, *_ in BACK_FIELDS]
                    fields = normalize_result(payload)["structured_fields"]
                    if (grade(payload)[0] == 1 and "issuing_authority" in requested
                            and incomplete_authority(fields["issuing_authority"])
                            and all(fields[key] for key in requested if key != "issuing_authority")):
                        # The side/orientation is already readable. More whole-
                        # image rotations cannot justify completing a cut name.
                        break
            best["recovery"] = {"method": method, "detected_regions": len(documents),
                                "rotation_degrees": rotation, "attempted_rotations": attempted}
            logger.warning("GPU OCR recovery; side=%s regions=%d method=%s rotation=%s fields=%d",
                           side, len(documents), method, rotation, self._quality(best, side)[1])
            self._stage_timings["recovery"] = (perf_counter() - started) * 1000
            return self._profiled(best)
        finally:
            self._lock.release()

    def _profiled(self, payload):
        payload["timings_ms"] = {key: round(value, 2) for key, value in self._stage_timings.items()}
        payload["work"] = dict(self._work_counts)
        return payload


def create_gpu_app(engine=None, restoration_engine=None):
    engine = engine if engine is not None else DocULocalEngine()
    from .qr_restoration_worker import QRRestorationEngine, add_restoration_routes
    restoration = restoration_engine if restoration_engine is not None else QRRestorationEngine(queue=getattr(engine, "_queue", None))

    @asynccontextmanager
    async def lifespan(_app):
        try:
            await run_in_threadpool(engine.initialize)
        except Exception:
            logger.exception("GPU worker initialization failed")
        yield
        if hasattr(engine, "close"):
            engine.close()
        restoration.close()

    app = FastAPI(title="Identification OCR GPU worker", version="0.6.6", lifespan=lifespan)
    token = os.getenv("GPU_OCR_API_KEY", "")
    add_restoration_routes(app, restoration, token)

    @app.middleware("http")
    async def no_store(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/v1/health/ready")
    def health():
        info = engine.describe()
        return JSONResponse(info, status_code=200 if info["pipeline_ready"] else 503)

    @app.post("/api/v1/extract")
    async def extract(file: UploadFile = File(...), doc_type_hint: Literal["cccd_front", "cccd_back"] = "cccd_front",
                      authorization: str | None = Header(None), fields: str | None = None):
        try:
            if token and not secrets.compare_digest(authorization or "", f"Bearer {token}"):
                return failure("Không có quyền truy cập OCR GPU.", 401, "gpu")
            data = await file.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                return failure("Mỗi ảnh tối đa 10 MB.", 413, "gpu")
            side = doc_type_hint.removeprefix("cccd_")
            needed = fields.split(",") if fields is not None else None
            allowed = {key for key, *_ in (BACK_FIELDS if side == "back" else FRONT_FIELDS)}
            if needed is not None and (not needed or not set(needed) <= allowed):
                raise InvalidImage("Danh sách trường OCR không hợp lệ.")
            return await run_in_threadpool(process_upload, data, engine, side, needed)
        except OCRMultipleDocuments as exc:
            return JSONResponse({"success": False, "error": str(exc), "error_code": exc.code,
                                 "detected_regions": exc.detected_regions}, status_code=422)
        except InvalidImage as exc:
            return failure(str(exc), 422, "gpu")
        except OCRBusy as exc:
            return failure(str(exc), 429, "gpu", {"Retry-After": "5"})
        except OCRUnavailable as exc:
            return failure(str(exc), 503, "gpu")
        except Exception:
            logger.exception("GPU extraction failed")
            return failure("Lỗi xử lý OCR GPU.", 503, "gpu")
        finally:
            await file.close()

    return app


def process_upload(data, engine, side, needed_fields=None):
    started = perf_counter()
    with closing(load_image(data)) as image:
        decoded_ms = (perf_counter() - started) * 1000
        if hasattr(engine, "extract_image"):
            result = engine.extract_image(image, side, needed_fields=needed_fields)
        else:
            with tempfile.TemporaryDirectory(prefix="cccd-gpu-") as directory:
                path = Path(directory) / "card.png"
                image.save(path)
                result = engine.extract_document(path, side)
    result.setdefault("timings_ms", {})["worker_image_decode"] = round(decoded_ms, 2)
    return result


app = create_gpu_app()
