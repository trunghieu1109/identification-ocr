"""Experimental QR → validated MRZ → printed OCR reconciliation only."""
import base64
import io
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from time import perf_counter

from .documents import BACK_FIELDS, FRONT_FIELDS, fold
from .engine import OCRBusy, OCRUnavailable
from .pipeline import CARD_FIELDS, OPTIONAL_FIELDS
from .processing import load_image, process_image, InvalidImage
from .qr import parse_cccd_qr
from .experiment_mrz import usable_mrz_fields


def _equivalent(key, first, second):
    if key in {"name", "nationality", "gender"}:
        return " ".join(fold(first).split()) == " ".join(fold(second).split())
    return first == second


def reconcile(candidates):
    fields, sources = dict.fromkeys(CARD_FIELDS, ""), dict.fromkeys(CARD_FIELDS)
    conflicts, review = [], []
    for key in CARD_FIELDS:
        values = [candidate for candidate in candidates if candidate["field"] == key and candidate["value"]]
        values.sort(key=lambda item: (item["priority"], item.get("order", 0)))
        if not values:
            continue
        chosen = values[0]
        # A conflicting verified ID indicates different cards or a bad read.
        # Preserve all evidence, but never combine identities automatically.
        disagreement = [item for item in values[1:] if not _equivalent(key, chosen["value"], item["value"])]
        if disagreement:
            conflicts.append({"field": key, "selected_source": chosen["source"], "candidates": values})
            review.append(key)
        if key == "name" and chosen["source"].startswith("mrz"):
            # Preserve the accented printed name when both readings agree.
            printed = next((item for item in values if item["source"].startswith("ocr")
                            and _equivalent(key, chosen["value"], item["value"])), None)
            if printed:
                chosen = printed
        fields[key], sources[key] = chosen["value"], chosen["source"]
        if chosen.get("check") == "not_covered_by_checksum":
            review.append(key)
    identity_conflict = any(item["field"] == "id_number" and len({candidate["value"] for candidate in item["candidates"]
        if candidate["priority"] <= 1}) > 1 for item in conflicts)
    if identity_conflict:
        fields, sources = dict.fromkeys(CARD_FIELDS, ""), dict.fromkeys(CARD_FIELDS)
    return fields, sources, conflicts, sorted(set(review)), identity_conflict


def _preview(image):
    with image.copy() as small:
        small.thumbnail((1100, 800))
        buffer = io.BytesIO()
        small.save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")



def process_experiment(images, engine, qr_reader, aligner, mrz_reader, *, align=True, rotations=None):
    started = perf_counter()
    candidates, warnings, stages, timings, intervals = [], [], {}, {}, {}
    rotations = rotations or {}
    telemetry_lock, mrz_stop = Lock(), Event()
    mrz_keys = {"id_number", "name", "date_of_birth", "gender", "nationality", "expiry_date"}

    def add(fields, source, priority, checks=None):
        for key, value in fields.items():
            if key in CARD_FIELDS and value:
                candidates.append({"field": key, "value": value, "source": source, "priority": priority,
                                   "order": len(candidates), "check": (checks or {}).get(key)})

    def timed(label, function, *args, **kwargs):
        before = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            after = perf_counter()
            with telemetry_lock:
                timings[label] = round((after - before) * 1000, 2)
                intervals[label] = {"start_ms": round((before - started) * 1000, 2),
                                    "end_ms": round((after - started) * 1000, 2)}

    def known_values(qr_data, mrz=None):
        values = {}
        for fields, _source in qr_data["values"]:
            for key, value in fields.items():
                if value:
                    values.setdefault(key, value)
        for key, value in usable_mrz_fields(mrz).items() if mrz else ():
            values.setdefault(key, value)
        return values

    with ExitStack() as stack:
        # A bad second upload never starts any model or external OCR request.
        originals = {side: stack.enter_context(load_image(images[side])) for side in ("front", "back")}
        selected = {}
        for side in ("front", "back"):
            original = originals[side]
            rotation = rotations.get(side, 0)
            if rotation:
                original = originals[side] = stack.enter_context(original.rotate(-rotation, expand=True))
            stage = stages[side] = {"input_size": list(original.size), "rotation_clockwise": rotation, "ocr": []}
            if align:
                try:
                    image, alignment = timed(f"align_{side}", aligner.align, original)
                    stack.callback(image.close)
                except Exception:
                    image, alignment = original, {"status": "error", "corners": []}
                    warnings.append(f"DocAligner {side} gặp lỗi; đã dùng ảnh gốc.")
            else:
                image, alignment = original, {"status": "disabled", "corners": []}
            selected[side], stage["alignment"] = image, alignment

        def read_qr():
            data = {"values": [], "sides": {}, "warnings": []}
            for side in ("front", "back"):
                results = data["sides"][side] = []
                variants = [("original", originals[side])]
                if stages[side]["alignment"]["status"] == "aligned":
                    variants.append(("aligned", selected[side]))
                found = False
                for variant, image in variants:
                    if found and variant == "aligned":
                        continue
                    try:
                        texts = timed(f"qr_{side}_{variant}", qr_reader.read, image)
                        parsed = [value for text in sorted(set(texts)) if (value := parse_cccd_qr(text))]
                        found = bool(parsed)
                        results.append({"variant": variant, "status": "found" if parsed else "not_found"})
                        data["values"].extend((fields, f"qr_{side}_{variant}") for fields in parsed)
                    except Exception:
                        results.append({"variant": variant, "status": "error"})
                        data["warnings"].append(f"Chưa xử lý được QR {side}/{variant}.")
            return data

        def read_mrz():
            aligned = stages["back"]["alignment"]["status"] == "aligned"
            try:
                result = timed("mrz_back", mrz_reader.read, selected["back"], stop_event=mrz_stop,
                               fallback_image=originals["back"] if aligned else None,
                               variant="aligned" if aligned else "original")
            except Exception:
                result = {"status": "error", "fields": {}, "checks": {}, "warnings": [
                    "FastMRZ chưa xử lý được ảnh mặt sau; vẫn dùng OCR đầy đủ khi cần."]}
            result["used_fields"] = list(usable_mrz_fields(result))
            return result

        def ocr_image(side, image, variant, needed):
            try:
                result = timed(f"ocr_{side}_{variant}", process_image, image, engine, side,
                               needed_fields=needed, include_expiry=True)
                return {"variant": variant, "requested_fields": needed, **result}
            except InvalidImage:
                return {"variant": variant, "requested_fields": needed, "status": "wrong_side", "fields": {}}
            except (OCRBusy, OCRUnavailable):
                return {"variant": variant, "requested_fields": needed, "status": "unavailable", "fields": {}}
            except Exception:
                return {"variant": variant, "requested_fields": needed, "status": "error", "fields": {}}

        def read_ocr(qr_future, mrz_future):
            # Origin/authority require visual OCR even with complete QR/MRZ.
            # Start the first printed pass immediately; merging still waits
            # for higher-priority sources, independent of completion order.
            qr_data = qr_future.result() if qr_future.done() else {"values": []}
            results, values, views = {}, known_values(qr_data), {}
            for side, schema in (("front", FRONT_FIELDS), ("back", BACK_FIELDS)):
                mrz = mrz_future.result() if mrz_future.done() else None
                for key, value in known_values(qr_data, mrz).items():
                    values[key] = value
                needed = [key for key, *_ in schema if not values.get(key)]
                if not values.get("expiry_date"):
                    needed.append("expiry_date")
                image = selected[side]
                label = "aligned" if stages[side]["alignment"]["status"] == "aligned" else "original"
                rotation = 0
                if side == "back" and mrz and usable_mrz_fields(mrz):
                    if mrz.get("image_variant") == "original" and label == "aligned":
                        image, label = originals[side], "original_mrz_fallback"
                    rotation = mrz.get("rotation_clockwise", 0)
                if not needed:
                    results[side], views[side] = [], (label, rotation)
                    continue
                with ExitStack() as local:
                    if rotation:
                        image = local.enter_context(image.rotate(-rotation, expand=True))
                    result = ocr_image(side, image, label, needed)
                results[side] = [result]
                views[side] = (label, rotation)
                for key, value in result["fields"].items():
                    if value:
                        values.setdefault(key, value)
                if (stages[side]["alignment"]["status"] == "aligned" and label == "aligned"
                        and result.get("recognized_fields", 0) < (4 if side == "front" else 2)):
                    fallback = ocr_image(side, originals[side], "original_fallback", needed)
                    results[side].append(fallback)
                    for key, value in fallback["fields"].items():
                        if value:
                            values.setdefault(key, value)
                # Cancel extra MRZ attempts only after all its possible fields
                # are supplied elsewhere. Its first actual read is always kept.
                qr_data = qr_future.result()
                values.update(known_values(qr_data))
                qr_has_identity = any(fields.get("id_number") for fields, _source in qr_data["values"])
                if qr_has_identity and mrz_keys <= {key for key, value in values.items() if value}:
                    mrz_stop.set()
            return results, views

        # Each task owns its results. Immutable, loaded pixels stay alive until
        # all futures finish; no detached inference or model re-entry on return.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="cccd-lab") as pool:
            qr_future = pool.submit(read_qr)
            mrz_future = pool.submit(read_mrz)
            ocr_future = pool.submit(read_ocr, qr_future, mrz_future)
            try:
                qr_data = qr_future.result()
                mrz = mrz_future.result()
                ocr_data, views = ocr_future.result()
            finally:
                mrz_stop.set()

        stages["back"]["mrz"] = mrz
        # A late MRZ orientation result must not leave a failed parallel OCR
        # pass upside down. Retry only when required back fields are missing.
        back_rotation = mrz.get("rotation_clockwise", 0)
        back_values = known_values(qr_data, mrz)
        for result in ocr_data["back"]:
            for key, value in result["fields"].items():
                if value:
                    back_values.setdefault(key, value)
        if (usable_mrz_fields(mrz) and back_rotation and views["back"][1] != back_rotation
                and any(not back_values.get(key) for key, *_ in BACK_FIELDS if key not in OPTIONAL_FIELDS)):
            image = originals["back"] if mrz.get("image_variant") == "original" else selected["back"]
            label = ("original" if image is originals["back"] else "aligned") + "_mrz_rotation"
            needed = [key for key, *_ in BACK_FIELDS if not back_values.get(key)]
            if not back_values.get("expiry_date"):
                needed.append("expiry_date")
            with image.rotate(-back_rotation, expand=True) as corrected:
                ocr_data["back"].append(ocr_image("back", corrected, label, needed))
            views["back"] = (label, back_rotation)
        warnings.extend(qr_data["warnings"])
        warnings.extend(mrz.get("warnings", []))
        for side in ("front", "back"):
            stages[side]["qr"] = qr_data["sides"][side]
            stages[side]["ocr"] = ocr_data[side]
        for fields, source in qr_data["values"]:
            add(fields, source, 0)
        add(usable_mrz_fields(mrz), "mrz_back", 1, mrz.get("field_checks"))
        primary, primary_sources, *_ = reconcile(candidates)
        # Use all printed fields if QR/MRZ fail. Otherwise only fill gaps, with
        # cross-side ID checking and free accent restoration from cached OCR.
        for side, results in ocr_data.items():
            for result in results:
                source = f"ocr_{side}_{result['variant']}"
                for key, value in result["fields"].items():
                    if not value:
                        continue
                    missing = not primary.get(key)
                    check_id = key == "id_number" and primary.get(key) and primary[key] != value
                    accent = (key == "name" and primary_sources.get(key) == "mrz_back"
                              and _equivalent(key, primary[key], value))
                    if missing or check_id or accent:
                        add({key: value}, source, 2)
                if result["status"] in {"wrong_side", "unavailable", "error"}:
                    warnings.append(f"OCR {side}/{result['variant']} chưa đọc được; giữ dữ liệu từ nguồn khác.")

        fields, sources, conflicts, review, identity_conflict = reconcile(candidates)
        if mrz.get("checks", {}).get("composite") and mrz.get("checks", {}).get("date_of_birth") and mrz.get("birth_date_yymmdd"):
            from datetime import datetime
            birth = fields.get("date_of_birth")
            try:
                birth_conflict = bool(birth) and datetime.strptime(birth, "%d/%m/%Y").strftime("%y%m%d") != mrz["birth_date_yymmdd"]
            except ValueError:
                birth_conflict = True
            if birth_conflict:
                warnings.append("Ngày sinh QR/OCR khác ngày sinh MRZ; cần đối chiếu thẻ.")
                mrz["birth_date_conflict"] = True
                review.append("date_of_birth")
        if identity_conflict:
            warnings.append("Có nhiều số căn cước khác nhau từ QR/MRZ. Chưa ghép thông tin; hãy kiểm tra hai ảnh cùng thẻ.")
        if conflicts:
            warnings.append("Có thông tin đọc khác nhau giữa các nguồn; cần đối chiếu các trường được đánh dấu.")
        for side in ("front", "back"):
            label, rotation = views[side]
            image = originals[side] if label.startswith("original") else selected[side]
            with ExitStack() as local:
                if rotation:
                    image = local.enter_context(image.rotate(-rotation, expand=True))
                stages[side]["preview"] = _preview(image)
            if label == "original_mrz_fallback":
                stages[side]["ocr_image"] = label
            for result in stages[side]["ocr"]:
                source = f"ocr_{side}_{result['variant']}"
                result["filled_fields"] = [key for key, value in sources.items() if value == source]
            stages[side]["ocr_skipped"] = not bool(stages[side]["ocr"])

    missing = [key for key in CARD_FIELDS if not fields[key] and key not in OPTIONAL_FIELDS]
    total = round((perf_counter() - started) * 1000, 2)
    timings["total"] = total
    return {"success": True, "experimental": True, "source": "qr_mrz_ocr", "quality": "high",
            "status": "identity_conflict" if identity_conflict else "review_required" if review else "partial" if missing else "complete",
            "fields": fields, "field_sources": sources, "candidates": candidates, "conflicts": conflicts,
            "review_fields": sorted(set(review)), "missing_fields": missing, "sides": stages,
            "recognized_fields": sum(bool(value) for value in fields.values()), "total_fields": len(CARD_FIELDS),
            "warnings": list(dict.fromkeys(warnings)), "timings_ms": timings, "stage_intervals_ms": intervals,
            "processing_time_ms": round(total), "execution": {
                "mode": "parallel_qr_mrz_ocr", "merge_priority": ["qr", "mrz", "ocr"],
                "ocr_policy": "fill_missing_with_full_image_fallback", "gpu_recognition_scope": "full_image",
                "mrz_attempt_limit": 3, "mrz_retry_budget_ms": 900}}
