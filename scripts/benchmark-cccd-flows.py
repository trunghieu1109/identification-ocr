#!/usr/bin/env python3
"""Compare deployed GPU and experiment APIs sequentially on identical images.

Reports timings and field match counts only; never saves identity values,
OCR text or response previews. Dataset paths are relative to the manifest.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import statistics
from pathlib import Path
from time import perf_counter
import urllib.error
import urllib.request
from uuid import uuid4


FLOWS = {"gpu": "/api/cccd/extract", "experiment": "/api/cccd/experiment/extract"}


def multipart(front, back, flow):
    boundary = "flow-benchmark-" + uuid4().hex
    parts = []
    for key, data in (("front_image", front), ("back_image", back)):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="card.png"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
        parts.extend((data, b"\r\n"))
    options = {"quality": "high"} if flow == "gpu" else {"align": "true"}
    for key, value in options.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def measure(opener, url, front, back, flow, expected):
    body, boundary = multipart(front, back, flow)
    req = urllib.request.Request(url + FLOWS[flow], body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = perf_counter()
    try:
        with opener.open(req, timeout=300) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
        exc.close()
    wall_ms = (perf_counter() - started) * 1000
    data = json.loads(raw)
    if status != 200 or data.get("success") is not True:
        raise RuntimeError(f"{flow}: HTTP {status}, benchmark stopped")
    times = data.get("timings_ms", {})
    groups = {key: sum(value for name, value in times.items() if name.startswith(prefix))
              for key, prefix in (("alignment", "align_"), ("mrz", "mrz_"), ("qr", "qr_"), ("ocr", "ocr_"))}
    # In parallel mode stage durations overlap; their sum is not wall time.
    intervals = sorted((item["start_ms"], item["end_ms"]) for item in data.get("stage_intervals_ms", {}).values())
    covered, finish = 0, 0
    for start, end in intervals:
        covered += max(0, end - max(finish, start))
        finish = max(finish, end)
    groups["other"] = max(0, times.get("total", 0) - (covered if intervals else sum(groups.values())))
    ocr = ([stage for side in data.get("sides", {}).values() for stage in side.get("ocr", [])]
           if flow == "experiment" else list(data.get("ocr", {}).values()))
    fields = data["fields"]
    return {"wall_ms": round(wall_ms, 2), "pipeline_ms": times.get("total"),
            "stage_ms": {key: round(value, 2) for key, value in groups.items()}, "timings_ms": times,
            "worker_queue_ms": data.get("worker_queue_ms", 0),
            "gpu_queue_ms": round(sum(stage.get("timings_ms", {}).get("queue_wait", 0) for stage in ocr), 2),
            "response_bytes": len(raw), "upload_bytes": len(body), "status": data.get("status"),
            "recognized_fields": data["recognized_fields"],
            "expected_fields": len(expected), "exact_expected_matches": sum(fields.get(key) == value for key, value in expected.items()),
            "mismatched_field_names": [key for key, value in expected.items() if fields.get(key) != value],
            "field_names": [key for key, value in fields.items() if value],
            "ocr_passes": len(ocr), "ocr_errors": sum(stage.get("status") in {"error", "unavailable", "wrong_side"} for stage in ocr),
            "alignment": {side: value["alignment"]["status"] for side, value in data.get("sides", {}).items()},
            "mrz_status": data.get("sides", {}).get("back", {}).get("mrz", {}).get("status"),
            "mrz_attempts": data.get("sides", {}).get("back", {}).get("mrz", {}).get("attempts", []),
            "mrz_stop_reason": data.get("sides", {}).get("back", {}).get("mrz", {}).get("stop_reason"),
            "mrz_used_fields": data.get("sides", {}).get("back", {}).get("mrz", {}).get("used_fields", []),
            "execution": data.get("execution"), "stage_intervals_ms": data.get("stage_intervals_ms", {}),
            "qr_status": data.get("qr", {}).get("status"), "conflict_fields": [item["field"] for item in data.get("conflicts", [])]}


def summarize(samples):
    return {"samples": len(samples), "median_wall_ms": round(statistics.median(s["wall_ms"] for s in samples), 2),
            "mean_wall_ms": round(statistics.mean(s["wall_ms"] for s in samples), 2),
            "min_wall_ms": min(s["wall_ms"] for s in samples), "max_wall_ms": max(s["wall_ms"] for s in samples),
            "mean_stage_ms": {key: round(statistics.mean(s["stage_ms"][key] for s in samples), 2) for key in samples[0]["stage_ms"]},
            "mean_gpu_queue_ms": round(statistics.mean(s["gpu_queue_ms"] for s in samples), 2),
            "mean_response_bytes": round(statistics.mean(s["response_bytes"] for s in samples)),
            "recognized_fields_range": [min(s["recognized_fields"] for s in samples), max(s["recognized_fields"] for s in samples)],
            "exact_expected_matches_range": [min(s["exact_expected_matches"] for s in samples), max(s["exact_expected_matches"] for s in samples)],
            "mismatched_field_names": sorted({key for s in samples for key in s["mismatched_field_names"]}),
            "ocr_passes_range": [min(s["ocr_passes"] for s in samples), max(s["ocr_passes"] for s in samples)],
            "ocr_errors": sum(s["ocr_errors"] for s in samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8888")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0:
        parser.error("requests must be positive; warmup must be nonnegative")
    dataset = json.loads(args.dataset.read_text())
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "synthetic": dataset.get("synthetic", False),
              "requests_per_flow_per_case": args.requests, "warmups_per_flow_per_case": args.warmup,
              "concurrency": 1, "order": "alternate flow order each measured round", "cases": []}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for case in dataset["cases"]:
        front = (args.dataset.parent / case["front"]).read_bytes()
        back = (args.dataset.parent / case["back"]).read_bytes()
        expected = case.get("expected", {})
        entry = {"name": case["name"], "input_sha256": {"front": hashlib.sha256(front).hexdigest(), "back": hashlib.sha256(back).hexdigest()},
                 "input_bytes": len(front) + len(back), "expected_fields": len(expected), "warmups": {}, "samples": {key: [] for key in FLOWS}}
        for flow in FLOWS:
            entry["warmups"][flow] = [measure(opener, args.url.rstrip("/"), front, back, flow, expected) for _ in range(args.warmup)]
        for index in range(args.requests):
            order = list(FLOWS) if index % 2 == 0 else list(reversed(FLOWS))
            for flow in order:
                sample = measure(opener, args.url.rstrip("/"), front, back, flow, expected)
                entry["samples"][flow].append(sample)
                print(json.dumps({"case": case["name"], "flow": flow, "round": index + 1,
                                  "wall_ms": sample["wall_ms"], "recognized_fields": sample["recognized_fields"]}), flush=True)
        entry["summary"] = {key: summarize(samples) for key, samples in entry["samples"].items()}
        gpu, new = [entry["summary"][key]["median_wall_ms"] for key in FLOWS]
        entry["delta_median_ms"] = round(new - gpu, 2)
        entry["delta_median_percent"] = round((new / gpu - 1) * 100, 2)
        report["cases"].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
