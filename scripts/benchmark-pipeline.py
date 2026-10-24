#!/usr/bin/env python3
"""Measure the two-sided API without printing or persisting identity fields."""
import argparse
import json
import math
import mimetypes
import statistics
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8888")
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--back", required=True, type=Path)
    parser.add_argument("--quality", choices=("medium", "high"), default="medium")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests and concurrency must be positive")
    boundary = "benchmark-" + uuid.uuid4().hex
    parts = []
    for key, path in (("front_image", args.front), ("back_image", args.back)):
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts.extend((f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="card"\r\nContent-Type: {mime}\r\n\r\n'.encode(),
                      path.read_bytes(), b"\r\n"))
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="quality"\r\n\r\n{args.quality}\r\n--{boundary}--\r\n'.encode())
    body = b"".join(parts)

    def request(_index):
        start = time.perf_counter()
        try:
            req = urllib.request.Request(args.url.rstrip("/") + "/api/cccd/extract", data=body,
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            with urllib.request.urlopen(req, timeout=300) as response:
                data = json.load(response)
                status = response.status
        except urllib.error.HTTPError as exc:
            data, status = {}, exc.code
            exc.close()
        except (OSError, ValueError):
            data, status = {}, "connection_error"
        stages = dict(data.get("timings_ms", {}))
        for side, result in data.get("ocr", {}).items():
            stages.update({f"{side}.{key}": value for key, value in result.get("timings_ms", {}).items()})
        return {"http": status, "status": data.get("status"), "recognized_fields": data.get("recognized_fields"),
                "wall_ms": round((time.perf_counter() - start) * 1000, 2), "timings_ms": stages,
                "work": {side: result.get("work", {}) for side, result in data.get("ocr", {}).items()}}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        samples = list(pool.map(request, range(args.requests)))
    elapsed = sorted(sample["wall_ms"] for sample in samples)
    keys = {key for sample in samples for key in sample["timings_ms"]}
    report = {"quality": args.quality, "concurrency": args.concurrency, "requests": args.requests,
              "median_wall_ms": statistics.median(elapsed), "p95_wall_ms": elapsed[math.ceil(len(elapsed) * .95) - 1],
              "mean_timings_ms": {key: round(statistics.mean(sample["timings_ms"][key] for sample in samples
                                                            if key in sample["timings_ms"]), 2) for key in sorted(keys)},
              "samples": samples}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
