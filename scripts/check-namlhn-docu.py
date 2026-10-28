#!/usr/bin/env python3
"""Measure exact fields on a labelled synthetic fixture manifest via the OCR API."""
import argparse
import json
import statistics
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import httpx


def normalized(value):
    # Preserve every diacritic; only normalize Unicode, case and whitespace.
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--qualities", nargs="+", choices=("medium", "high"), default=["high"])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads(args.dataset.read_text())
    if manifest.get("synthetic") is not True:
        parser.error("This check stores OCR outputs and only accepts explicitly synthetic fixtures")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    runs = []
    with httpx.Client(base_url=args.base_url, timeout=180, trust_env=False) as client:
        for pair in manifest["pairs"]:
            files = {f"{side}_image": (f"{side}.png", (args.dataset.parent / pair[side]).read_bytes(), "image/png")
                     for side in ("front", "back")}
            expected = {key: value for fields in pair.get("expected", {}).values() for key, value in fields.items()}
            for engine in args.qualities:
                for repeat in range(-1, args.repeats):
                    response = client.post("/api/cccd/extract", files=files, data={"quality": engine})
                    response.raise_for_status()
                    result = response.json()
                    if result.get("success") is not True:
                        raise RuntimeError(f"OCR failed: {pair['name']} / {engine}")
                    if repeat < 0:
                        continue
                    matches = {key: result["fields"].get(key) == value for key, value in expected.items()}
                    normalized_matches = {key: normalized(result["fields"].get(key, "")) == normalized(value)
                                          for key, value in expected.items()}
                    row = {"case": pair["name"], "engine": engine, "repeat": repeat,
                           "exact_matches": matches, "normalized_matches": normalized_matches, "fields": result["fields"],
                           "processing_time_ms": result["processing_time_ms"],
                           "ocr_engine": result["ocr_engine"], "expiry": result["expiry"], "ocr": result["ocr"]}
                    runs.append(row)
                    print(json.dumps({"case": row["case"], "engine": engine, "repeat": repeat,
                        "ms": row["processing_time_ms"], "correct": sum(normalized_matches.values()), "total": len(matches),
                        "wrong_fields": [key for key, value in normalized_matches.items() if not value]}, ensure_ascii=False), flush=True)
    summary = [{"case": pair["name"], "engine": engine,
                "median_ms": statistics.median(row["processing_time_ms"] for row in runs
                    if row["case"] == pair["name"] and row["engine"] == engine)}
               for pair in manifest["pairs"] for engine in args.qualities]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"label": args.label, "date": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic fixtures only; main QR + OCR API; one warmup and repeated HTTP calls per case/engine. Not a real-card accuracy benchmark.",
        "summary": summary, "runs": runs}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
