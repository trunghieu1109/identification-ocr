#!/usr/bin/env python3
"""Compare recognition alone on identical labelled line crops, preserving accents."""
import argparse
import gc
import hashlib
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(expected, observed):
    row = list(range(len(observed) + 1))
    for index, left in enumerate(expected, 1):
        previous, row = row, [index]
        for position, right in enumerate(observed, 1):
            row.append(min(row[-1] + 1, previous[position] + 1, previous[position - 1] + (left != right)))
    return row[-1]


def measure(name, images, samples, device):
    import numpy as np

    if name in {"vgg_seq2seq", "vgg_transformer"}:
        sys.path.insert(0, os.getenv("DOCU_SOURCE_DIR", "/opt/docu"))
        from core.recognition.vietocr_recognizer import VietOCRRecognizer
        predictor = VietOCRRecognizer(config_name=name, weights_path=f"/app/model/recognition/{name}.pth", device=device).predictor
        read = lambda: predictor.predict_batch(images, return_prob=True)
    else:
        from paddleocr import TextRecognition
        predictor = TextRecognition(model_name=name, device=device.replace("cuda", "gpu"), enable_mkldnn=False)

        def read():
            texts, scores = [], []
            for result in predictor.predict([np.asarray(image)[:, :, ::-1].copy() for image in images], batch_size=8):
                payload = getattr(result, "json", result)
                payload = payload.get("res", payload)
                texts.append(payload["rec_text"])
                scores.append(float(payload["rec_score"]))
            return texts, scores

    started = perf_counter()
    texts, scores = read()
    elapsed = (perf_counter() - started) * 1000
    if len(texts) != len(samples) or len(scores) != len(samples):
        raise ValueError(f"Incomplete recognition output from {name}")
    rows = []
    for sample, text, score in zip(samples, texts, scores):
        expected, prediction = normalized(sample["text"]), normalized(text)
        rows.append({**sample, "prediction": prediction, "confidence": float(score),
                     "errors": edit_distance(expected, prediction), "correct": expected == prediction})
    return {"model": name, "device": device, "exact": sum(row["correct"] for row in rows), "total": len(rows),
            "cer": sum(row["errors"] for row in rows) / max(1, sum(len(normalized(row["text"])) for row in rows)),
            "elapsed_ms": round(elapsed, 2), "rows": rows}


def main():
    from PIL import Image
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="JSON with samples: image, text, name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["vgg_seq2seq", "vgg_transformer"],
                        choices=["vgg_seq2seq", "vgg_transformer", "PP-OCRv6_medium_rec", "latin_PP-OCRv5_mobile_rec"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.dataset.read_text(encoding="utf-8"))
    samples, images = manifest["samples"], []
    if not samples:
        parser.error("Dataset must contain at least one labelled crop")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "synthetic": manifest.get("synthetic", False),
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "scope": "Same decoded line crops; recognition only. NFC and whitespace normalized; case and accents preserved. "
                       "Elapsed time includes first-shape warmup and is not a latency benchmark.", "runs": []}
    try:
        for sample in samples:
            with Image.open(args.dataset.parent / sample["image"]) as image:
                images.append(image.convert("RGB"))
        for name in args.models:
            result = measure(name, images, samples, args.device)
            report["runs"].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True)
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        for image in images:
            image.close()


if __name__ == "__main__":
    main()
