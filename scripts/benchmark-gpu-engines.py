#!/usr/bin/env python3
"""Compare isolated GPU OCR engines on identical images, excluding HTTP/QR."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summary(samples):
    ordered = sorted(samples)
    return {"median_ms": round(statistics.median(ordered), 2),
            "mean_ms": round(statistics.mean(ordered), 2),
            "min_ms": round(ordered[0], 2), "max_ms": round(ordered[-1], 2),
            "p95_ms": round(ordered[math.ceil(len(ordered) * .95) - 1], 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("namlhn", "docu"), required=True)
    parser.add_argument("--model", choices=("vgg_transformer", "vgg_seq2seq"), required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="JSON pairs with front/back file paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must be positive")
    import torch
    from identification_demo.processing import load_image
    from identification_demo.documents import FRONT_FIELDS, BACK_FIELDS, fold
    from identification_demo.expiry import parse_expiry

    required = {side: [key for key, *_ in schema if key != "identifying_features"]
                for side, schema in (("front", FRONT_FIELDS), ("back", BACK_FIELDS))}

    torch.set_num_threads(int(os.getenv("OCR_CPU_THREADS", "4")))
    assert torch.cuda.is_available(), "GPU required: CPU fallback invalidates this benchmark"
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                   "gpu_capability": list(torch.cuda.get_device_capability(0)),
                   "gpu_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
                   "cpu_threads": torch.get_num_threads()}
    dataset = json.loads(args.dataset.read_text())
    pairs, manifest = [], []
    for pair in dataset["pairs"]:
        item = {"name": pair["name"], "expected": pair.get("expected", {}), "images": {}}
        metadata = {"name": pair["name"], "images": {}}
        for side in ("front", "back"):
            path = args.dataset.parent / pair[side]
            data = path.read_bytes()
            item["images"][side] = load_image(data)
            metadata["images"][side] = {"file": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                                        "size": list(item["images"][side].size), "bytes": len(data)}
        pairs.append(item)
        manifest.append(metadata)
    weights = Path("/app/model/recognition") / f"{args.model}.pth"
    config_root = Path(os.getenv("DOCU_SOURCE_DIR", "/opt/docu")) / "configs/vietocr"
    initialized = perf_counter()
    if args.engine == "namlhn":
        import paddle
        environment.update({"paddle": paddle.__version__, "paddle_cuda": paddle.version.cuda(),
                            "paddleocr": importlib.metadata.version("paddleocr")})
        assert paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        paddle.set_device("gpu:0")
        assert float(paddle.sum(paddle.ones([2, 2])).numpy()) == 4
        paddle.device.cuda.synchronize()
        os.environ["OCR_DEVICE"] = "cuda"
        os.environ["OCR_MODEL"] = args.model
        os.environ["OCR_WEIGHTS"] = str(weights)
        from identification_demo.engine import NamLhnEngine
        from vietocr.tool.config import Cfg
        engine = NamLhnEngine()
        # Reuse the checked-in full VietOCR config and the identical DocU
        # checkpoint; avoid online configuration downloads in the measurement.
        with patch.object(Cfg, "load_config_from_name", side_effect=lambda name: Cfg.load_config_from_file(config_root / f"{name}.yml")):
            engine.initialize()
        assert str(next(engine._extractor._vietocr.model.parameters()).device).startswith("cuda")
        environment["recognizer_device"] = str(next(engine._extractor._vietocr.model.parameters()).device)
        environment["paddle_device"] = paddle.device.get_device()
        paddle_ocr = engine._extractor._detector._engine
        environment["paddle_ocr_device"] = paddle_ocr._common_args["device"]
        environment["paddle_models"] = {key: paddle_ocr._params[key] for key in
                                        ("text_detection_model_name", "text_recognition_model_name")}
        assert str(environment["paddle_ocr_device"]).startswith("gpu")

        def sync():
            torch.cuda.synchronize()
            paddle.device.cuda.synchronize()

        def read(image, side):
            return engine.extract_image(image, side, needed_fields=required[side])
    else:
        os.environ["GPU_OCR_RECOGNITION_MODEL"] = args.model
        os.environ["GPU_OCR_AUTHORITY_MODEL"] = args.model
        from identification_demo.gpu_worker import DocULocalEngine
        from identification_demo.gpu_engine import normalize_result
        engine = DocULocalEngine()
        engine.initialize()
        assert str(next(engine._pipeline.recognizer.predictor.model.parameters()).device).startswith("cuda")
        environment["recognizer_device"] = str(next(engine._pipeline.recognizer.predictor.model.parameters()).device)

        def sync():
            torch.cuda.synchronize()

        def read(image, side):
            payload = engine.extract_image(image, side, needed_fields=required[side])
            result = normalize_result(payload)
            result["text_lines"] = payload.get("text_lines", [])
            return result
    sync()
    init_ms = (perf_counter() - initialized) * 1000
    model_hash = hashlib.sha256(weights.read_bytes()).hexdigest()

    def run_pair(pair):
        times, checks, stages, work = {}, {}, {}, {}
        sync()
        pair_start = perf_counter()
        for side, schema in (("front", FRONT_FIELDS), ("back", BACK_FIELDS)):
            sync()
            started = perf_counter()
            raw = read(pair["images"][side], side)
            sync()
            times[side] = round((perf_counter() - started) * 1000, 2)
            fields = raw["structured_fields"]
            expected = pair["expected"].get(side, {})
            checks[side] = {"nonempty_fields": [key for key, *_ in schema if fields.get(key)],
                            "expected_matches": {key: fold(str(fields.get(key, ""))).strip() == fold(str(value)).strip()
                                                 for key, value in expected.items()},
                            "expiry_values": sorted({item["expiry_date"] for item in parse_expiry(raw.get("text_lines", []))}),
                            "ocr_line_count": len(raw.get("text_lines", []))}
            stages[side] = raw.get("timings_ms", {})
            work[side] = raw.get("work", {})
        sync()
        return {"pair": pair["name"], "pair_ms": round(sum(times.values()), 2),
                "pair_wall_ms": round((perf_counter() - pair_start) * 1000, 2),
                "sides_ms": times, "checks": checks, "timings_ms": stages, "work": work}

    first = run_pair(pairs[0])
    print(json.dumps({"engine": args.engine, "model": args.model, "initialization_ms": round(init_ms, 2),
                      "first_pair_ms": first["pair_ms"], "environment": environment}), flush=True)
    # Warm every image before collecting samples; first inference stays separate.
    for _ in range(args.warmup):
        for pair in pairs:
            run_pair(pair)
    samples = []
    for repeat in range(args.repeats):
        for pair in pairs:
            sample = {"repeat": repeat + 1, **run_pair(pair)}
            samples.append(sample)
            print(json.dumps({"repeat": repeat + 1, "pair": pair["name"], "pair_ms": sample["pair_ms"],
                              "sides_ms": sample["sides_ms"]}), flush=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "engine": args.engine, "model": args.model, "environment": environment,
              "checkpoint_sha256": model_hash, "dataset": manifest,
              "scope": "predecoded RGB images; no HTTP, upload, QR or image decode; sequential sides; CUDA synchronized",
              "requested_fields": required,
              "initialization_ms": round(init_ms, 2), "first_pair": first,
              "warmup_rounds": args.warmup, "repeats": args.repeats,
              "summary": {"all_pairs": summary([sample["pair_ms"] for sample in samples]),
                          "by_pair": {pair["name"]: summary([sample["pair_ms"] for sample in samples if sample["pair"] == pair["name"]])
                                      for pair in pairs},
                          "by_side": {side: summary([sample["sides_ms"][side] for sample in samples]) for side in ("front", "back")}},
              "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    engine.close()
    for pair in pairs:
        for image in pair["images"].values():
            image.close()
    print(json.dumps({"saved": str(args.output), "summary": report["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
