#!/usr/bin/env python3
"""Copy only upstream modules needed by the Paddle + DocU recognizer worker."""
import argparse
import shutil
from pathlib import Path


def prepare_sources(namlhn, docu, output):
    ocr = output / "vietnam-cccd-ocr"
    shutil.copytree(namlhn / "ocr", ocr / "ocr", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    target = output / "docu"
    (target / "core/recognition").mkdir(parents=True)
    # Upstream core.__init__ imports its entire YOLO pipeline. The main worker
    # uses only the recognizer, its interfaces, and its data types.
    (target / "core/__init__.py").write_text('"""DocU recognizer runtime."""\n')
    for relative in ("core/base.py", "core/datatypes.py", "core/recognition/__init__.py",
                     "core/recognition/vietocr_recognizer.py", "configs/vietocr/vgg_seq2seq.yml"):
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(docu / relative, dest)
    config = target / "configs/vietocr/vgg_seq2seq.yml"
    original = config.read_text()
    if "cnn:\n" not in original:
        raise ValueError("DocU Seq2Seq config must define its CNN")
    # The full recognizer checkpoint overwrites the CNN weights. Avoid loading
    # a second 549 MiB ImageNet backbone or downloading it at startup.
    config.write_text(original.replace("cnn:\n", "cnn:\n  pretrained: false\n", 1))
    for source, destination in ((namlhn, ocr), (docu, target)):
        for license_file in source.glob("LICENSE*"):
            if license_file.is_file():
                shutil.copy2(license_file, destination / license_file.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namlhn", type=Path, required=True)
    parser.add_argument("--docu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_sources(args.namlhn, args.docu, args.output)


if __name__ == "__main__":
    main()
