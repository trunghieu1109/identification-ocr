"""Download only the two additional experiment assets, verifying their hashes."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

ASSETS = {
    "docaligner-lcnet050.onnx": (
        "https://drive.google.com/uc?export=download&id=1J7cRuupeEIudYrH_CCSV9WvFfu9JM_qU",
        "32d186080ce16442674d4c0eaaaaac878eea289b56a8d1284f05fff1ff42e220",
    ),
    "mrz.traineddata": (
        "https://raw.githubusercontent.com/sivakumar-mahalingam/fastmrz/"
        "bb1d4ee38eb1eaa88cd80e1eba88bacc1842cd76/tessdata/mrz.traineddata",
        "e44f5b7a6bdd3f382ef3bfa84ee0057f5897946a84a094c26910e0a124f3a9bd",
    ),
}


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in ASSETS.items():
        path = directory / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        with urlopen(url, timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Model checksum mismatch: {name}")
        path.write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    download(parser.parse_args().directory)
