"""Where things are: the images (flat folder images/, listed in images_manifest.csv) and the outputs."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
OUT = ROOT / "outputs"
MANIFEST = ROOT / "images_manifest.csv"


def manifest():
    """Rows of images_manifest.csv as dicts: file, op, role, note, bytes, sha256."""
    with open(MANIFEST, newline="") as fh:
        return list(csv.DictReader(fh))


def images(role):
    """Paths of the images with this role (e.g. "flat_pair"), sorted by file name (= capture time)."""
    return sorted(IMAGES / r["file"] for r in manifest() if r["role"] == role)


def notes(role):
    """{path: note} for a role, e.g. {".../IMG_...174844.dng": "chart 1", ...}."""
    return {IMAGES / r["file"]: r["note"] for r in manifest() if r["role"] == role}


def out(name):
    """Path of an output file in outputs/ (created if needed)."""
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    return OUT / name
