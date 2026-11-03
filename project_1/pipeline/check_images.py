"""Check that images/ holds every file in images_manifest.csv, unchanged (size and SHA-256).

Extra files in images/ are ignored. Exits with status 1 and a list of problems if anything is missing or
altered, so nothing is measured from incomplete or corrupted data.
Usage: python -m pipeline.check_images
"""
import hashlib
import sys

from .paths import IMAGES, manifest


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    rows = manifest()
    missing, altered = [], []
    for r in rows:
        p = IMAGES / r["file"]
        if not p.exists():
            missing.append(r["file"])
        elif p.stat().st_size != int(r["bytes"]) or sha256(p) != r["sha256"]:
            altered.append(r["file"])
    if missing or altered:
        print(f"images/ check FAILED: {len(missing)} missing, {len(altered)} altered (of {len(rows)} required)")
        for name in missing[:20]:
            print("  missing:", name)
        for name in altered[:20]:
            print("  altered:", name)
        print("Download the files listed in images_manifest.csv (README) into images/.")
        sys.exit(1)
    print(f"images/ check passed: all {len(rows)} required files present and unchanged")


if __name__ == "__main__":
    main()
