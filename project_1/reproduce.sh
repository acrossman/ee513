#!/usr/bin/env bash
# Reproduce every value in the Pixel 7a datasheet from the raw images, in one command:
#
#     bash reproduce.sh
#
# Linux or macOS with Python 3.12. Before running, download the images listed in images_manifest.csv
# into a folder named images/ next to this script (see README). The script:
#   1. creates a virtual environment (.venv) and installs the pinned libraries (requirements.txt);
#   2. checks images/ holds every required file, unchanged (SHA-256);
#   3. runs the unit tests on synthetic data with known answers, and stops if any fails;
#   4. runs the four measurements and writes outputs/results.json and the figures;
#   5. writes outputs/report_condensed.pdf, including a value-by-value match against the datasheet.
# About 10 minutes plus the library install. Set PYTHON=/path/to/python3.12 to choose the interpreter.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
step() { echo; echo "=== $*"; }

step "Python environment"
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
    echo "Python 3.12 is required (found: $("$PYTHON" --version 2>&1)). Install it and set PYTHON=/path/to/python3.12"
    exit 1
fi
[ -d .venv ] || "$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo "installed: $(python -m pip freeze | grep -ciE '^(numpy|scipy|matplotlib|opencv-contrib-python|rawpy|tifffile|colour-science|pytest)==') pinned libraries"

step "Images"
python -m pipeline.check_images

step "Unit tests (synthetic data, known answers)"
mkdir -p outputs
if ! python -m pytest -q -p no:cacheprovider camchar/tests > outputs/tests_full.txt 2>&1; then
    tail -n 25 outputs/tests_full.txt
    echo "A unit test failed: stopping before measuring anything with that code."
    exit 1
fi
tail -n 1 outputs/tests_full.txt | tee outputs/tests.txt

step "OP1 geometry";             python -m pipeline.op1_geometry
step "OP2 spatial resolution";   python -m pipeline.op2_mtf
step "OP3 noise";                python -m pipeline.op3_noise
step "OP4 color";                python -m pipeline.op4_color
step "results.json";             python -m pipeline.make_results
step "Figures";                  python -m pipeline.figures
step "Condensed report";         python -m pipeline.report
echo
echo "Done: outputs/report_condensed.pdf, outputs/results.json, outputs/figures/"
