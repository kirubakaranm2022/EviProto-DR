#!/usr/bin/env bash
# 02_preprocess.sh
# Applies Ben Graham preprocessing (circular crop + local mean subtraction) once
# and writes lossless PNGs to cache/. Training reads cached images directly,
# removing the full-resolution decode + Gaussian blur from every step.
#
# Run this after 01_make_splits.sh and before 03_train_five_seeds.sh.
#
# Usage:
#   bash 02_preprocess.sh --messidor-csv data/messidor2.csv
#
# Expects:
#   splits/aptos/train.csv          (from 01_make_splits.sh)
#   splits/aptos/val.csv            (from 01_make_splits.sh)
#   splits/aptos/test.csv           (from 01_make_splits.sh)
#   data/messidor2.csv              (columns: path,grade  — grades 0..4 only)
#
# Output:
#   cache/aptos/train.csv           (cached=1, points to PNG files)
#   cache/aptos/val.csv
#   cache/aptos/test.csv
#   cache/aptos/images/             (preprocessed PNGs, 512×512)
#   cache/messidor2/messidor2.csv
#   cache/messidor2/images/

set -euo pipefail

MESSIDOR_CSV=""
SIZE=512
WORKERS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --messidor-csv) MESSIDOR_CSV="$2"; shift 2 ;;
        --size)         SIZE="$2";         shift 2 ;;
        --workers)      WORKERS="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MESSIDOR_CSV" ]]; then
    echo "Usage: bash 02_preprocess.sh --messidor-csv path/to/messidor2.csv"
    exit 1
fi

for f in splits/aptos/train.csv splits/aptos/val.csv splits/aptos/test.csv; do
    if [[ ! -f "$f" ]]; then
        echo "Missing: $f — run 01_make_splits.sh first."
        exit 1
    fi
done

echo "Preprocessing APTOS splits (size=${SIZE}, workers=${WORKERS})..."
python -m eviproto_dr.run preprocess \
    --csv splits/aptos/train.csv splits/aptos/val.csv splits/aptos/test.csv \
    --out cache/aptos \
    --size "$SIZE" \
    --workers "$WORKERS"

echo ""
echo "Preprocessing MESSIDOR-2 (size=${SIZE}, workers=${WORKERS})..."
python -m eviproto_dr.run preprocess \
    --csv "$MESSIDOR_CSV" \
    --out cache/messidor2 \
    --size "$SIZE" \
    --workers "$WORKERS"

echo ""
echo "Done. Cached images and CSVs written to cache/"
echo "  APTOS     : cache/aptos/"
echo "  MESSIDOR-2: cache/messidor2/"
