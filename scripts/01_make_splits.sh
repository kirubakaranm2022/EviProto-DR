#!/usr/bin/env bash
# 01_make_splits.sh
# Produces splits/aptos/{train,val,test}.csv using a 70/15/15 grade-stratified split.
# Run this once before preprocessing or training.
#
# Usage:
#   bash 01_make_splits.sh --source-csv data/aptos.csv
#
# Your aptos.csv must have columns:  path,grade
# Grades must be in {0,1,2,3,4}. Rows with any other grade are dropped automatically.
#
# Output:
#   splits/aptos/train.csv   (~2563 rows)
#   splits/aptos/val.csv     (~549  rows)
#   splits/aptos/test.csv    (~550  rows)
#   splits/aptos/manifest.json

set -euo pipefail

SOURCE_CSV="${1:---source-csv}"
if [[ "$SOURCE_CSV" == "--source-csv" ]]; then
    SOURCE_CSV="$2"
fi

# Resolve --source-csv flag if passed as named arg
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-csv) SOURCE_CSV="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "${SOURCE_CSV:-}" ]]; then
    echo "Usage: bash 01_make_splits.sh --source-csv path/to/aptos.csv"
    exit 1
fi

echo "Source CSV : $SOURCE_CSV"
echo "Output dir : splits/aptos"
echo ""

python -m eviproto_dr.run split \
    --source-csv "$SOURCE_CSV" \
    --out splits/aptos \
    --seed 42

echo ""
echo "Done. Splits written to splits/aptos/"
echo "  manifest : splits/aptos/manifest.json"
