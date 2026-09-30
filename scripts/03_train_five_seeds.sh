#!/usr/bin/env bash
# Five-seed protocol (Section 4.2.2). Usage: bash scripts/run_five_seeds.sh aptos
# Expects cache/<name>/{train,val,test}.csv from `run.py preprocess` and cache/messidor2/messidor2.csv.
set -euo pipefail
NAME=${1:-aptos}
for SEED in 42 123 456 789 1024; do
  OUT=runs/${NAME}_s${SEED}
  python -m eviproto_dr.run train --train-csv cache/${NAME}/train.csv --val-csv cache/${NAME}/val.csv \
      --out ${OUT} --seed ${SEED} --save-prototypes
  python -m eviproto_dr.run evaluate --csv cache/${NAME}/test.csv --checkpoint ${OUT}/last.pt \
      --predictions ${OUT}/test_predictions.csv --metrics-out ${OUT}/test_metrics.json --embeddings ${OUT}/test_embeddings.npz
  python -m eviproto_dr.run evaluate --csv cache/messidor2/messidor2.csv --checkpoint ${OUT}/last.pt \
      --predictions ${OUT}/external_predictions.csv --metrics-out ${OUT}/external_metrics.json
done
