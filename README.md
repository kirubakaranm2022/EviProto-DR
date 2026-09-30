# EviProto-DR

PyTorch code for *Ordinal Prototype Learning with Evidential Uncertainty for Uncertainty-Aware Diabetic Retinopathy Grading* (Kirubakaran & Vijayarajan, Frontiers in Artificial Intelligence, 2026).

Code release: `v1.0.0`, archived at Zenodo, DOI `10.5281/zenodo.XXXXXXX` (replace after release).

## Relation to the published results

> **Authors: keep exactly one of the two statements below and delete the other.**
>
> **(A)** All results in the article were produced with this code, commit `<hash>`, using the split manifests in `splits/` and the checkpoints listed under *Trained weights*.
>
> **(B)** This is a clean re-implementation written from the article. Results reproduced with it on the same splits are listed in `RESULTS.md`; they differ from the published tables by the amounts shown there.

## Install

Python 3.10, CUDA 12.1:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tests/smoke_test.py      # synthetic data, CPU, about 1-2 minutes
```

The smoke test checks the full pipeline, that cached preprocessing is pixel-identical to on-the-fly preprocessing, and that training gives identical weights with 0 or 2 DataLoader workers.

## Data

The datasets are not redistributed. Download them from their providers:

| Dataset | Use | Images used |
|---|---|---|
| APTOS 2019 (Kaggle, training set with labels) | development | 3,662 |
| DDR (Nankai University) | development | grades 0-4 only; ungradable (grade 5) removed |
| MESSIDOR-2 (ADCIS) with adjudicated grades | external test only | see `splits/messidor2.csv` |

Each CSV has columns `path,grade` (grades 0-4). Paths may be absolute or relative to the CSV.
The exact image lists used in the article are in `splits/` (file names only, with SHA-256 of each CSV in `manifest.json`). Use them to rebuild identical splits.

## Run

```bash
# 1. split (skip if you use the published split files)
python -m eviproto_dr.run split --source-csv aptos_all.csv --out splits/aptos --seed 42
#    add --group-column patient_id when patient IDs are available

# 2. preprocess once (Ben Graham crop + local mean subtraction, 512 px, lossless PNG)
python -m eviproto_dr.run preprocess --csv splits/aptos/train.csv splits/aptos/val.csv splits/aptos/test.csv --out cache/aptos
python -m eviproto_dr.run preprocess --csv messidor2.csv --out cache/messidor2

# 3. five seeds, internal test + external MESSIDOR-2
bash scripts/run_five_seeds.sh aptos

# 4. mean, SD, 95% CI, paired tests with Holm correction
python -m eviproto_dr.run aggregate --a "runs/aptos_s*/test_metrics.json" --b "runs/ceonly_s*/test_metrics.json" --out stats.json

# 5. parameters, FLOPs, latency, peak training memory
python -m eviproto_dr.run profile --batch 32
```

`train` writes `run_config.json` (all arguments, package version, SHA-256 of the split files), `train_log.jsonl` (loss terms, learning rate, prototype geometry, validation metrics per epoch), `last.pt` (full state for `--resume`) and `model_weights.pt` (weights only; loads with `torch.load(..., weights_only=True)`).

`evaluate` writes per-image predictions with probabilities, uncertainty and prototype distances, a metrics JSON with per-class F1, reliability-diagram bins, a referral-threshold sweep (0.10-0.60), OMS and Proto-EDC, and optionally test embeddings for t-SNE.

## Settings used in the article

| Item | Value |
|---|---|
| Backbone | EfficientNet-B4, ImageNet weights |
| Input | 512 x 512, Ben Graham preprocessing |
| Projection | 1792 -> 512 -> 256, L2-normalised |
| Prototypes | 5, class-mean initialisation in epoch 1, EMA beta = 0.99 |
| Evidence head | distances (5) -> 64 -> 5, softplus + 1 |
| Loss weights | lambda1 = 0.5, lambda2 = 0.1, lambda3 = 1.0, lambda4 = 0.5, tau = 0.07, delta_p = 0.1 |
| Schedule | epochs 1-10 CE only, 11-100 full objective, cosine T_max = 90 |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4, batch 32 |
| Sampling | grade-aware with replacement, >= 4 images per grade per batch (training only) |
| Seeds | 42, 123, 456, 789, 1024 |
| Checkpoint | last epoch, no early stopping |
| Referral | u > 0.3 |

## Options that are not the reported setting

- `--amp` runs the backbone in bf16 (heads and losses stay float32). Faster on A100/H100, small numeric differences.
- `--channels-last` can speed up convolutions on recent GPUs.
- `--mono-mode batch` applies the ordinal monotonicity loss to differentiable per-batch class means. In the default `ema` mode the loss is computed on EMA prototype buffers, which carry no gradient, so it is logged but does not change the weights.
- `--stop-after N` ends a session after N epochs; `--resume runs/.../last.pt` continues with identical results (RNG states are saved).

## Trained weights

`model_weights.pt` for each dataset and seed: Zenodo DOI `10.5281/zenodo.XXXXXXX`.

## Limitations

Splits are image-level; the public labels do not identify patients for all images. Referral results are retrospective simulations. This code is for research use and is not a medical device.

## Licence and citation

MIT licence (see `LICENSE`). Please cite the article; `CITATION.cff` has the entry.
