EviProto-DR

PyTorch implementation of Ordinal Prototype Learning with Evidential Uncertainty for Uncertainty-Aware Diabetic Retinopathy Grading (Kirubakaran & Vijayarajan, Frontiers in Artificial Intelligence, 2026).

Code release: v1.0.0

Relation to the Published Results

All results in the article were produced with this code, using the split manifests in splits/ and the checkpoints listed under Trained weights.

The exact Git commit corresponding to this release is identified by the GitHub v1.0.0 release tag.

Install

The implementation was developed and tested with Python 3.10 and CUDA 12.1.

python -m venv .venv
source .venv/bin/activate

pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

Run the smoke test:

python tests/smoke_test.py

The smoke test uses synthetic data and CPU execution to verify the main pipeline.

Data

The original retinal fundus image datasets are not redistributed with this repository. Users must obtain the datasets from their respective authorized sources and comply with the applicable dataset and competition terms.

Dataset

Use

Images

APTOS 2019

Development

3,662

DDR

Development

Grades 0-4; ungradable grade 5 removed

MESSIDOR-2

External test only

1,748

The public repository does not contain the original retinal images or preprocessed image files.

APTOS Split Manifests

The repository provides the APTOS split manifests:

splits/aptos/
├── train.csv
├── val.csv
├── test.csv
└── manifest.json

The CSV files contain only:

path,grade

where path identifies the original image and grade is the diabetic retinopathy grade from 0 to 4.

The image files themselves are not included.

The APTOS dataset was partitioned using a grade-stratified 70/15/15 split:

Training: 2,563 images

Validation: 549 images

Internal test: 550 images

Total: 3,662 images

The split manifests can be used to reconstruct the same train/validation/test partition after obtaining the original dataset from its authorized source.

MESSIDOR-2

MESSIDOR-2 is used only for external evaluation. The original MESSIDOR-2 images and annotations are not redistributed in this repository.

Run

1. Create Splits

If you are using the published APTOS split manifests in splits/aptos/, this step can be skipped.

To generate a new grade-stratified split from an authorized APTOS CSV:

python -m eviproto_dr.run split \
  --source-csv aptos_all.csv \
  --out splits/aptos \
  --seed 42

If patient identifiers are available in the source data, they can be supplied using:

--group-column patient_id

The published split manifests should be used when reproducing the reported results.

2. Preprocess the APTOS Images

After obtaining the original APTOS images, run:

python -m eviproto_dr.run preprocess \
  --csv splits/aptos/train.csv splits/aptos/val.csv splits/aptos/test.csv \
  --out cache/aptos

The preprocessing uses the Ben Graham-style retinal image preprocessing at 512 × 512 pixels.

Preprocessed files are written to the local cache/ directory and are not part of the public repository.

3. Train Using Five Random Seeds

The reported training protocol uses five random seeds:

42
123
456
789
1024

Run:

bash scripts/03_train_five_seeds.sh

The training workflow performs five independent training runs and evaluates the trained models on the held-out APTOS test set and the external MESSIDOR-2 dataset.

Training outputs are written to the local runs/ directory.

4. Aggregate Results

After completing the runs, aggregate the evaluation results using:

python -m eviproto_dr.run aggregate \
  --a "runs/aptos_s*/test_metrics.json" \
  --b "runs/ceonly_s*/test_metrics.json" \
  --out stats.json

The analysis includes mean, standard deviation, confidence intervals, and paired statistical testing with Holm correction where applicable.

5. Profile the Model

To measure model parameters, computational cost, latency, and memory:

python -m eviproto_dr.run profile --batch 32

Outputs

A training run produces files including:

run_config.json
train_log.jsonl
last.pt
model_weights.pt

run_config.json

Contains the training configuration and reproducibility information.

train_log.jsonl

Contains training information including loss terms, learning rate, prototype-related information, and validation metrics.

last.pt

Full training checkpoint used for resuming training.

model_weights.pt

Weights-only checkpoint intended for model inference and archival distribution.

The weights-only checkpoint can be loaded using:

torch.load("model_weights.pt", weights_only=True)

Evaluation

The evaluation pipeline produces:

Per-image predictions

Class probabilities

Uncertainty estimates

Prototype distances

Per-class F1 scores

Calibration/reliability information

Referral-threshold analysis

OMS and Proto-EDC measurements

Optional test embeddings for visualization

Test embeddings can be used for downstream t-SNE analysis.

Settings Used in the Article

Item

Value

Backbone

EfficientNet-B4 with ImageNet weights

Input size

512 × 512

Preprocessing

Ben Graham preprocessing

Projection head

1792 → 512 → 256

Embedding

L2-normalized

Number of prototypes

5

Prototype initialization

Class-mean initialization in epoch 1

Prototype update

EMA, β = 0.99

Evidence head

5 → 64 → 5

Evidence activation

Softplus + 1

λ₁

0.5

λ₂

0.1

λ₃

1.0

λ₄

0.5

τ

0.07

δₚ

0.1

Epochs

100

Initial objective

CE only, epochs 1-10

Full objective

Epochs 11-100

LR scheduler

Cosine annealing, T_max = 90

Optimizer

AdamW

Learning rate

3 × 10⁻⁴

Weight decay

1 × 10⁻⁴

Batch size

32

Sampling

Grade-aware sampling with replacement

Minimum samples per grade/batch

4

Random seeds

42, 123, 456, 789, 1024

Checkpoint

Last epoch

Early stopping

Not used

Referral threshold

u > 0.3

Optional Configuration

The implementation also provides optional settings for:

Automatic Mixed Precision

--amp

This runs the backbone using bfloat16 where supported, while keeping the heads and losses in float32.

Channels-Last Memory Format

--channels-last

This can improve convolution performance on compatible GPUs.

Monotonicity-Loss Mode

--mono-mode batch

The default mode is ema.

Resume Training

--resume runs/aptos_s42/last.pt

The checkpoint stores the relevant random-number-generator states to support reproducible continuation.

Stop After a Specified Number of Epochs

--stop-after N

Trained Weights

The trained model checkpoints are archived separately from the GitHub source repository.

For the APTOS five-seed experiments, the Zenodo record will contain:

weights/
├── aptos_seed42/
│   └── model_weights.pt
├── aptos_seed123/
│   └── model_weights.pt
├── aptos_seed456/
│   └── model_weights.pt
├── aptos_seed789/
│   └── model_weights.pt
└── aptos_seed1024/
    └── model_weights.pt

Zenodo DOI: To be added after publication of the Zenodo record.

Reproducibility

For reproduction of the reported experiments:

Obtain the original datasets from their authorized sources.

Use the split manifests provided in splits/aptos/.

Follow the preprocessing configuration described above.

Use the specified five random seeds.

Use the training configuration reported in this README.

Evaluate the resulting models using the provided evaluation utilities.

Compare the resulting metrics with the reported manuscript results.

The repository does not redistribute the original retinal images, preprocessed images, or raw dataset files.

Repository Structure

EviProto-DR/
│
├── eviproto_dr/
│   ├── __init__.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── preprocess.py
│   ├── profile.py
│   ├── run.py
│   └── stats.py
│
├── scripts/
│   ├── 01_make_splits.sh
│   ├── 02_preprocess.sh
│   └── 03_train_five_seeds.sh
│
├── splits/
│   └── aptos/
│       ├── manifest.json
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── tests/
│   └── smoke_test.py
│
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt

Limitations

The publicly available split manifests are image-level. The available dataset annotations do not provide sufficient patient identifiers to verify patient-level independence across all images.

The referral analysis represents a retrospective evaluation and should not be interpreted as prospective clinical validation.

This software is intended for research use and is not a medical device.

License

The source code in this repository is released under the MIT License.

See LICENSE for the complete license text.

The datasets used by this project are subject to their respective dataset, provider, and competition terms and are not redistributed with this repository.

Citation

If you use EviProto-DR in your research, please cite the associated article.

The citation metadata are provided in CITATION.cff.

The Zenodo DOI will be added to this README and to CITATION.cff after the archival record is publi
