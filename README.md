# Cardiac MRI Multi-Agent System for HCM Analysis

Automated decision support for Hypertrophic Cardiomyopathy (HCM) diagnosis using a multi-agent deep learning pipeline on cardiac MRI.

## Dataset

- Source: [Kaggle - Hypertrophic Cardiomyopathy Dataset](https://www.kaggle.com/datasets/danialsharifrazi/hypertrophic-cardiomyopathy-dataset)
- Current raw format used in this repo: **JPEG slice series** per patient/series directory
- Pipeline format conversion: **JPEG series → preprocessed NIfTI (.nii.gz)**
- Main task: **Binary classification** (`Normal` vs `Sick`)

---

## Project Pipeline

The default workflow is:

1. Data scan & sanity checks
2. Patient-level split creation (`train/val/test`)
3. Preprocessing (routing + ingestion + normalization/resizing)
4. Classification training
5. Prediction + metrics/figures export
6. Optional k-fold evaluation for research-grade reporting

Scripts are under `scripts/` and are numbered in execution order.

---

## Environment Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you already have a virtual environment (for example `venv/`), activate that one instead.

---

## Expected Data Layout

Place your dataset under:

```text
data/raw/
	Normal/
		Directory_1/
			series.../
				imgXXXX-z.jpg
	Sick/
		Directory_17/
			series.../
				imgXXXX-z.jpg
```

Each `Directory_X` is treated as one patient.

---

## How to Run (End-to-End)

### 1) Optional: dataset intelligence and count checks

```bash
python scripts/01_data_intelligence.py
python scripts/02_count_samples.py
```

### 2) Create patient-level splits

```bash
python scripts/03_create_patient_splits.py \
	--data_root data/raw \
	--splits_dir data/splits \
	--train_ratio 0.80 \
	--val_ratio 0.10 \
	--test_ratio 0.10 \
	--seed 42
```


Outputs:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

---

## 2D Image Pipeline (Nested PNG/JPG)

Use this path when training directly on image slices (without NIfTI conversion).

### 1) Create leakage-safe patient-level image splits

```bash
python scripts/08_create_image_splits.py \
	--data_root data/raw \
	--splits_dir data/splits_image \
	--train_ratio 0.80 \
	--val_ratio 0.10 \
	--test_ratio 0.10 \
	--seed 42
```

Optional 10-fold image splits:

```bash
python scripts/08_create_image_splits.py --kfolds 10 --kfold_val_ratio 0.10
```

### 2) Train 2D image classifier

```bash
python scripts/09_train_image_classifier.py --config src/config/image2d.yaml
```

Example override:

```bash
python scripts/09_train_image_classifier.py --config src/config/image2d.yaml \
	training.batch_size=32 \
	model.backbone=resnet50
```

Saved outputs include:

- `results/models/image2d/<run_name>/best.pt`
- `results/logs/image2d/<run_name>/test_metrics.json`
- `results/logs/image2d/<run_name>/test_predictions.csv`

`test_metrics.json` reports Recall/Sensitivity, Specificity, F1, AUROC, and confusion matrices at threshold `0.5` and at a validation-optimized threshold.

### 3) Single-image inference with confidence score

```bash
python scripts/10_predict_single_image.py \
	--image data/raw/Sick/Directory_17/example.png \
	--checkpoint results/models/image2d/<run_name>/best.pt \
	--config src/config/image2d.yaml
```

---

## Configuration

Main configuration file: `src/config/base.yaml`

Important keys to tune:

- `training.imbalance_strategy`: `loss_weights` | `sampler` | `none`
- `training.loss`: `cross_entropy` | `focal` | `weighted_cross_entropy`
- `training.epochs`, `training.batch_size`, `training.learning_rate`
- `classification.model`: `resnet3d_18` | `resnet3d_50` | `slice_aggregation`

---

## Common Issues

- **Very slow training on CPU**  
	This is expected for 3D CNNs. Reduce epochs/batch size for quick experiments or use GPU.

---

## Research Note

For research-based experimentation (including MSc CS workflows), the recommended reporting path is:

1. Run fixed split training/inference for fast iteration.
2. Lock a decision policy (for example validation-tuned threshold metric).
3. Run k-fold evaluation and report **mean ± std** across folds.
4. Archive configs and generated artifacts under `results/` for reproducibility.

This gives stronger evidence than single-split metrics on small cohorts.

---

## Author / Academic Profile

- MSc in Computer Science (Research), Universiti Malaya.
- Research focus: deep learning for medical imaging and reproducible clinical AI evaluation.

