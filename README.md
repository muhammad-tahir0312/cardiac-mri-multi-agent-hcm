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
```brightnessctl set 50%
1234

xrandr --output eDP-1 --brightness 1.0


Outputs:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

### 3) Preprocess all patients to NIfTI

```bash
python scripts/04_preprocess_dataset.py \
	--config src/config/base.yaml \
	--splits data/splits \
	--output data/processed \
	--workers 4
```

Useful flags:

- `--dry_run` (preview only)
- `--overwrite` (rebuild existing processed files)

Main output:

- `data/processed/manifest.csv`

### 4) Train classifier

```bash
python scripts/05_train.py --config src/config/base.yaml
```

Example with overrides:

```bash
python scripts/05_train.py --config src/config/base.yaml \
	training.epochs=30 \
	training.batch_size=4 \
	training.learning_rate=5e-5
```

Outputs:

- checkpoints: `results/models/cls_<model>_<timestamp>/`
- logs: `results/logs/cls_<model>_<timestamp>/`

### 5) Run prediction/evaluation

Default (latest checkpoint on test split):

```bash
python scripts/06_predict.py
```

With validation threshold tuning:

```bash
python scripts/06_predict.py \
	--splits test \
	--tune_threshold_on_val \
	--threshold_metric f1_macro \
	--threshold_min 0.05 \
	--threshold_max 0.95 \
	--threshold_steps 181
```

Prediction outputs:

- `results/predictions/<run_name>/predictions.csv`
- `results/predictions/<run_name>/metrics.json`
- `results/predictions/<run_name>/classification_report.txt`
- `results/predictions/<run_name>/confusion_matrix.png`
- `results/predictions/<run_name>/roc_curve.png`

### 6) Optional: Stratified k-fold evaluation

Generate k-fold split files only:

```bash
python scripts/07_kfold_evaluate.py --n_splits 5 --val_ratio 0.2
```

Run fold-wise training and aggregate metrics:

```bash
python scripts/07_kfold_evaluate.py \
	--run_training \
	--epochs 30 \
	--patience 10 \
	--num_workers 0
```

Outputs:

- fold splits: `data/splits_kfold/fold_XX/{train,val,test}.csv`
- fold summary: `data/splits_kfold/folds_summary.json`
- aggregate results (after training): `data/splits_kfold/kfold_results.json`

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

- **No checkpoint found in prediction**  
	Run `scripts/05_train.py` first, or pass `--checkpoint` explicitly.

- **Manifest missing**  
	Run `scripts/04_preprocess_dataset.py` first.

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

