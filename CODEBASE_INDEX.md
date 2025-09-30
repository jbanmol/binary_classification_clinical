## Codebase Index

### Purpose
High-signal map of where training, preprocessing, evaluation, and prediction live; how to run and modify training; and what artifacts are produced.

---

## Training: Orchestration and Flow

- Entrypoint (automated, multi-seed + bagging): `train_final.sh`
  - Sets thread env for stability on macOS/BLAS/LightGBM
  - Creates/uses `venv`, installs `requirements.txt`
  - Runs 3 seeds via `scripts/clinical_fair_pipeline.py` with flags:
    - `--target-sens 0.86 --target-spec 0.70`
    - `--use-umap-cosine --umap-components 16 --umap-neighbors 50`
    - `--calibration isotonic --final-calibration temperature`
    - `--models lightgbm,xgboost,brf,extratrees --top-k-models 2`
    - `--threshold-policy np --threshold-transfer iqr_mid`
    - `--holdout-seed 777 --save-preds`
  - Bags runs with `scripts/bag_scores.py` and exports a bundle for seed 17 to `models/final_np_iqrmid_u16n50_k2/`

- Main training pipeline: `scripts/clinical_fair_pipeline.py`
  - `build_child_dataset()` loads session-level behavioral data via `rag_system/research_engine.py`, filters to labeled ASD/TD, aggregates to child-level, engineers domain ratios and quantile-bin features.
  - Holdout split: `GroupShuffleSplit` by `child_id` (80/20), fixed `--holdout-seed` for reproducibility.
  - CV: `StratifiedGroupKFold` (5 folds) if available; fallback to stratified K-fold.
  - Per-fold preprocessing: `StandardScaler`, optional UMAP(cosine) or `PolynomialFeatures`.
  - Models per fold: LightGBM, XGBoost, Balanced RF, ExtraTrees, Logistic Regression (subset via `--models`).
  - Per-model calibration in CV via `CalibratedClassifierCV` (isotonic/sigmoid).
  - Thresholding policy per fold: `both_targets`, `spec_first`, `youden`, or `np` (Neyman–Pearson) via `choose_threshold()`.
  - Builds sub-ensembles `E_sens`/`E_spec` from best OOF AUC models; searches alpha-grid or trains meta-combiner (logistic) with per-fold OOF to avoid leakage.
  - Final calibration (optional): temperature scaling of combiner output; maps fold thresholds through the same monotone transform.
  - Threshold transfer to holdout: `median`, `iqr_mid`, or `quantile_map` with optional KS guard.
  - Outputs JSON summary to `results/*.json`; optionally exports bundle to `models/final_np_iqrmid_u16n50_k2/` including preprocessors, used models, combiner, temperature T, threshold, and summaries.

- Alternative script (reference): `scripts/bagged_end_to_end.py`
  - Mirrors the above pipeline, hardcodes best settings, and supports bagging across seeds in a single process.

---

## Data: Ingestion and Feature Engineering

- Source: `rag_system/research_engine.py`
  - `ColoringDataProcessor` parses raw `Coloring_*.json`, loads labels from `rag_system/config.py` resolved path.
  - Extracts per-session behavioral features: velocity stats, accelerometer-derived tremor, touch and stroke counts, zones/colors, temporal features, multi-touch, completion metrics.
  - `RAGResearchEngine` orchestrates ingestion, behavioral DB, optional vector indexing (ChromaDB) for research.

- Child-level dataset construction: `scripts/clinical_fair_pipeline.py::build_child_dataset()`
  - Aggregates canonical numeric features by child (mean) + `session_count`.
  - Engineers domain ratios: `touches_per_zone`, `strokes_per_zone`, `zones_per_minute`, `vel_std_over_mean`, `acc_std_over_mean`, `avg_ibp_norm`, `interpoint_rate`, `touch_rate`, `stroke_rate`.
  - Adds quantile-bin one-hot flags for the above ratios (quartiles).

- General-purpose library (not the main path in training pipeline, but available): `src/data_processing.py`
  - Imputation (`SimpleImputer`, `KNNImputer`), scaling (`StandardScaler`/`MinMax`/`Robust`), categorical encoding (`OneHotEncoder`/`OrdinalEncoder`), optional representation learning (`src/representation.py`: PCA/UMAP/autoencoder) and feature selection.

---

## Models, Ensembles, Calibration

- Model construction (training pipeline): `scripts/clinical_fair_pipeline.py::make_models()`
  - `lightgbm` (`LGBMClassifier`), `xgboost` (`XGBClassifier`), `brf` (`BalancedRandomForestClassifier` if installed), `extratrees`, `logreg`.
  - CV-time per-model calibration: `CalibratedClassifierCV(method='isotonic'|'sigmoid', cv=3)`.

- Ensemble formation
  - `E_sens`/`E_spec`: means over chosen per-model OOF probabilities (best LGBM/XGB + BRF/ET where present).
  - Combiner strategies:
    - Alpha blend: `P_ens = α·E_sens + (1−α)·E_spec`; α chosen by maximizing fold feasibility and surplus.
    - Meta-combiner: logistic regression on `[E_sens, E_spec, |E_sens−E_spec|]` trained leak-safely using OOF; can be calibrated.
  - Optional final calibration: temperature scaling on combiner output.

- Thresholding
  - Policies: `both_targets`, `spec_first`, `youden`, `np` (Neyman–Pearson with FPR constraint = `1 - target_spec`).
  - Transfer from CV to holdout: `median`, `iqr_mid`, or `quantile_map` with optional KS guard.

- Library trainer (generic, not used by the main pipeline): `src/model.py`
  - Provides a model zoo, grid search, custom sensitivity/specificity scorer, calibration, and threshold optimization utilities.

---

## Evaluation and Reporting

- In-pipeline: `scripts/clinical_fair_pipeline.py`
  - Per-fold metrics: AUC, selected threshold, sensitivity, specificity.
  - Holdout evaluation at transferred threshold; optional spec-first diagnostic.
  - JSON results: CV summary and holdout metrics; optionally includes holdout IDs, labels, probabilities for bagging (`--save-preds`).

- Library: `src/evaluation.py`
  - Metrics computation (accuracy, precision/recall, sensitivity/specificity, MCC, Kappa, AUC, AUPR, log loss) and plotting (ROC/PR, confusion, calibration, threshold analysis).

- Bagging across seeds: `scripts/bag_scores.py`
  - Averages per-child holdout probabilities across multiple runs (same holdout), selects a threshold on holdout (diagnostic), and reports final metrics.

---

## Exported Bundle and Prediction

- Bundle directory (example): `models/final_np_iqrmid_u16n50_k2/`
  - `bundle.json` keys: `feature_columns`, `models_used`, `ensembles` (`e_sens_names`, `e_spec_names`), `combiner` (`type`, `best_alpha` or `meta_combiner` path, `temperature_T`, `temperature_applied`), `threshold`, `holdout_metrics`, `cv_summary`, `feature_config`.
  - `preprocess/`: `scaler_all.joblib`, optional `umap_all.joblib`, optional `poly_all.joblib`.
  - `models/`: joblib files for used base learners and optional `meta_combiner.joblib`.
  - `training_reference.csv`: optional reference features for device/domain adaptation at inference.

- Prediction paths
  - End-to-end: `scripts/e2e_predict.py` builds raw→child features (labeled path or unlabeled fallback), aligns to bundle features, then calls `scripts/predict_cli.py`.
  - Direct scoring: `scripts/predict_cli.py` loads bundle, preprocessors, and models, reconstructs ensemble and combiner, applies temperature and threshold, and writes CSV with `prob_asd`, `pred_label`.
  - Optional demographic-aware thresholding at prediction time when demographics are provided.

---

## Configuration and Hyperparameters

- Training flags (`scripts/clinical_fair_pipeline.py --help`):
  - Targets: `--target-sens`, `--target-spec`
  - Features: `--use-umap-cosine`, `--umap-components`, `--umap-neighbors`, `--use-polynomial`
  - Models: `--models`, `--top-k-models`
  - Calibration: `--calibration` (CV-time), `--final-calibration` (holdout-time, includes `temperature`)
  - Thresholding: `--threshold-policy`, `--threshold-transfer`, `--use-quantile-threshold`, `--quantile-guard-ks`
  - Reproducibility: `--seed`, `--holdout-seed`
  - I/O: `--save-preds`, `--export-dir`, `--out-name`, `--demographics-path`

- Data paths and labels: `rag_system/config.py`
  - Env overrides: `PROJECT_PATH`, `RAW_DATA_PATH`, `LABELS_PATH`; default labels fallback: `data/processed/labels.csv`.
  - Note: the pipeline also attempts legacy `data/knowledge_base/lables_fileKeys.csv` if set/present.

- Threading stability (macOS):
  - `train_final.sh` and scripts set `OMP_*`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `LIGHTGBM_NUM_THREADS` for robust runs.

---

## How to Retrain or Modify Settings

- Fast path (recommended): edit `train_final.sh`
  - Adjust `SEEDS`, `HOLDOUT_SEED`, and the CLI flags to `scripts/clinical_fair_pipeline.py`.
  - Run:
    ```bash
    ./train_final.sh
    ```

- Manual run (single seed):
  ```bash
  ./venv/bin/python scripts/clinical_fair_pipeline.py \
    --target-sens 0.86 --target-spec 0.70 \
    --use-umap-cosine --umap-components 16 --umap-neighbors 50 \
    --calibration isotonic --final-calibration temperature \
    --models lightgbm,xgboost,brf,extratrees --top-k-models 2 \
    --threshold-policy np --threshold-transfer iqr_mid \
    --seed 17 --holdout-seed 777 --save-preds \
    --export-dir models/final_np_iqrmid_u16n50_k2
  ```

- Bag results across matching runs:
  ```bash
  ./venv/bin/python scripts/bag_scores.py \
    --pattern "results/final_s*_ho777_np_iqrmid_temp_u16n50_k2.json" \
    --out results/final_bagged_ho777_np_iqrmid_temp_u16n50_k2.json
  ```

---

## Key Files Index

- Orchestration
  - `train_final.sh`
  - `scripts/clinical_fair_pipeline.py`
  - `scripts/bagged_end_to_end.py` (reference/alt)
  - `scripts/bag_scores.py`

- Data & Features
  - `rag_system/research_engine.py`
  - `rag_system/config.py`
  - `src/data_processing.py` (general-purpose)
  - `src/representation.py` (PCA/UMAP/AE abstractions)

- Modeling & Evaluation
  - `scripts/clinical_fair_pipeline.py` (models, calibration, ensembles, thresholding)
  - `src/model.py` (generic trainer utilities)
  - `src/evaluation.py` (metrics/plots)

- Prediction
  - `scripts/e2e_predict.py`
  - `scripts/predict_cli.py`
  - `models/final_np_iqrmid_u16n50_k2/` (bundle + artifacts)

---

## Notes and Caveats

- Labels path: ensure `data/processed/labels.csv` exists (or set `LABELS_PATH`). Legacy `data/knowledge_base/lables_fileKeys.csv` is probed if present.
- Balanced Random Forest requires `imbalanced-learn`.
- UMAP requires `umap-learn`; alpha/meta combiner chooses best via CV feasibility+surplus.
- For reproducibility, keep the same `--holdout-seed` when comparing seeds for bagging.


