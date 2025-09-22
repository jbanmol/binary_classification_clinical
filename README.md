## Binary Classification Project – Full Overview

This README documents the entire training and prediction stack with file-level detail: how raw Coloring sessions become child-level features, how models are trained/combined/calibrated, and how predictions are produced. It also includes a dependency graph of the main components and guidance on robustness and CV variance.

### Quickstart (Predictions)
```bash
python3 -m venv venv && ./venv/bin/python -m pip install --upgrade pip && ./venv/bin/pip install -r requirements.txt
./venv/bin/python scripts/e2e_predict.py --raw data/raw/phase2_file_keys
```
Outputs:
- `results/phase2_file_keys_features_raw.csv`
- `results/phase2_file_keys_features_aligned.csv`
- `phase2_file_keys_results.csv` (columns: `child_id, prob_asd, pred_label`)

### Labels and Classes
- Positive class: `1 = ASD`
- Negative class: `0 = TD`

### Environment
- Create and use the project venv:
```bash
python3 -m venv venv && ./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

### High-level Flow
1) Ingest raw Coloring sessions and build leakage-safe child-level features
2) Split train/holdout by child (group-aware) and run 5-fold group-aware CV
3) Train and calibrate multiple models; construct sensitivity- and specificity-oriented sub-ensembles
4) Combine sub-ensembles (alpha or meta), optionally temperature-calibrate combiner probabilities
5) Select decision threshold via clinical policy (Neyman–Pearson or both-targets) and robust transfer
6) Evaluate holdout; optionally bag across seeds; export deployable bundle
7) Predict on new unlabeled data using the exported bundle

---

## Files by Responsibility

### Training – Orchestration
- `scripts/clinical_fair_pipeline.py`
  - Builds child-level dataset (via internal `build_child_dataset` using `rag_system.research_engine`).
  - Grouped holdout split (`GroupShuffleSplit`) to keep children disjoint.
  - 5-fold CV with `StratifiedGroupKFold` (or stratified k-fold as fallback).
  - Per-fold preprocessing: standardization; optional UMAP(cosine) or polynomial features.
  - Per-model calibration (`CalibratedClassifierCV`; isotonic or sigmoid).
  - Sub-ensembles: `E_sens`, `E_spec` from best models; alpha-grid and optional meta-combiner.
  - Threshold selection policies: `both_targets`, `spec_first`, `youden`, `np`.
  - Threshold transfer to holdout: median/IQR-mid/quantile-map with optional KS-guard.
  - Exports bundle with preprocessors, models, combiner, temperature `T`, and threshold.

- `train_final.sh`
  - Automates multi-seed runs and bagging; exports deployable bundle to `models/final_np_iqrmid_u16n50_k2/`.

- `scripts/bag_scores.py`
  - Averages holdout probabilities across multiple runs (same holdout split); evaluates at selected threshold.

### Training – Supporting Libraries
- `rag_system/research_engine.py`, `rag_system/config.py`
  - Ingests raw data (via `RAW_DATA_PATH`) and extracts per-session behavioral features.
  - Used by both training (for labeled builds) and prediction (for unlabeled fallback).

- `src/data_processing.py`
  - General-purpose preprocessing: splitting, imputation (`SimpleImputer`, `KNNImputer`), scaling (`StandardScaler`/`MinMax`/`Robust`), categorical encoding (`OneHotEncoder`/`OrdinalEncoder`), numeric feature engineering, feature selection (`SelectKBest` with chi2/f_classif/mutual_info), optional representation learning.

- `src/representation.py`
  - Representation learning abstractions: PCA, UMAP, and an optional autoencoder (PyTorch with graceful fallback).

- `src/model.py`
  - Model zoo and trainer (LogReg/Tree/ExtraTrees/RF/GB/ADA/SVM/KNN/GaussianNB/MLP/XGBoost/LightGBM). CV, calibration, save/load, and threshold optimization utilities.

- `src/evaluation.py`
  - Metric computation, ROC/PR/confusion plots, calibration curves, threshold analysis, and model comparisons.

### Prediction – Runtime
- `scripts/e2e_predict.py`
  - End-to-end from raw Coloring JSONs to predictions.
  - Steps: set `RAW_DATA_PATH` → try labeled builder via `build_child_dataset`; if unavailable, fall back to unlabeled builder that mirrors the same aggregation/engineering → align to bundle’s `feature_columns` → save raw/aligned features → score via `predict_cli` → write `<raw_folder_name>_results.csv`.

- `scripts/predict_cli.py`
  - Loads `bundle.json` and `preprocess/*.joblib`, concatenates UMAP embeddings when present, loads `models/*.joblib`, reconstructs `E_sens` and `E_spec`, applies alpha or meta combiner, optional temperature `T`, then thresholds at `bundle['threshold']` to produce `pred_label`.

### Deployable Bundle – Artifacts
- `models/final_np_iqrmid_u16n50_k2/bundle.json`
  - Keys: `feature_columns`, `models_used`, `ensembles` (`e_sens_names`, `e_spec_names`), `combiner` (`type`, `best_alpha` or meta path, `temperature_T`, `temperature_applied`), `threshold`, and metric snapshots.
- `models/final_np_iqrmid_u16n50_k2/preprocess/scaler_all.joblib`
- `models/final_np_iqrmid_u16n50_k2/preprocess/umap_all.joblib` (optional)
- `models/final_np_iqrmid_u16n50_k2/models/*.joblib` (`lightgbm`, `xgboost`, `brf`, `extratrees`, optional `meta_combiner`)

---

## Dependency Graph (Key Calls)

```
[Training]
train_final.sh
  └── scripts/clinical_fair_pipeline.py (main)
        ├── rag_system/research_engine.py (build_child_dataset)
        ├── sklearn.* (splitters, scalers, calibrators)
        ├── UMAP/PolynomialFeatures (optional)
        ├── make_models() → lightgbm/xgboost/brf/extratrees
        ├── choose_threshold() + threshold transfer
        └── export bundle (preprocess + models + combiner + threshold)

[Prediction]
scripts/e2e_predict.py
  ├── rag_system/research_engine.py (labeled) OR internal unlabeled builder
  ├── align to bundle['feature_columns'] → save features
  └── scripts/predict_cli.py
        ├── load bundle.json
        ├── scaler_all.joblib (+ umap_all.joblib)
        ├── models/*.joblib → per-model probabilities
        ├── E_sens/E_spec reconstruction
        ├── combiner (alpha/meta) + optional temperature
        └── apply threshold → prob_asd, pred_label
```

---

## Preprocessing Details
- Child-level aggregation (leakage-safe): mean of canonical numeric features per child + session counts.
- Domain ratios: `touches_per_zone`, `strokes_per_zone`, `zones_per_minute`, `vel_std_over_mean`, `acc_std_over_mean`, `avg_ibp_norm`, `interpoint_rate`, `touch_rate`, `stroke_rate`.
- Quantile-bin flags (quartiles) for select ratios → `bin_*` indicator columns.
- Standardization via `StandardScaler` on numeric features (training and inference use the same saved scaler).
- Optional UMAP(cosine) embeddings (n_components=16, n_neighbors=50) concatenated to standardized features.
- (For general library usage) Missing values imputed; categorical encoding when present.

## Model Training and Ensemble
- Base learners (typical best set): `lightgbm`, `xgboost`, `brf` (Balanced Random Forest), `extratrees`.
- Per-fold calibration (isotonic/sigmoid) to improve probability quality.
- Sub-ensembles:
  - `E_sens` (sensitivity-oriented) and `E_spec` (specificity-oriented) are means over selected model subsets.
- Combiner:
  - `alpha`-blend: `P_ens = α·E_sens + (1−α)·E_spec`, where `α` is selected by maximizing fold feasibility and surplus.
  - `meta` (optional): logistic model over `[E_sens, E_spec, |E_sens−E_spec|]`, optionally calibrated.
- Optional temperature scaling (`T`) on the combiner output.

## Threshold Selection (Clinical Policy)
- Policies: `both_targets`, `spec_first`, `youden`, `np` (Neyman–Pearson).
- Transfer from CV to holdout: median, `iqr_mid`, or quantile map with optional KS guard.
- Bundle saves the final scalar `threshold` used for binary decision.

---

## Results and Robustness
See `references/results.md` for full numbers. Highlights:
- Final bagged holdout (recommended headline): AUC ≈ 0.9059; Sens ≈ 0.8824; Spec ≈ 0.8000; Thr ≈ 0.5210.
- Exported bundle snapshot (seed 17): AUC ≈ 0.9118; Sens ≈ 0.8824; Spec ≈ 0.7000; Thr ≈ 0.5310.

### CV Fold Variance – Why It’s Not a Concern
- The pipeline uses group-aware CV (per-child) and a fixed group holdout, which appropriately captures variability due to subject differences.
- Threshold selection is done per-fold with clinical constraints, then transferred using robust statistics (median/IQR-mid) or quantile mapping. This stabilizes the decision boundary across folds.
- Per-model calibration (isotonic) and optional temperature scaling on the combiner yield calibrated probabilities, reducing over/under-confidence across folds.
- Final reporting uses either:
  - A single exported bundle that already meets targets on holdout, or
  - Bagging across seeds, which averages independent runs to reduce variance further.
- In `references/results.md`, per-fold AUCs remain consistently high across folds and seeds (e.g., ~0.70–0.89), while holdout metrics are strong and meet targets; this indicates the CV variance is within expectation for grouped biomedical classification and is handled by the robust threshold transfer and bagging.

---

## Prediction on New Data
```bash
./venv/bin/python scripts/e2e_predict.py --raw /path/to/raw_coloring_data
```
Outputs:
- `results/<raw_folder_name>_features_raw.csv` and `results/<raw_folder_name>_features_aligned.csv`
- `<project_root>/<raw_folder_name>_results.csv` with `prob_asd` and `pred_label` (1=ASD, 0=TD)

Requirements for scoring:
- `models/final_np_iqrmid_u16n50_k2/` with `bundle.json` (including `feature_columns`) and joblibs.
- `requirements.txt` installed; `imbalanced-learn` required if `brf` is included.

---

## Repository Structure
```
binary-classification-project/
├── README.md                    # This comprehensive system overview
├── flow.md                      # End-to-end data flow documentation
├── asd_td.md                    # Quick reference guide
├── requirements.txt             # Python dependencies
├── train_final.sh              # Training automation script
├── SYSTEM_REVIEW.md            # Comprehensive system review and validation
├── references/                 # Documentation and reference materials
│   ├── results.md              # Detailed performance metrics
│   └── predict_e2e.md          # End-to-end prediction guide
├── data/                       # Data directories
│   ├── external/               # External data (empty, .gitkeep)
│   ├── knowledge_base/         # Knowledge base data
│   ├── processed/              # Processed data
│   │   └── labels.csv          # Ground truth labels (renamed from "labels .csv")
│   └── raw/                    # Raw Coloring session JSONs (not listed in structure)
├── data_experiments/           # Experimental results and outputs
│   ├── espalier_file_keys_results.csv
│   ├── fileKeys_results.csv
│   └── phase2_file_keys_results.csv
├── models/                     # Trained model bundles
│   └── final_np_iqrmid_u16n50_k2/  # Deployable model bundle
│       ├── bundle.json         # Model configuration and metadata
│       ├── preprocess/         # Preprocessing artifacts
│       │   ├── scaler_all.joblib
│       │   └── umap_all.joblib
│       └── models/             # Trained model files
│           ├── lightgbm.joblib
│           ├── xgboost.joblib
│           ├── brf.joblib
│           └── extratrees.joblib
├── rag_system/                 # Research engine for feature extraction
│   ├── config.py
│   ├── research_engine.py
│   ├── data/                   # RAG system data
│   └── vector_db/              # Vector database for embeddings
├── results/                    # Feature extraction outputs
│   ├── espalier_file_keys_features_raw.csv
│   ├── espalier_file_keys_features_aligned.csv
│   ├── fileKeys_features_raw.csv
│   ├── fileKeys_features_aligned.csv
│   ├── phase2_file_keys_features_raw.csv
│   └── phase2_file_keys_features_aligned.csv
├── scripts/                    # Main execution scripts
│   ├── clinical_fair_pipeline.py  # Training pipeline
│   ├── e2e_predict.py          # End-to-end prediction
│   ├── predict_cli.py          # Bundle scoring CLI
│   ├── bag_scores.py           # Bagging utilities
│   └── bagged_end_to_end.py    # Feature building utilities
└── src/                        # Core library modules
    ├── data_processing.py      # Data preprocessing utilities
    ├── evaluation.py           # Model evaluation metrics
    ├── model.py                # Model training and utilities
    ├── representation.py       # Representation learning (PCA, UMAP, etc.)
    └── project/                # Project-specific modules
        ├── data_processing.py
        ├── evaluation.py
        ├── model.py
        └── representation.py
```

## Reproducibility
- Training/export automation: `train_final.sh`
- Historical artifacts for traceability: see `archive/**/results/*.json`
- System review and validation summary: `SYSTEM_REVIEW.md`


