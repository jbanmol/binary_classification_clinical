## End-to-End Prediction Guide (ASD vs TD)

This guide shows how to go from raw Coloring session JSON files to final ASD/TD predictions using `scripts/e2e_predict.py`. It includes all steps, the exact commands, and a plain-English description of the model ensemble and processing pipeline used under the hood.

### What you need
- A Unix-like shell (macOS/Linux; Windows WSL is fine)
- Python 3.10+ recommended
- Your trained bundle at `models/final_np_iqrmid_u16n50_k2/bundle.json` (already in this repo)
- Raw Coloring JSON data (files named like `Coloring_*.json`)

### 0) Project setup
```bash
python3 -m venv venv && ./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

**Note**: If you encounter UMAP loading issues on macOS, the system will automatically install compatible versions of `llvmlite` and `numba`.

Notes:
- Dependencies include: numpy, pandas, scikit-learn, lightgbm, xgboost, umap-learn, imbalanced-learn, joblib, etc.
- `imbalanced-learn` is required to load Balanced Random Forest models in the bundle when present.

### 1) Prepare raw data
Place your raw data folder under `data/raw/` or anywhere on disk. It should contain the Coloring session files named like `Coloring_*.json`. The helper will ask you to select a folder (interactive), or you can pass it using `--raw`.

**Recommended structure** (under `data/raw/`):
```
data/raw/
  phase2_file_keys/              # Example dataset
    child123/
      Coloring_2021-01-01 10:00:00.000000_child123.json
      Tracing_2021-01-01 10:05:00.000000_child123.json
      ...
    child456/
      Coloring_2021-01-05 12:34:56.000000_child456.json
      Tracing_2021-01-05 12:40:00.000000_child456.json
      ...
  espalier_file_keys/            # Another dataset
    ...
  fileKeys/                      # Another dataset
    ...
```

**Alternative structure** (anywhere on disk):
```
/path/to/my_raw_data/
  ├─ 2025-08-01/
  │   ├─ Coloring_abc123.json
  │   └─ ...
  ├─ 2025-08-02/
  │   ├─ Coloring_def456.json
  │   └─ ...
  └─ ...
```

### 2) Run end-to-end prediction
You can run interactively (it lists subfolders under `data/`), or non-interactively with `--raw`.

- **Non-interactive** (recommended):
```bash
# Using data under data/raw/
./venv/bin/python scripts/e2e_predict.py --raw data/raw/phase2_file_keys

# Using data anywhere on disk
./venv/bin/python scripts/e2e_predict.py --raw /path/to/my_raw_data
```

- **Interactive** (will prompt to select a folder under `data/`):
```bash
./venv/bin/python scripts/e2e_predict.py
```

What happens during this command (step-by-step):
1. The script sets `RAW_DATA_PATH` to your folder so the ingestion knows where to look.
2. It counts `Coloring_*.json` files for your sanity check.
3. It builds "child-level" features:
   - Parses all sessions via `rag_system.research_engine`.
   - Aggregates per child using mean statistics on canonical numeric features such as `velocity_mean`, `velocity_std`, `acc_magnitude_mean`, `session_duration`, `stroke_count`, `total_touch_points`, `unique_zones`, `avg_time_between_points`, etc.
   - Engineers domain ratios (all leakage-safe) including:
     - `touches_per_zone = total_touch_points / unique_zones`
     - `strokes_per_zone = stroke_count / unique_zones`
     - `zones_per_minute = unique_zones / (session_duration/60)`
     - `vel_std_over_mean = velocity_std / velocity_mean`
     - `acc_std_over_mean = acc_magnitude_std / acc_magnitude_mean`
     - `avg_ibp_norm = avg_time_between_points / session_duration`
     - `interpoint_rate = session_duration / avg_time_between_points`
     - `touch_rate = total_touch_points / session_duration`
     - `stroke_rate = stroke_count / session_duration`
   - Adds quantile-bin indicator flags (quartiles) for selected ratios: `bin_touch_rate_*`, `bin_strokes_per_zone_*`, `bin_vel_std_over_mean_*`, `bin_acc_std_over_mean_*`, `bin_zones_per_minute_*`, `bin_interpoint_rate_*`.
   - If labeled build fails (e.g., no labels available), it falls back to an unlabeled builder that mirrors the same transformations.
4. It aligns the resulting DataFrame columns to match the bundle’s `feature_columns` (missing columns are filled with 0 to maintain exact ordering/shape).
5. It saves two CSVs for reproducibility:
   - Raw features: `results/<raw_folder_name>_features_raw.csv`
   - Aligned features: `results/<raw_folder_name>_features_aligned.csv`
6. It loads the trained bundle from `models/final_np_iqrmid_u16n50_k2/bundle.json` and performs prediction with the saved preprocessing and ensemble.
7. It writes final predictions to `<project_root>/<raw_folder_name>_results.csv` with at least: `child_id, prob_asd, pred_label`.

### 3) Outputs
- `<project_root>/<raw_folder_name>_results.csv`: final predictions
- `results/<raw_folder_name>_features_raw.csv`: human-inspectable, engineered features per child
- `results/<raw_folder_name>_features_aligned.csv`: the exact feature matrix the model consumed

### Under the hood: model ensemble and preprocessing
This section describes exactly how the model processes your data and produces predictions. All of this is encoded in the exported bundle and the `scripts/predict_cli.py` logic.

- Feature alignment
  - The DataFrame columns are reindexed to `bundle['feature_columns']` to ensure training/inference parity.

- Preprocessing
  - Standardization: a saved `StandardScaler` (`preprocess/scaler_all.joblib`) transforms all numeric features to the same scale as training.
  - UMAP augmentation (cosine metric): when present in the bundle, `preprocess/umap_all.joblib` transforms standardized features into `n_components = 16` low-dimensional embeddings using `n_neighbors = 50`, `metric = cosine`. These embeddings are concatenated to the standardized features for the final model input.
  - Polynomial features are not used by the final exported bundle (kept disabled in production), so no polynomial transformer is applied at inference.

- Base models (per the final bundle)
  - Typical `models_used`: `lightgbm`, `xgboost`, `brf` (Balanced Random Forest), and `extratrees`.
  - Each model is loaded from `models/<name>.joblib` in the bundle directory and produces calibrated probabilities for the ASD class.
  - Calibration: models were trained with per-fold calibration (usually isotonic) during CV. At inference, we load the calibrated estimators as saved.

- Two sub-ensembles (for balanced performance)
  - E_sens: a mean of a subset of models emphasizing sensitivity.
  - E_spec: a mean of a subset of models emphasizing specificity.
  - The bundle stores which models feed each sub-ensemble under `ensembles.e_sens_names` and `ensembles.e_spec_names`.

- Combiner (two possibilities)
  - Alpha combiner: a fixed weight `alpha` in `[0,1]` combines `E_sens` and `E_spec`:
    - `P_ens = alpha * E_sens + (1 - alpha) * E_spec`
  - Meta combiner: a small logistic model blends `[E_sens, E_spec, |E_sens - E_spec|]` into `P_ens`. If present, it’s saved as `models/meta_combiner.joblib` and used automatically.

- Optional temperature calibration on the combiner
  - If the bundle has `combiner.temperature_applied = true`, then probabilities are passed through a temperature scaling with `T = combiner.temperature_T` to improve probability calibration.

- Final thresholding
  - The bundle includes a single scalar `threshold` selected by clinical policy (Neyman–Pearson or both-targets with fallbacks) based on cross-validation and then transferred to holdout using a robust rule (often `iqr_mid` of per-fold thresholds).
  - The final predicted label is `pred_label = 1` if `prob_asd >= threshold` else `0`.

### 4) Verifying your run
After the command completes, check:
- Predictions file exists: `<raw_folder_name>_results.csv`
- Feature CSVs exist in `results/`
- Quick peek at predictions:
```bash
head -n 5 <raw_folder_name>_results.csv | sed -n '1,5p'
```

### Troubleshooting
- "Best model bundle not found": Ensure `models/final_np_iqrmid_u16n50_k2/bundle.json` exists (it’s shipped in this repo) and relative paths are unchanged.
- "No Coloring_*.json files found": Confirm your `--raw` path points to a directory containing session JSONs.
- ImportError for `imblearn` when loading BRF: Install `imbalanced-learn` (already in `requirements.txt`); reinstall if your env predates the change.
- UMAP not available: `umap-learn` must be installed. The requirements include it. If still failing, ensure the bundle was actually exported with UMAP; otherwise the loader will skip it.

### Command recap
```bash
# 1) Create and activate environment
python3 -m venv venv && ./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# 2) Run end-to-end prediction from raw Coloring JSONs
./venv/bin/python scripts/e2e_predict.py --raw /path/to/my_raw_data

# 3) Find outputs
ls -l results/*_features_*.csv
ls -l *_results.csv
```

### Where the logic lives
- Prediction driver: `scripts/e2e_predict.py`
- Bundle scorer: `scripts/predict_cli.py`
- Training-time exporter (for context): `scripts/clinical_fair_pipeline.py` and `train_final.sh`
- Ensemble details and child-level feature builder used by both training and prediction: `scripts/bagged_end_to_end.py`

This completes the raw-to-prediction process with an explicit, stable ensemble and clinically oriented thresholding policy.

### Files used during an inference run
When you execute:
```bash
./venv/bin/python scripts/e2e_predict.py --raw /path/to/my_raw_data
```
these files and artifacts are used:

- Python entrypoints and helpers
  - `scripts/e2e_predict.py`
    - imports and uses:
      - `scripts.bagged_end_to_end.build_child_dataset` (labeled path)
      - `scripts.predict_cli.load_bundle`, `scripts.predict_cli.predict` (scoring)
      - `rag_system.research_engine.ColoringDataProcessor` (unlabeled fallback)
  - `scripts/predict_cli.py`

- RAG ingestion (feature building)
  - `rag_system/research_engine.py`
  - `rag_system/config.py` (reads `RAW_DATA_PATH`)
  - `rag_system/vector_db/` (runtime cache/db directory)

- Model bundle and artifacts (scoring)
  - `models/final_np_iqrmid_u16n50_k2/bundle.json`
    - Must include: `feature_columns`, `models_used`, `ensembles.e_sens_names`, `ensembles.e_spec_names`, `combiner`, `threshold`
  - `models/final_np_iqrmid_u16n50_k2/preprocess/scaler_all.joblib`
  - `models/final_np_iqrmid_u16n50_k2/preprocess/umap_all.joblib` (if present)
  - `models/final_np_iqrmid_u16n50_k2/models/*.joblib` for each model in `bundle['models_used']`, typically:
    - `models/final_np_iqrmid_u16n50_k2/models/lightgbm.joblib`
    - `models/final_np_iqrmid_u16n50_k2/models/xgboost.joblib`
    - `models/final_np_iqrmid_u16n50_k2/models/brf.joblib`
    - `models/final_np_iqrmid_u16n50_k2/models/extratrees.joblib`
  - `models/final_np_iqrmid_u16n50_k2/models/meta_combiner.joblib` (only if `combiner.type == 'meta'`)

- Raw inputs and outputs (data files)
  - Raw input folder passed via `--raw` (e.g., `data/raw/phase2_file_keys`)
    - All files matching `Coloring_*.json` inside that folder (recursively)
  - Output feature CSVs written by `e2e_predict.py`:
    - `results/<raw_folder_name>_features_raw.csv`
    - `results/<raw_folder_name>_features_aligned.csv`
  - Final predictions CSV:
    - `<project_root>/<raw_folder_name>_results.csv`


