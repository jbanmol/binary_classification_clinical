## End-to-End Data Flow: Raw Data → Features → Final Predictions

This document describes the complete data flow from raw Coloring session JSONs to final ASD/TD predictions, including all intermediate steps, file locations, and transformations.

### 1) Raw Data Input
Raw Coloring session JSON files are organized in folders under `data/raw/`:
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

**Note**: Raw data contents are intentionally not listed in the main repository structure to keep documentation focused on code and artifacts.

### 2) Feature Extraction (Child-Level)
**Script**: `scripts/e2e_predict.py`
**Engine**: `rag_system/research_engine.py`

#### Process:
1. **Session Processing**: Reads all `Coloring_*.json` files from the specified raw folder
2. **Feature Extraction**: Extracts per-session behavioral features including:
   - Movement metrics: `velocity_mean`, `velocity_std`, `velocity_max`, `velocity_cv`
   - Touch patterns: `palm_touch_ratio`, `unique_fingers`, `max_finger_id`
   - Session statistics: `session_duration`, `stroke_count`, `total_touch_points`, `unique_zones`, `unique_colors`
   - Completion metrics: `final_completion`, `completion_progress_rate`, `avg_time_between_points`, `canceled_touches`
3. **Child-Level Aggregation**: Groups by `child_id` and computes mean statistics (leakage-safe)
4. **Domain Engineering**: Adds derived ratios:
   - `touches_per_zone = total_touch_points / unique_zones`
   - `strokes_per_zone = stroke_count / unique_zones`
   - `zones_per_minute = unique_zones / (session_duration/60)`
   - `vel_std_over_mean = velocity_std / velocity_mean`
   - `acc_std_over_mean = acc_magnitude_std / acc_magnitude_mean`
   - `avg_ibp_norm = avg_time_between_points / session_duration`
   - `interpoint_rate = session_duration / avg_time_between_points`
   - `touch_rate = total_touch_points / session_duration`
   - `stroke_rate = stroke_count / session_duration`
5. **Quantile Binning**: Creates 4-bin indicator flags for selected ratios (`bin_*` columns)

#### Output Files:
- **Raw features**: `results/<raw_folder_name>_features_raw.csv` (human-readable)
- **Aligned features**: `results/<raw_folder_name>_features_aligned.csv` (model-ready, matches bundle schema)

### 3) Model Inference Pipeline
**Script**: `scripts/predict_cli.py`
**Bundle**: `models/final_np_iqrmid_u16n50_k2/`

#### Preprocessing Steps:
1. **Feature Alignment**: Reorder columns to match `bundle['feature_columns']` (53 base features)
2. **Standardization**: Apply `preprocess/scaler_all.joblib` (StandardScaler from training)
3. **UMAP Embedding**: Apply `preprocess/umap_all.joblib` (16 components, cosine metric, n_neighbors=50)
4. **Feature Concatenation**: Combine standardized features (53) + UMAP embeddings (16) = 69 total features

#### Model Ensemble:
1. **Base Models** (from `bundle['models_used']`):
   - `lightgbm.joblib` - LightGBM classifier
   - `xgboost.joblib` - XGBoost classifier  
   - `brf.joblib` - Balanced Random Forest
   - `extratrees.joblib` - Extra Trees classifier

2. **Sub-Ensembles**:
   - **E_sens** (sensitivity-focused): `lightgbm`, `xgboost`, `brf`
   - **E_spec** (specificity-focused): `lightgbm`, `xgboost`, `extratrees`

3. **Combiner**: Alpha blend `P_ens = α·E_sens + (1-α)·E_spec` where α=0.0 (from bundle)

4. **Temperature Scaling**: Apply T≈1.3579 for probability calibration

5. **Thresholding**: Use bundle threshold τ≈0.5310 for final binary decision

#### Output Files:
- **Final predictions**: `<raw_folder_name>_results.csv` (at project root)
- **Feature snapshots**: `results/<raw_folder_name>_features_*.csv` (for transparency)

### 4) Execution Commands

#### End-to-End Prediction:
```bash
# Interactive (prompts for folder selection)
./venv/bin/python scripts/e2e_predict.py

# Non-interactive (specify raw data folder)
./venv/bin/python scripts/e2e_predict.py --raw data/raw/phase2_file_keys
```

#### Direct Bundle Scoring (if you have aligned features):
```bash
./venv/bin/python scripts/predict_cli.py \
  --bundle models/final_np_iqrmid_u16n50_k2/bundle.json \
  --in results/phase2_file_keys_features_aligned.csv \
  --out predictions.csv
```

### 5) Complete Data Flow Summary

```
Raw JSONs → Feature Extraction → Preprocessing → Model Inference → Predictions
     ↓              ↓                    ↓              ↓              ↓
data/raw/    results/*_features_*    StandardScaler   Ensemble      *_results.csv
             .csv                    + UMAP           + Threshold
```

### 6) Key Files and Their Roles

| File/Directory | Purpose | Used By |
|---|---|---|
| `data/raw/*/` | Raw Coloring session JSONs | `e2e_predict.py` |
| `data/processed/labels.csv` | Ground truth labels | Training only |
| `rag_system/research_engine.py` | Feature extraction engine | `e2e_predict.py` |
| `results/*_features_raw.csv` | Human-readable features | Transparency |
| `results/*_features_aligned.csv` | Model-ready features | `predict_cli.py` |
| `models/final_np_iqrmid_u16n50_k2/` | Deployable model bundle | `predict_cli.py` |
| `*_results.csv` | Final predictions | End user |

### 7) Notes
- **No retraining**: This is the production inference path using the pre-trained best model
- **Fallback handling**: If labels are unavailable, uses unlabeled-safe feature builder
- **Reproducibility**: All preprocessing and model parameters are saved in the bundle


