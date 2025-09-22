## Binary Classification: Best Pipeline Summary 

### Task
ASD binary classification (positive class label = 1). Report best-performing pipeline with sensitivity/specificity for CV and final holdout, plus full configuration for reproducibility.

### Best Pipeline (Final)
- **Approach**: Bagged ensemble across seeds of calibrated model blend with UMAP-based representation
- **Models**: `lightgbm`, `xgboost`, `brf` (BalancedRandomForest), `extratrees`
- **Representation**: UMAP with cosine metric, **components=16**, **neighbors=50**
- **Calibration**: Per-model isotonic; final temperature scaling applied (bundle `T≈1.3579`)
- **Threshold policy**: `np` (Neyman–Pearson style), **threshold transfer**: `iqr_mid`
- **Seeds bagged**: 29, 43 (per the final bagged results set)
- **Ensemble composition**:
  - Sensitivity-focused ensemble `E_sens`: `lightgbm`, `xgboost`, `brf`
  - Specificity-focused ensemble `E_spec`: `lightgbm`, `xgboost`, `extratrees`
  - Alpha combiner (blend of `E_sens` and `E_spec`). Best α varies by seed; export bundle uses α=0.0

### Final Holdout Metrics (Best to Report)
- Source: `archive/cleanup_everything_20250916_002251/results/final_bagged_ho777_np_iqrmid_temp_u16n50_k2.json`
- **AUC**: 0.9059
- **Sensitivity**: 0.8824
- **Specificity**: 0.8000
- **Decision threshold**: ≈ 0.5210

Quick view:
| Run | AUC | Sensitivity | Specificity | Threshold |
|---|---:|---:|---:|---:|
| Final bagged (recommended) | 0.9059 | 0.8824 | 0.8000 | ~0.5210 |
| Export bundle snapshot | 0.9118 | 0.8824 | 0.7000 | ~0.5310 |

### Per-Seed Holdout (for reference)
- Seed 29 (`final_s29_ho777_np_iqrmid_temp_u16n50_k2.json`): AUC=0.9059, Sens=0.8529, Spec=0.8000, Thr≈0.5476
- Seed 43 (`final_s43_ho777_np_iqrmid_temp_u16n50_k2.json`): AUC=0.9081, Sens=0.8235, Spec=0.8500, Thr≈0.5292

### Cross-Validation (CV) Summary
Reported per-seed CV means (5-fold CV on training split before holdout). These accompany the seeds used in the final bag.
- Seed 29: Sensitivity mean ≈ 0.8033, Specificity mean ≈ 0.7348, AUC mean ≈ 0.8108, CV median threshold ≈ 0.5516
- Seed 43: Sensitivity mean ≈ 0.7400, Specificity mean ≈ 0.7832, AUC mean ≈ 0.8024, CV median threshold ≈ 0.5878

#### Seed 29 per-fold CV
| Fold | Threshold | Sensitivity | Specificity | AUC |
|---|---:|---:|---:|---:|
| 1 | 0.5175568362319168 | 0.7142857142857143 | 0.75 | 0.7209821428571429 |
| 2 | 0.5516222686809132 | 0.8260869565217391 | 0.7 | 0.817391304347826 |
| 3 | 0.46246812824193945 | 0.8461538461538461 | 0.7647058823529411 | 0.8484162895927602 |
| 4 | 0.5640023870972477 | 0.88 | 0.7222222222222222 | 0.8911111111111111 |
| 5 | 0.5808627755494855 | 0.75 | 0.736842105263158 | 0.7763157894736843 |

#### Seed 43 per-fold CV
| Fold | Threshold | Sensitivity | Specificity | AUC |
|---|---:|---:|---:|---:|
| 1 | 0.4931598555334425 | 0.7037037037037037 | 0.75 | 0.7592592592592593 |
| 2 | 0.5914741773048229 | 0.6923076923076923 | 0.8333333333333334 | 0.8141025641025641 |
| 3 | 0.5877671276529197 | 0.8888888888888888 | 0.72 | 0.8511111111111112 |
| 4 | 0.5096913070237937 | 0.7857142857142857 | 0.8 | 0.8095238095238095 |
| 5 | 0.7203864342860259 | 0.6296296296296297 | 0.8125 | 0.7777777777777778 |

Exported bundle CV snapshot (seed 17 reference bundle):
- Path: `models/final_np_iqrmid_u16n50_k2/bundle.json`
- CV means: Sensitivity ≈ 0.7220, Specificity ≈ 0.7563, AUC ≈ 0.7813, CV median threshold ≈ 0.5314

Per-fold details (bundle snapshot):
| Fold | Threshold | Sensitivity | Specificity | AUC |
|---|---:|---:|---:|---:|
| 1 | 0.5133215747316103 | 0.75 | 0.75 | 0.7946428571428572 |
| 2 | 0.5395438255332604 | 0.8 | 0.7777777777777778 | 0.8444444444444446 |
| 3 | 0.49526712704621895 | 0.7692307692307693 | 0.7647058823529411 | 0.8031674208144796 |
| 4 | 0.7425963963217372 | 0.55 | 0.7391304347826086 | 0.6956521739130435 |
| 5 | 0.5313595548852716 | 0.7407407407407407 | 0.75 | 0.7685185185185185 |

### Configuration Details
- Targets: Sensitivity ≥ 0.86, Specificity ≥ 0.70
- UMAP: cosine metric, components=16, neighbors=50
- Models list: `lightgbm,xgboost,brf,extratrees`
- Calibration: isotonic per-model; final temperature scaling in export bundle (T≈1.3579)
- Thresholding: policy=`np`, transfer=`iqr_mid`
- Combiner α (seed-dependent):
  - Seed 29: α≈0.9
  - Seed 43: α≈0.0
  - Export bundle: α=0.0
- Bundle decision threshold: ≈ 0.5310 (bundle holdout meets targets)

### Reproducibility
- Script: `train_final.sh`
  - Creates venv, installs `requirements.txt`
  - Runs `scripts/clinical_fair_pipeline.py` for seeds (e.g., 17/29/43) with UMAP, isotonic calibration, `np` policy, `iqr_mid` transfer, and temperature scaling enabled for export
  - Bags results via `scripts/bag_scores.py`
  - Exports reference bundle to `models/final_np_iqrmid_u16n50_k2/`

### Key Artifacts and Paths
- **Final bagged metrics JSON**: `archive/cleanup_everything_20250916_002251/results/final_bagged_ho777_np_iqrmid_temp_u16n50_k2.json`
- **Per-seed JSONs**:
  - `archive/cleanup_everything_20250916_002251/results/final_s29_ho777_np_iqrmid_temp_u16n50_k2.json`
  - `archive/cleanup_everything_20250916_002251/results/final_s43_ho777_np_iqrmid_temp_u16n50_k2.json`
- **Exported bundle (deployable)**: `models/final_np_iqrmid_u16n50_k2/`
  - `bundle.json` contains ensemble names (`E_sens`/`E_spec`), α, temperature `T`, threshold, holdout metrics
- **Experimental results**: `data_experiments/`
  - `espalier_file_keys_results.csv`
  - `fileKeys_results.csv` 
  - `phase2_file_keys_results.csv`
- **Feature extraction outputs**: `results/`
  - `*_features_raw.csv` (human-readable features)
  - `*_features_aligned.csv` (model-ready features)

### Notes
- The bagged metrics are the recommended headline numbers for reporting and comparison.
- The bundle’s holdout snapshot also meets the clinical targets (Sens=0.8824, Spec=0.7000, AUC=0.9118) and can be used for deployment.


### How to Test on a New Dataset (Simple Steps)
Think of this like using a calculator: you give it a table (CSV) with the right columns, and it gives you a score for each row and a Yes/No label.

1) Prepare your CSV
- Make a CSV with one row per child/sample.
- Include all feature columns expected by the model bundle.
- To see the exact column names expected, print them from the bundle:

```bash
./venv/bin/python - << 'PY'
import json
bundle = json.load(open('models/final_np_iqrmid_u16n50_k2/bundle.json'))
print('\n'.join(bundle.get('feature_columns', [])))
PY
```

2) Create the Python environment
```bash
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

3) Run predictions
```bash
./venv/bin/python scripts/predict_cli.py \
  --bundle models/final_np_iqrmid_u16n50_k2/bundle.json \
  --in path/to/your_features.csv \
  --out predictions.csv
```

What you get
- `predictions.csv` will include:
  - `prob_asd`: the model’s probability that the sample is ASD (between 0 and 1)
  - `pred_label`: the final prediction (1 = ASD, 0 = TD) using the bundle’s threshold (~0.5310 in the export)

Behind the scenes (plain language)
- We standardize your features using the same scaler as training.
- We add UMAP features (a compact summary of the data) with the same settings used in training.
- We run four trained models (`lightgbm`, `xgboost`, `brf`, `extratrees`).
- We blend their probabilities using the best recipe we found (alpha combiner), and apply temperature scaling to calibrate the scores.
- We apply a decision threshold to turn the score into a Yes/No label.

Which pipeline is best?
- Best for reporting and general use: the bagged ensemble across seeds (headline metrics above).
- Best single deployable artifact: the exported bundle in `models/final_np_iqrmid_u16n50_k2/` (easy to run and already meets targets on holdout).


