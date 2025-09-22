# Clinical Modeling Progress Log (tillnow.md)

Purpose
- This document captures all modeling approaches, configurations, and results to date for the ASD/TD binary classification project, with emphasis on the clinical operating targets (Sensitivity ≥ 0.86 and Specificity ≥ 0.70).
- It serves as a living knowledge base for replication, analysis, and planning next steps.

Project context and data
- Ingestion and indexing via the internal RAG engine
  - Sessions processed: 823
  - Unique children (child-level unit): 270
  - Session-level label distribution: ASD 489, TD 334
- Train/holdout split
  - GroupShuffleSplit at child level: 80% train, 20% holdout
- Cross-validation
  - StratifiedGroupKFold with 5 folds on training (child-level groups). Falls back to StratifiedKFold if SGK is unavailable.

Clinical targets and thresholding policy
- Objective: select an operating threshold that meets BOTH Sensitivity ≥ 0.86 and Specificity ≥ 0.70.
- Per-fold threshold selection in CV:
  1) Prefer thresholds meeting both targets.
  2) If none, spec-first fallback (maximize sensitivity subject to Spec ≥ target).
  3) Else, Youden’s J (tpr - fpr).
- Holdout threshold transfer:
  - Baseline: median of selected per-fold thresholds.
  - Quantile-mapped transfer (optional): maps CV fold thresholds to holdout score quantiles to reduce transfer instability under distribution shift.

Feature and representation pipeline (child-level; leak-safe)
- Canonical numeric signals (means at child level):
  velocity_mean, velocity_std, velocity_max, velocity_cv,
  tremor_indicator, acc_magnitude_mean, acc_magnitude_std,
  palm_touch_ratio, unique_fingers, max_finger_id,
  session_duration, stroke_count, total_touch_points,
  unique_zones, unique_colors, final_completion,
  completion_progress_rate, avg_time_between_points,
  canceled_touches, session_count
- Engineered domain ratios (computed on aggregated features without label leakage):
  - touches_per_zone = total_touch_points / (unique_zones + eps)
  - strokes_per_zone = stroke_count / (unique_zones + eps)
  - zones_per_minute = unique_zones / (session_duration/60 + eps)
  - vel_std_over_mean = velocity_std / (velocity_mean + eps)
  - acc_std_over_mean = acc_magnitude_std / (acc_magnitude_mean + eps)
  - avg_ibp_norm = avg_time_between_points / (session_duration + eps)
  - interpoint_rate = (session_duration + eps) / (avg_time_between_points + eps)
  - touch_rate = total_touch_points / (session_duration + eps)
  - stroke_rate = stroke_count / (session_duration + eps)
- Quantile-binned indicators (one-hot) for key ratios:
  touch_rate, strokes_per_zone, vel_std_over_mean, acc_std_over_mean, zones_per_minute, interpoint_rate
- Representations:
  - StandardScaler (per fold and full-train contexts).
  - UMAP (metric=cosine): per-fold leak-safe embeddings concatenated to scaled features; typical n_components ∈ {10, 16}, n_neighbors=50.
  - PolynomialFeatures (degree=2, include_bias=False) as a UMAP alternative.

Model families and grids
- LightGBM grid:
  - feature_fraction ∈ {0.6, 0.8}
  - min_child_samples ∈ {20, 40, 60}
  - scale_pos_weight ∈ {0.75, 1.0, 1.25}
  - common: learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.8, random_state=42
- XGBoost grid (expanded):
  - max_depth ∈ {3,4,5}, min_child_weight ∈ {1,3,5}
  - scale_pos_weight ∈ {0.75, 1.0, 1.25}
  - regularization explorations: reg_alpha ∈ {0.0, 0.5, 1.0}, gamma ∈ {0.0, 0.5}
- Balanced Random Forest (BRF): n_estimators=400, max_features='sqrt', min_samples_leaf=6
- ExtraTreesClassifier (conservative)
- Logistic Regression: class_weight='balanced', max_iter=2000, C≈0.5
- Calibration: per-fold isotonic or sigmoid; optional final calibration for holdout.

Clinical ensemble/combiner design (constrained)
- Build per-model OOF probabilities in CV; compute model AUCs.
- Define two sub-ensembles:
  - E_sens (sensitivity-oriented): typically best LGBM/XGB + BRF
  - E_spec (specificity-oriented): typically best LGBM/XGB + ExtraTrees
- Alpha-blend combiner:
  - p = α * E_sens + (1 - α) * E_spec, α ∈ {0.0, 0.05, …, 1.0}
  - Select α by maximizing (#feasible folds meeting both targets, then surplus over targets).
- Optional meta-combiner:
  - LogisticRegression over [z_sens, z_spec, |z_sens - z_spec|], leak-safe per fold, with calibration.
  - Use meta if (feasible_folds, surplus) surpasses alpha-blend.
- Thresholding performed on the selected combiner’s scores.

Earlier approaches (pre-combiner and soft-voting)
- Polynomial feature expansion + conservative ExtraTreesClassifier and a clinical soft-voting ensemble, with cloning via sklearn.clone to prevent state leakage.
- Per-fold calibration; spec-first policy.
- Result: No single model, nor the soft-voting ensemble, achieved both targets jointly on CV; motivated exploration of richer features/architectures.

Key experiments and results (artifacts in results/)

1) Baseline clinical fair pipeline (Polynomial; LightGBM+BRF+LogReg)
- File: results/clinical_fair_pipeline_results.json
- Settings: use_polynomial=true, calibration=isotonic, models=lightgbm,brf,logreg
- CV means: Sens 0.783, Spec 0.798, AUC 0.823; median τ≈0.5539
- Holdout: Sens 0.818, Spec 0.667, AUC 0.818; τ≈0.5539
- Outcome: Close on sensitivity, specificity short of target.

2) LGBM grid (with Poly); ExtraTrees/BRF/LogReg included
- File: results/clinical_fair_lgbmgrid_results.json
- CV means: Sens 0.799, Spec 0.730, AUC 0.829; median τ≈0.5094
- Holdout: Sens 0.818, Spec 0.571, AUC 0.830
- Outcome: Higher CV AUC; holdout specificity too low.

3) LGBM+XGB expanded grids (with Poly)
- File: results/clinical_fair_lgbm_xgb_grid_results.json
- CV means: Sens 0.806, Spec 0.797, AUC 0.830; median τ≈0.5222
- Holdout: Sens 0.788, Spec 0.619, AUC 0.820
- Outcome: Improved CV, modest holdout specificity.

4) UMAP (cosine) representation; broader model family (no Poly)
- File: results/clinical_fair_umap_cosine_results.json
- CV means: Sens 0.790, Spec 0.820, AUC 0.834; median τ≈0.4533
- Holdout: Sens 0.818, Spec 0.571, AUC 0.810
- Outcome: Sensitivity strong, specificity short on holdout.

5) Constrained combiner with UMAP (cosine) — older combined run
- File: results/clinical_fair_combiner_umap.json
- CV means: Sens 0.790, Spec 0.820, AUC 0.834; median τ≈0.4533
- Holdout: Sens 0.818, Spec 0.619, AUC 0.798
- Outcome: Similar to (4) with slightly higher holdout specificity.

6) Constrained combiner — UMAP 10, top-3, no quantile mapping
- File: results/clinical_fair_combiner_umap10_top3.json
- CV means: Sens 0.816, Spec 0.765, AUC 0.839; median τ≈0.516
- Holdout: Sens 0.848, Spec 0.667, AUC 0.805; τ≈0.492
- Holdout (spec-first diagnostic): Sens 0.848, Spec 0.762; τ≈0.5588
- Combiner: best_alpha=0.2
- Outcome: Very strong sensitivity on holdout; specificity slightly under 0.70 by default but can reach ~0.76 at same sensitivity under spec-first—still not both targets simultaneously.

7) Constrained combiner — UMAP 10, top-3, quantile-mapped threshold
- File: results/clinical_fair_combiner_umap10_top3_quantile.json
- CV means: Sens 0.789, Spec 0.800, AUC 0.846; median τ≈0.498
- Holdout (quantile-mapped): Sens 0.727, Spec 0.667, AUC 0.786; τ≈0.575
- Holdout (spec-first diagnostic): Sens 0.667, Spec 0.714; τ≈0.6207
- Combiner: best_alpha=0.0
- Outcome: Quantile mapping improved specificity over some prior runs but reduced sensitivity.

8) Constrained combiner — UMAP 16, top-2, quantile-mapped threshold [Latest]
- File: results/clinical_fair_combiner_umap16_top2_quantile.json
- Settings (command):
  ```bash path=null start=null
  ./venv/bin/python scripts/clinical_fair_pipeline.py \
    --target-sens 0.86 --target-spec 0.70 \
    --use-umap-cosine --umap-components 16 --umap-neighbors 50 \
    --calibration isotonic --final-calibration isotonic \
    --models lgbm_grid,xgb_grid,brf,extratrees --top-k-models 2 \
    --report-holdout-specfirst --use-quantile-threshold \
    --out-name clinical_fair_combiner_umap16_top2_quantile.json
  ```
- CV means: Sens 0.788, Spec 0.807, AUC 0.840; median τ≈0.521
- Holdout (quantile-mapped): Sens 0.727, Spec 0.762, AUC 0.808; τ≈0.613
- Holdout (spec-first diagnostic): Sens 0.788, Spec 0.714; τ≈0.5713
- Combiner: best_alpha=0.0
- Outcome: Best overall holdout trade-off to date (Spec ~0.76 at Sens ~0.73; highest recent holdout AUC).

9) Constrained combiner — GBM-only (LGBM+XGB) top-2, UMAP 10, quantile-mapped
- File: results/clinical_fair_combiner_umap10_lgbm_xgb_top2_quantile.json
- CV means: Sens 0.789, Spec 0.800, AUC 0.847; median τ≈0.497
- Holdout (quantile-mapped): Sens 0.697, Spec 0.714, AUC 0.759; τ≈0.613
- Holdout (spec-first diagnostic): Sens 0.697, Spec 0.714; τ≈0.616
- Combiner: best_alpha=0.0
- Outcome: Lower holdout AUC/metrics than (8); variance reduction didn’t improve the clinical trade-off.

10) Constrained combiner — Polynomial (deg=2), top-3, quantile-mapped
- File: results/clinical_fair_combiner_poly_top3_quantile.json
- CV means: Sens 0.763, Spec 0.817, AUC 0.800; median τ≈0.490
- Holdout (quantile-mapped): Sens 0.636, Spec 0.762, AUC 0.784; τ≈0.676
- Holdout (spec-first diagnostic): Sens 0.727, Spec 0.762; τ≈0.660
- Combiner: best_alpha=0.0
- Outcome: Similar specificity to (8) but notably lower sensitivity; UMAP-based representations are more favorable here.

11) TPOT-related experiments
- results/clinical_fair_with_tpot.json and results/clinical_fair_with_tpot_20m.json
  - Settings reflected the baseline (Polynomial; LightGBM+BRF+LogReg) with CV/holdout identical to (1):
    - Holdout: Sens 0.818, Spec 0.667, AUC 0.818; τ≈0.5539
- results/clinical_fair_pipeline_tpot_results.json
  - Auto-generated pipeline (RandomForest-focused) summary
  - Holdout: Sens 0.788, Spec 0.714, AUC 0.789; τ≈0.591
- Outcome: TPOT offered alternative pipelines but did not reach the clinical targets jointly; performance comparable to earlier baselines.

12) Auxiliary model summaries (for reference)
- results/model_results.json and results/model_results_umap.json capture per-model CV metrics and holdout summaries for Random Forest, Gradient Boosting, Logistic Regression, etc. They corroborate that high CV sensitivity often pairs with lower specificity on this dataset, reinforcing the need for targeted threshold/ensemble strategies.

Comparative summary (holdout)
- Best overall recent trade-off: UMAP 16 + top-2 + quantile-threshold (Sens ~0.727, Spec ~0.762, AUC ~0.808).
- Highest holdout sensitivity observed in recent ensembles: ~0.848 (UMAP 10, top-3, no quantile), but specificity slightly below 0.70 at default; spec-first diagnostic improved Spec to ~0.762 at the same Sens, yet still not both targets simultaneously when transferring CV thresholds.
- Quantile mapping tends to increase holdout specificity stability but can reduce sensitivity; tuning neighbors/components and final calibration can partially mitigate this.

Findings and interpretations
- CV/holdout gap: Strong CV does not always translate to holdout; indicates distributional differences and/or calibration/threshold-transfer instability.
- Combiner behavior: best_alpha frequently resolves to 0.0, favoring specificity-oriented sub-ensembles; meta-combiner is available and can be toggled if feasibility/surplus indicates benefit.
- Representation choice: UMAP (cosine) with moderate n_components (10–16) generally outperforms polynomial-only expansion under the constrained combiner with current features.
- Feature engineering: Ratio features and quantile bins are helpful; further domain-informed features or multi-modal inputs are likely needed to reach Sens ≥ 0.86 at Spec ≥ 0.70.

Reproduction (excerpt)
- Environment (from WARP.md):
  ```bash path=null start=null
  # Create and activate venv
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate

  # Install dependencies
  pip install -r requirements.txt

  # Create necessary directories
  python config.py
  ```
- MLflow UI (optional):
  ```bash path=null start=null
  mlflow ui --backend-store-uri ./mlruns
  # http://localhost:5000
  ```
- Example training/evaluation commands (representative):
  ```bash path=null start=null
  ./venv/bin/python scripts/clinical_fair_pipeline.py \
    --target-sens 0.86 --target-spec 0.70 \
    --use-umap-cosine --umap-components 16 --umap-neighbors 50 \
    --calibration isotonic --final-calibration isotonic \
    --models lgbm_grid,xgb_grid,brf,extratrees --top-k-models 2 \
    --report-holdout-specfirst --use-quantile-threshold \
    --out-name clinical_fair_combiner_umap16_top2_quantile.json
  ```

Open issues and next steps
- Threshold robustness
  - Multi-seed evaluation for CV and quantile threshold transfer to quantify and reduce variability.
  - Consider selecting thresholds from a percentile band (e.g., IQR) for conservative transfer.
- Ensemble refinements
  - Starting from the current best config (UMAP-16/top-2), test top-k ∈ {2,3}, vary final calibration (sigmoid vs isotonic), and evaluate meta-combiner vs alpha-blend.
- Feature augmentation
  - Add more domain ratios, temporal stability/variability statistics, and explore monotonic constraints for LightGBM if clinically justified.
- Distribution shift diagnostics
  - Plot score distributions (CV vs holdout), calibration curves, ROC/PR; perform per-child error analysis for subgroup effects.
- Data enrichment
  - If feasible, integrate richer multi-modal features to improve separability while preserving specificity.

Artifact index
- results/clinical_fair_pipeline_results.json
- results/clinical_fair_lgbmgrid_results.json
- results/clinical_fair_lgbm_xgb_grid_results.json
- results/clinical_fair_umap_cosine_results.json
- results/clinical_fair_combiner_umap.json
- results/clinical_fair_combiner_umap10_top3.json
- results/clinical_fair_combiner_umap10_top3_quantile.json
- results/clinical_fair_combiner_umap16_top2_quantile.json
- results/clinical_fair_combiner_umap10_lgbm_xgb_top2_quantile.json
- results/clinical_fair_combiner_poly_top3_quantile.json
- results/clinical_fair_with_tpot.json
- results/clinical_fair_with_tpot_20m.json
- results/clinical_fair_pipeline_tpot_results.json
- results/model_results.json
- results/model_results_umap.json

Current recommendation
- Use the UMAP 16, top-2, quantile-threshold configuration (results/clinical_fair_combiner_umap16_top2_quantile.json) as the working candidate while pursuing the next steps listed above to attempt to reach Sens ≥ 0.86 at Spec ≥ 0.70 on holdout.

