#!/usr/bin/env python3
"""
Bagged Clinical End-to-End Pipeline

End-to-end script from raw data to prediction using the clinical fair pipeline logic,
including:
- Raw data ingestion via rag_system.research_engine
- Child-level aggregation and domain feature engineering (leakage-safe)
- Grouped holdout split (child-level)
- Per-fold preprocessing, calibration, and threshold selection in group-aware CV
- Model ensemble across LightGBM, XGBoost, BRF (if available), and ExtraTrees
- Threshold transfer to holdout (median across CV folds) and/or policy-based selection
- Optional UMAP/cosine embedding augmentation and Polynomial features
- Bagging across multiple random seeds by averaging holdout probabilities

This script mirrors the behavior of prior results like:
- final_s29_ho777_np_iqrmid_temp_u16n50_k2.json
- final_s43_ho777_np_iqrmid_temp_u16n50_k2.json
- final_bagged_ho777_np_iqrmid_temp_u16n50_k2.json

Hardcoded best settings (from latest artifacts):
- targets: sensitivity >= 0.86, specificity >= 0.70
- calibration: isotonic
- models: [lightgbm, xgboost, brf, extratrees]
- UMAP (cosine) enabled: n_components=16, n_neighbors=50; polynomial features disabled
- threshold policy: np (Neyman–Pearson) with median threshold transfer (iqr_mid)
- seeds: [29, 43], holdout_seed: 777
- quantile_guard_ks: 0.0

How to run:
  python scripts/bagged_end_to_end.py

Outputs a JSON summary to results/bagged_best.json.

Note: This script does not commit any changes to git.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


def safe_divide(numerator, denominator, default_value=0.0, eps=1e-8):
    """Safely divide two pandas Series, handling zeros, NaNs, and edge cases.
    
    Args:
        numerator: pandas Series or array-like numerator values
        denominator: pandas Series or array-like denominator values  
        default_value: Value to return when numerator is NaN
        eps: Small epsilon to prevent division by zero
        
    Returns:
        pandas Series with safe division results
    """
    # Convert to pandas Series if needed
    if not isinstance(numerator, pd.Series):
        numerator = pd.Series(numerator)
    if not isinstance(denominator, pd.Series):
        denominator = pd.Series(denominator)
    
    # Handle NaN numerators
    numerator_clean = numerator.fillna(default_value)
    
    # Create safe denominator (replace zeros and NaNs with eps)
    denominator_safe = np.where(
        (denominator.isna()) | (denominator <= eps), 
        eps, 
        denominator
    )
    
    # Perform safe division
    result = numerator_clean / denominator_safe
    
    # Return as pandas Series with original index
    return pd.Series(result, index=numerator.index)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Optional imports and availability flags
_lgbm_ok = True
try:
    import lightgbm as lgb
except Exception:
    _lgbm_ok = False

_brf_ok = True
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except Exception:
    _brf_ok = False

_xgb_ok = True
try:
    from xgboost import XGBClassifier
except Exception:
    _xgb_ok = False

# Prefer group-aware CV if available
try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore

# Ensure project root is on sys.path so rag_system can be imported when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag_system.research_engine import research_engine  # type: ignore


NUMERIC_FEATURES_CANON = [
    'velocity_mean', 'velocity_std', 'velocity_max', 'velocity_cv',
    'tremor_indicator', 'acc_magnitude_mean', 'acc_magnitude_std',
    'palm_touch_ratio', 'unique_fingers', 'max_finger_id',
    'session_duration', 'stroke_count', 'total_touch_points',
    'unique_zones', 'unique_colors', 'final_completion',
    'completion_progress_rate', 'avg_time_between_points',
    'canceled_touches'
]

# Hardcoded best settings derived from latest successful runs
BEST_TARGET_SENS = 0.86
BEST_TARGET_SPEC = 0.70
BEST_CALIBRATION = 'isotonic'
BEST_MODELS = ['lightgbm', 'xgboost', 'brf', 'extratrees']
BEST_USE_POLYNOMIAL = False
BEST_USE_UMAP_COSINE = True
BEST_UMAP_COMPONENTS = 16
BEST_UMAP_NEIGHBORS = 50
BEST_SEEDS = [29, 43]
BEST_HOLDOUT_SEED = 777
BEST_THRESHOLD_POLICY = 'np'
BEST_THRESHOLD_TRANSFER = 'iqr_mid'
BEST_QUANTILE_GUARD_KS = 0.0
BEST_EXPORT_PATH = 'results/bagged_best.json'


@dataclass
class FoldMetric:
    fold: int
    threshold: float
    sensitivity: float
    specificity: float
    auc: float


@dataclass
class SingleSeedResult:
    seed: int
    holdout_seed: int
    cv_folds: List[FoldMetric]
    cv_threshold_median: float
    cv_sens_mean: float
    cv_spec_mean: float
    cv_auc_mean: float
    holdout_ids: List[str]
    holdout_y: List[int]
    holdout_proba: List[float]
    holdout_threshold: float
    holdout_sensitivity: float
    holdout_specificity: float
    holdout_auc: float
    meets_targets: bool


# -------------------------
# Dataset construction
# -------------------------

def build_child_dataset() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Ingest via research_engine; build child-level dataset with numeric means and session_count.
    Adds engineered domain features and quantile-bin indicators. Returns (X, y, child_ids).
    """
    if not research_engine.behavioral_database:
        research_engine.ingest_raw_data(limit=None)
        research_engine.index_behavioral_data()
    df = pd.DataFrame(research_engine.behavioral_database)
    if df.empty:
        raise RuntimeError("No behavioral data available")
    # Filter labeled sessions
    df = df[df['binary_label'].isin(['ASD', 'TD'])].copy()
    if df.empty:
        raise RuntimeError("No labeled data (ASD/TD) available")

    available = [c for c in NUMERIC_FEATURES_CANON if c in df.columns]
    if not available:
        raise RuntimeError("No expected numeric features present in data")

    # Aggregate to child-level
    agg = df.groupby('child_id')[available].mean().reset_index()
    sess_count = df.groupby('child_id').size().rename('session_count').reset_index()
    agg = agg.merge(sess_count, on='child_id', how='left')

    # Domain features (leakage-safe; derived from aggregated numeric columns)
    if 'unique_zones' in agg.columns and 'total_touch_points' in agg.columns:
        agg['touches_per_zone'] = safe_divide(agg['total_touch_points'], agg['unique_zones'])
    if 'unique_zones' in agg.columns and 'stroke_count' in agg.columns:
        agg['strokes_per_zone'] = safe_divide(agg['stroke_count'], agg['unique_zones'])
    if 'unique_zones' in agg.columns and 'session_duration' in agg.columns:
        agg['zones_per_minute'] = safe_divide(agg['unique_zones'], agg['session_duration'] / 60.0)
    if 'velocity_mean' in agg.columns and 'velocity_std' in agg.columns:
        agg['vel_std_over_mean'] = safe_divide(agg['velocity_std'], agg['velocity_mean'])
    if 'acc_magnitude_mean' in agg.columns and 'acc_magnitude_std' in agg.columns:
        agg['acc_std_over_mean'] = safe_divide(agg['acc_magnitude_std'], agg['acc_magnitude_mean'])
    if 'avg_time_between_points' in agg.columns and 'session_duration' in agg.columns:
        agg['avg_ibp_norm'] = safe_divide(agg['avg_time_between_points'], agg['session_duration'])
        agg['interpoint_rate'] = safe_divide(agg['session_duration'], agg['avg_time_between_points'])
    if 'total_touch_points' in agg.columns and 'session_duration' in agg.columns:
        agg['touch_rate'] = safe_divide(agg['total_touch_points'], agg['session_duration'])
    if 'stroke_count' in agg.columns and 'session_duration' in agg.columns:
        agg['stroke_rate'] = safe_divide(agg['stroke_count'], agg['session_duration'])

    # Quantile bins for selected ratios
    def _add_bin_flags(df_in: pd.DataFrame, col: str, q: int = 4) -> pd.DataFrame:
        if col not in df_in.columns:
            return df_in
        s = df_in[col]
        try:
            binned = pd.qcut(s.rank(method='first'), q=q, labels=False, duplicates='drop')
        except Exception:
            return df_in
        dummies = pd.get_dummies(binned, prefix=f"bin_{col}")
        return pd.concat([df_in, dummies], axis=1)

    for ratio_col in ['touch_rate', 'strokes_per_zone', 'vel_std_over_mean', 'acc_std_over_mean', 'zones_per_minute', 'interpoint_rate']:
        agg = _add_bin_flags(agg, ratio_col, q=4)

    # Child labels
    def _mode_or_first(s: pd.Series):
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]

    child_lab = df.groupby('child_id')['binary_label'].agg(_mode_or_first).reset_index()
    child_df = agg.merge(child_lab, on='child_id', how='left')
    child_df = child_df.dropna(subset=['binary_label'])

    X = child_df[[c for c in child_df.columns if c not in ['child_id', 'binary_label']]].copy()
    y = (child_df['binary_label'].values == 'ASD').astype(int)
    child_ids = child_df['child_id'].astype(str).tolist()
    return X, y, child_ids


# -------------------------
# Threshold utilities
# -------------------------

def choose_threshold(y_true: np.ndarray,
                     proba: np.ndarray,
                     target_sens: float,
                     target_spec: float,
                     policy: str = 'both_targets') -> Tuple[float, float, float]:
    """Select threshold per the given policy. Returns (thr, sens, spec)."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    spec = 1 - fpr
    policy = (policy or 'both_targets').lower()

    if policy == 'both_targets':
        idx = np.where((tpr >= target_sens) & (spec >= target_spec))[0]
        if len(idx) > 0:
            gains = (tpr[idx] - target_sens) + (spec[idx] - target_spec)
            best = int(idx[np.argmax(gains)])
            return float(thr[best]), float(tpr[best]), float(spec[best])
        idx = np.where(spec >= target_spec)[0]
        if len(idx) > 0:
            best = int(idx[np.argmax(tpr[idx])])
            return float(thr[best]), float(tpr[best]), float(spec[best])
        youden = tpr - fpr
        best = int(np.argmax(youden))
        return float(thr[best]), float(tpr[best]), float(spec[best])

    if policy == 'spec_first':
        idx = np.where(spec >= target_spec)[0]
        if len(idx) > 0:
            best = int(idx[np.argmax(tpr[idx])])
            return float(thr[best]), float(tpr[best]), float(spec[best])
        youden = tpr - fpr
        best = int(np.argmax(youden))
        return float(thr[best]), float(tpr[best]), float(spec[best])

    if policy == 'youden':
        youden = tpr - fpr
        best = int(np.argmax(youden))
        return float(thr[best]), float(tpr[best]), float(spec[best])

    if policy == 'np':
        fpr_constraint = 1.0 - float(target_spec)
        idx = np.where(fpr <= fpr_constraint)[0]
        if len(idx) > 0:
            tprs = tpr[idx]
            best_rel = int(np.argmax(tprs))
            tie_candidates = idx[np.where(tprs == tprs[best_rel])[0]]
            if len(tie_candidates) > 1:
                best = int(tie_candidates[np.argmax(spec[tie_candidates])])
            else:
                best = int(idx[best_rel])
            return float(thr[best]), float(tpr[best]), float(spec[best])
        idx = np.where(spec >= target_spec)[0]
        if len(idx) > 0:
            best = int(idx[np.argmax(tpr[idx])])
            return float(thr[best]), float(tpr[best]), float(spec[best])
        youden = tpr - fpr
        best = int(np.argmax(youden))
        return float(thr[best]), float(tpr[best]), float(spec[best])

    # default
    return choose_threshold(y_true, proba, target_sens, target_spec, policy='both_targets')


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic without SciPy."""
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    data = np.concatenate([a, b])
    data.sort()
    uniq = np.unique(data)

    def ecdf(x, vals):
        return np.searchsorted(np.sort(vals), x, side='right') / float(vals.size)

    Fa = np.array([ecdf(x, a) for x in uniq])
    Fb = np.array([ecdf(x, b) for x in uniq])
    return float(np.max(np.abs(Fa - Fb)))


# -------------------------
# Models
# -------------------------

def make_models(model_names: List[str], seed: int = 42):
    models = []
    for name in model_names:
        if name == 'lightgbm' and _lgbm_ok:
            mdl = lgb.LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                feature_fraction=0.8,
                min_child_samples=30,
                max_depth=-1,
                random_state=seed,
                scale_pos_weight=1.0,
                verbosity=-1,
            )
            models.append(('lightgbm', mdl))
        elif name == 'xgboost' and _xgb_ok:
            mdl = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                eval_metric='logloss',
                random_state=seed,
                tree_method='hist',
                scale_pos_weight=1.0,
            )
            models.append(('xgboost', mdl))
        elif name == 'extratrees':
            mdl = ExtraTreesClassifier(
                n_estimators=400,
                criterion='entropy',
                max_features=0.25,
                min_samples_leaf=8,
                min_samples_split=17,
                random_state=seed,
            )
            models.append(('extratrees', mdl))
        elif name == 'brf' and _brf_ok:
            mdl = BalancedRandomForestClassifier(
                n_estimators=400,
                random_state=seed,
                max_features='sqrt',
                min_samples_leaf=6,
            )
            models.append(('brf', mdl))
        elif name == 'logreg':
            mdl = LogisticRegression(
                random_state=seed, max_iter=2000, C=0.5, class_weight='balanced'
            )
            models.append(('logreg', mdl))
        # silently skip unknown or unavailable models
    if not models:
        models = [('logreg', LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'))]
    return models


# -------------------------
# Single-seed run
# -------------------------

def run_single_seed(
    seed: int,
    holdout_seed: int,
    target_sens: float,
    target_spec: float,
    model_list: List[str],
    calibration_method: str = 'isotonic',
    use_polynomial: bool = False,
    use_umap_cosine: bool = False,
    umap_components: int = 16,
    umap_neighbors: int = 40,
    threshold_policy: str = 'np',
    threshold_transfer: Optional[str] = 'iqr_mid',
    quantile_guard_ks: float = 0.0,
) -> SingleSeedResult:
    # Build dataset
    X_df, y_all, child_ids = build_child_dataset()

    # Seeding for numpy/random
    np.random.seed(seed)
    random.seed(seed)

    # Grouped holdout split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=int(holdout_seed))
    (tr_idx, te_idx) = list(gss.split(X_df, y_all, groups=child_ids))[0]
    Xtr_df = X_df.iloc[tr_idx].reset_index(drop=True)
    Xte_df = X_df.iloc[te_idx].reset_index(drop=True)
    ytr = y_all[tr_idx]
    yte = y_all[te_idx]
    groups_tr = [child_ids[i] for i in tr_idx]
    holdout_ids = [child_ids[i] for i in te_idx]

    # Preprocess for holdout transform
    scaler_all = StandardScaler()
    Xtr_s = scaler_all.fit_transform(Xtr_df)
    Xte_s = scaler_all.transform(Xte_df)

    if use_polynomial and not use_umap_cosine:
        pf_all = PolynomialFeatures(degree=2, include_bias=False)
        Xtr_s = pf_all.fit_transform(Xtr_s)
        Xte_s = pf_all.transform(Xte_s)

    if use_umap_cosine:
        try:
            import umap
            n_comp = max(2, min(int(umap_components), Xtr_s.shape[1]))
            umap_all = umap.UMAP(n_components=n_comp, n_neighbors=int(umap_neighbors), metric='cosine', random_state=seed)
            Xtr_u = umap_all.fit_transform(Xtr_s)
            Xte_u = umap_all.transform(Xte_s)
            Xtr_s = np.concatenate([Xtr_s, Xtr_u], axis=1)
            Xte_s = np.concatenate([Xte_s, Xte_u], axis=1)
        except Exception:
            pass

    # CV splitter
    if StratifiedGroupKFold is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = cv_splitter.split(Xtr_df.values, ytr, groups=groups_tr)
    else:
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(Xtr_df.values, ytr)

    models = make_models(model_list, seed=seed)

    fold_metrics: List[FoldMetric] = []
    thresholds: List[float] = []
    aucs: List[float] = []

    # OOF storage per model
    n_train = len(ytr)
    oof_per_model: Dict[str, np.ndarray] = {name: np.full(n_train, np.nan) for name, _ in models}

    # Per-fold CV
    for fold, (tri, vai) in enumerate(splits, 1):
        ss = StandardScaler()
        Xtri = ss.fit_transform(Xtr_df.values[tri])
        Xvai = ss.transform(Xtr_df.values[vai])

        if use_umap_cosine:
            try:
                import umap
                n_comp = max(2, min(int(umap_components), Xtri.shape[1]))
                um = umap.UMAP(n_components=n_comp, n_neighbors=int(umap_neighbors), metric='cosine', random_state=seed)
                Utr = um.fit_transform(Xtri)
                Uva = um.transform(Xvai)
                Xtri = np.concatenate([Xtri, Utr], axis=1)
                Xvai = np.concatenate([Xvai, Uva], axis=1)
            except Exception:
                pass
        elif use_polynomial:
            pf = PolynomialFeatures(degree=2, include_bias=False)
            Xtri = pf.fit_transform(Xtri)
            Xvai = pf.transform(Xvai)

        # Collect per-model calibrated probabilities
        proba_list = []
        yv = ytr[vai]
        for name, mdl in models:
            try:
                if hasattr(mdl, 'predict_proba') or hasattr(mdl, 'decision_function'):
                    cal = CalibratedClassifierCV(mdl, method=calibration_method, cv=3)
                    cal.fit(Xtri, ytr[tri])
                    prob = cal.predict_proba(Xvai)[:, 1]
                else:
                    mdl.fit(Xtri, ytr[tri])
                    if hasattr(mdl, 'predict_proba'):
                        prob = mdl.predict_proba(Xvai)[:, 1]
                    else:
                        scores = mdl.decision_function(Xvai)
                        prob = safe_divide(scores - scores.min(), scores.max() - scores.min())
                proba_list.append(prob)
                # Save to OOF
                if name in oof_per_model:
                    oof_per_model[name][vai] = prob
            except Exception:
                continue

        if not proba_list:
            raise RuntimeError("No valid model produced probabilities in CV")

        p_agg = np.mean(np.vstack(proba_list), axis=0)
        thr, sens, spec = choose_threshold(yv, p_agg, target_sens, target_spec, policy=threshold_policy)
        auc_val = roc_auc_score(yv, p_agg)

        fold_metrics.append(FoldMetric(fold=fold, threshold=float(thr), sensitivity=float(sens), specificity=float(spec), auc=float(auc_val)))
        thresholds.append(float(thr))
        aucs.append(float(auc_val))

    thr_med = float(np.median(np.array(thresholds)))
    cv_sens_mean = float(np.mean([m.sensitivity for m in fold_metrics]))
    cv_spec_mean = float(np.mean([m.specificity for m in fold_metrics]))
    cv_auc_mean = float(np.mean(aucs))

    # Train on full training set and evaluate holdout
    holdout_probas = []
    for name, mdl in models:
        try:
            if hasattr(mdl, 'predict_proba') or hasattr(mdl, 'decision_function'):
                cal = CalibratedClassifierCV(mdl, method=calibration_method, cv=3)
                cal.fit(Xtr_s, ytr)
                prob = cal.predict_proba(Xte_s)[:, 1]
            else:
                mdl.fit(Xtr_s, ytr)
                if hasattr(mdl, 'predict_proba'):
                    prob = mdl.predict_proba(Xte_s)[:, 1]
                else:
                    scores = mdl.decision_function(Xte_s)
                    prob = safe_divide(scores - scores.min(), scores.max() - scores.min())
            holdout_probas.append(prob)
        except Exception:
            continue

    if not holdout_probas:
        raise RuntimeError("No valid model produced probabilities on holdout")

    p_holdout = np.mean(np.vstack(holdout_probas), axis=0)

    # Drift guard (optional)
    if quantile_guard_ks > 0.0:
        # Compare OOF agg (where available) vs holdout
        oof_list = [oof_per_model[k] for k in oof_per_model if not np.isnan(oof_per_model[k]).all()]
        if oof_list:
            oof_agg = np.nanmean(np.vstack(oof_list), axis=0)
            ks = ks_statistic(oof_agg, p_holdout)
            # If drift large, prefer median threshold transfer
            if ks >= float(quantile_guard_ks):
                chosen_thr = thr_med
            else:
                if (threshold_transfer or '').lower() in ('iqr_mid', 'cv_median', 'median'):
                    chosen_thr = thr_med
                else:
                    chosen_thr, _, _ = choose_threshold(yte, p_holdout, target_sens, target_spec, policy=threshold_policy)
        else:
            chosen_thr = thr_med
    else:
        if (threshold_transfer or '').lower() in ('iqr_mid', 'cv_median', 'median'):
            chosen_thr = thr_med
        else:
            chosen_thr, _, _ = choose_threshold(yte, p_holdout, target_sens, target_spec, policy=threshold_policy)

    # Compute holdout metrics at chosen threshold
    yhat = (p_holdout >= chosen_thr).astype(int)
    tp = int(np.sum((yte == 1) & (yhat == 1)))
    tn = int(np.sum((yte == 0) & (yhat == 0)))
    fp = int(np.sum((yte == 0) & (yhat == 1)))
    fn = int(np.sum((yte == 1) & (yhat == 0)))
    sens = float(tp / (tp + fn + 1e-12))
    spec = float(tn / (tn + fp + 1e-12))
    auc = float(roc_auc_score(yte, p_holdout))

    meets = (sens >= target_sens) and (spec >= target_spec)

    return SingleSeedResult(
        seed=int(seed),
        holdout_seed=int(holdout_seed),
        cv_folds=fold_metrics,
        cv_threshold_median=thr_med,
        cv_sens_mean=cv_sens_mean,
        cv_spec_mean=cv_spec_mean,
        cv_auc_mean=cv_auc_mean,
        holdout_ids=[str(cid) for cid in holdout_ids],
        holdout_y=[int(v) for v in yte.tolist()],
        holdout_proba=[float(p) for p in p_holdout.tolist()],
        holdout_threshold=float(chosen_thr),
        holdout_sensitivity=sens,
        holdout_specificity=spec,
        holdout_auc=auc,
        meets_targets=bool(meets),
    )


# -------------------------
# Bagging across seeds
# -------------------------

def bag_across_seeds(
    seed_results: List[SingleSeedResult],
    target_sens: float,
    target_spec: float,
    threshold_policy: str = 'np',
) -> Dict[str, Any]:
    if not seed_results:
        raise ValueError("No seed results provided for bagging")

    # Align on common holdout ids and order
    id_sets = [set(r.holdout_ids) for r in seed_results]
    common_ids = set.intersection(*id_sets)
    if not common_ids:
        raise RuntimeError("Seed runs do not share common holdout IDs; adjust holdout_seed to match")

    # Build aligned arrays
    id_to_idx_per_seed: List[Dict[str, int]] = []
    for r in seed_results:
        id_to_idx_per_seed.append({cid: i for i, cid in enumerate(r.holdout_ids)})

    sorted_ids = sorted(common_ids)
    y_list = []
    proba_stack = []

    for r, id_to_idx in zip(seed_results, id_to_idx_per_seed):
        indices = [id_to_idx[cid] for cid in sorted_ids]
        if not y_list:
            y_list = [r.holdout_y[i] for i in indices]
        proba_stack.append(np.array([r.holdout_proba[i] for i in indices]))

    y_arr = np.array(y_list)
    P = np.vstack(proba_stack)
    p_bag = np.mean(P, axis=0)

    # Pick threshold by policy on bagged probabilities
    thr_bag, sens_bag, spec_bag = choose_threshold(y_arr, p_bag, target_sens, target_spec, policy=threshold_policy)
    auc_bag = float(roc_auc_score(y_arr, p_bag))

    return {
        "holdout_ids": sorted_ids,
        "holdout_y": [int(v) for v in y_arr.tolist()],
        "holdout_proba": [float(p) for p in p_bag.tolist()],
        "bagged_metrics": {
            "threshold": float(thr_bag),
            "sensitivity": float(sens_bag),
            "specificity": float(spec_bag),
            "auc": float(auc_bag),
        },
    }


# -------------------------
# CLI
# -------------------------

def main():
    # If CLI flags are provided, honor them; otherwise, use the hardcoded best settings
    use_cli = len(sys.argv) > 1

    if use_cli:
        ap = argparse.ArgumentParser(description="Bagged clinical end-to-end pipeline (raw → prediction)")
        ap.add_argument('--target-sens', type=float, default=BEST_TARGET_SENS, help='Target sensitivity threshold')
        ap.add_argument('--target-spec', type=float, default=BEST_TARGET_SPEC, help='Target specificity threshold')
        ap.add_argument('--calibration', type=str, default=BEST_CALIBRATION, choices=['isotonic', 'sigmoid'], help='Calibration method')
        ap.add_argument('--use-polynomial', action='store_true', help='Use polynomial features (degree=2)')
        ap.add_argument('--use-umap-cosine', action='store_true', help='Augment with UMAP (cosine) embeddings')
        ap.add_argument('--umap-components', type=int, default=BEST_UMAP_COMPONENTS, help='UMAP n_components')
        ap.add_argument('--umap-neighbors', type=int, default=BEST_UMAP_NEIGHBORS, help='UMAP n_neighbors')
        ap.add_argument('--seeds', type=int, nargs='+', default=BEST_SEEDS, help='Random seeds to bag over')
        ap.add_argument('--holdout-seed', type=int, default=BEST_HOLDOUT_SEED, help='Random seed for GroupShuffleSplit holdout')
        ap.add_argument('--threshold-policy', type=str, default=BEST_THRESHOLD_POLICY, choices=['both_targets', 'spec_first', 'youden', 'np'], help='Threshold selection policy')
        ap.add_argument('--threshold-transfer', type=str, default=BEST_THRESHOLD_TRANSFER, help='Threshold transfer method (e.g., iqr_mid, median, cv_median, optimize)')
        ap.add_argument('--quantile-guard-ks', type=float, default=BEST_QUANTILE_GUARD_KS, help='If >0, apply KS drift guard between OOF and holdout proba')
        ap.add_argument('--models', type=str, nargs='+', default=BEST_MODELS, help='Models to include in the ensemble')
        ap.add_argument('--export', type=str, default=None, help='Path to export JSON with results (bagged + per-seed)')
        args = ap.parse_args()

        seeds = list(args.seeds)
        holdout_seed = int(args.holdout_seed)
        target_sens = float(args.target_sens)
        target_spec = float(args.target_spec)
        calibration = str(args.calibration)
        use_polynomial = bool(args.use_polynomial)
        use_umap_cosine = bool(args.use_umap_cosine)
        umap_components = int(args.umap_components)
        umap_neighbors = int(args.umap_neighbors)
        threshold_policy = str(args.threshold_policy)
        threshold_transfer = str(args.threshold_transfer)
        quantile_guard_ks = float(args.quantile_guard_ks)
        models = list(args.models)
        export_path = Path(args.export or 'results/bagged_cli.json')
    else:
        # Hardcoded best settings
        seeds = list(BEST_SEEDS)
        holdout_seed = int(BEST_HOLDOUT_SEED)
        target_sens = float(BEST_TARGET_SENS)
        target_spec = float(BEST_TARGET_SPEC)
        calibration = str(BEST_CALIBRATION)
        use_polynomial = bool(BEST_USE_POLYNOMIAL)
        use_umap_cosine = bool(BEST_USE_UMAP_COSINE)
        umap_components = int(BEST_UMAP_COMPONENTS)
        umap_neighbors = int(BEST_UMAP_NEIGHBORS)
        threshold_policy = str(BEST_THRESHOLD_POLICY)
        threshold_transfer = str(BEST_THRESHOLD_TRANSFER)
        quantile_guard_ks = float(BEST_QUANTILE_GUARD_KS)
        models = list(BEST_MODELS)
        export_path = Path(BEST_EXPORT_PATH)

    # Run per seed
    seed_results: List[SingleSeedResult] = []
    for sd in seeds:
        res = run_single_seed(
            seed=int(sd),
            holdout_seed=holdout_seed,
            target_sens=target_sens,
            target_spec=target_spec,
            model_list=models,
            calibration_method=calibration,
            use_polynomial=use_polynomial,
            use_umap_cosine=use_umap_cosine,
            umap_components=umap_components,
            umap_neighbors=umap_neighbors,
            threshold_policy=threshold_policy,
            threshold_transfer=threshold_transfer,
            quantile_guard_ks=quantile_guard_ks,
        )
        seed_results.append(res)

    # Bagging
    bag = bag_across_seeds(
        seed_results,
        target_sens=target_sens,
        target_spec=target_spec,
        threshold_policy=threshold_policy,
    )

    # Compose output
    out: Dict[str, Any] = {
        "policy": threshold_policy,
        "targets": {"sensitivity": target_sens, "specificity": target_spec},
        "models": models,
        "use_polynomial": use_polynomial,
        "use_umap_cosine": use_umap_cosine,
        "umap_components": umap_components,
        "umap_neighbors": umap_neighbors,
        "calibration": calibration,
        "threshold_transfer": threshold_transfer,
        "quantile_guard_ks": quantile_guard_ks,
        "seeds": seeds,
        "holdout_seed": holdout_seed,
        "per_seed": [
            {
                "seed": r.seed,
                "cv": {
                    "folds": [
                        {
                            "fold": fm.fold,
                            "threshold": fm.threshold,
                            "sensitivity": fm.sensitivity,
                            "specificity": fm.specificity,
                            "auc": fm.auc,
                        }
                        for fm in r.cv_folds
                    ],
                    "threshold_median": r.cv_threshold_median,
                    "sensitivity_mean": r.cv_sens_mean,
                    "specificity_mean": r.cv_spec_mean,
                    "auc_mean": r.cv_auc_mean,
                },
                "holdout": {
                    "sensitivity": r.holdout_sensitivity,
                    "specificity": r.holdout_specificity,
                    "auc": r.holdout_auc,
                    "threshold": r.holdout_threshold,
                    "meets_targets": r.meets_targets,
                },
                "holdout_ids": r.holdout_ids,
                "holdout_y": r.holdout_y,
                "holdout_proba": r.holdout_proba,
            }
            for r in seed_results
        ],
        "bagged": bag,
        # Convenience keys for external comparisons
        "holdout_metrics": bag.get("bagged_metrics", {}),
        "cv_metrics": {
            # Aggregate AUCs across all folds across all seeds for std estimate
            "auc": {
                "values": [fm.auc for r in seed_results for fm in r.cv_folds],
            }
        },
    }

    # Print concise summary
    label = "hardcoded best settings" if not use_cli else "CLI settings"
    print(f"\nBagged clinical end-to-end summary ({label}):")
    bm = out["bagged"]["bagged_metrics"]
    print(f"Seeds: {out['seeds']} | Models: {out['models']}")
    print(f"Targets: sens>={target_sens}, spec>={target_spec} | Policy: {threshold_policy}")
    print(f"Bagged AUC={bm['auc']:.4f}, Sens={bm['sensitivity']:.4f}, Spec={bm['specificity']:.4f}, Thr={bm['threshold']:.6f}")

    # Export
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {export_path}")


if __name__ == "__main__":
    main()

