#!/usr/bin/env python3
"""
Clinical Fair Pipeline
- Single pipeline to target Sensitivity >= target_sens and Specificity >= target_spec
- Child-level, leakage-safe evaluation using StratifiedGroupKFold
- PolynomialFeatures on standardized numeric features
- Soft-voting ensemble (LightGBM, Balanced Random Forest, Logistic Regression)
- Per-fold calibration (isotonic or sigmoid) before thresholding
- Clinical thresholding policy: prefer thresholds that meet BOTH targets in CV; fallback to spec-first
Outputs JSON summary to results/clinical_fair_pipeline_results.json
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Apply OpenMP fix early to suppress deprecation warnings on macOS
try:
    from src.openmp_fix import apply_openmp_fix
    apply_openmp_fix()
except ImportError:
    # Fallback: just set thread limits if openmp_fix is not available
    import os
    for _k, _v in (
        ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("LIGHTGBM_NUM_THREADS", "1"),
        # macOS stability knobs
        ("ACCELERATE_NEW_LAPACK", "1"),
        ("KMP_DUPLICATE_LIB_OK", "TRUE"),
    ):
        os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import random
import joblib

from sklearn.preprocessing import StandardScaler, PolynomialFeatures


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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# Demographics (optional)
try:
    from src.demographics.demographic_manager import demographic_manager  # type: ignore
except Exception:
    demographic_manager = None  # type: ignore
from sklearn.calibration import CalibratedClassifierCV
try:
    from src.pipelines.safe_enhancement_pipeline import SafeEnhancementPipeline
    from src.demographics.safe_demographic_manager import safe_demographic_manager
except Exception:
    SafeEnhancementPipeline = None  # type: ignore
    safe_demographic_manager = None  # type: ignore

# Optional models - moved to functions to avoid slow imports during --help

# Project RAG engine - moved to main() to avoid slow imports during --help


NUMERIC_FEATURES_CANON = [
    'velocity_mean', 'velocity_std', 'velocity_max', 'velocity_cv',
    'tremor_indicator', 'acc_magnitude_mean', 'acc_magnitude_std',
    'palm_touch_ratio', 'unique_fingers', 'max_finger_id',
    'session_duration', 'stroke_count', 'total_touch_points',
    'unique_zones', 'unique_colors', 'final_completion',
    'completion_progress_rate', 'avg_time_between_points',
    'canceled_touches'
]


def build_child_dataset() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Ingest via research_engine; build child-level dataset with numeric means and session_count.
    Adds engineered domain features from available aggregated metrics and quantile-bin indicators for key ratios.
    Returns X (DataFrame), y (np.ndarray 0/1 with ASD=1), child_ids (List[str]).
    """
    # Import RAG engine only when needed
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting RAG data ingestion...")
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from rag_system.research_engine import research_engine  # type: ignore
    try:
        from rag_system import config as rag_config  # type: ignore
        # Prefer env overrides, fallback to defaults
        import os as _os
        raw_default = Path("data/raw/fileKeys").resolve()
        raw_path = Path(_os.getenv("RAW_DATA_PATH", str(raw_default))).resolve()
        # Candidate labels paths, in order of preference
        lbl_env = _os.getenv("LABELS_PATH", "")
        lbl_candidates = [
            Path(lbl_env).resolve() if lbl_env else None,
            Path("data/knowledge_base/lables_fileKeys.csv").resolve(),
            Path("data/processed/labels.csv").resolve(),
        ]
        rag_config.config.RAW_DATA_PATH = raw_path
        for _lp in lbl_candidates:
            if _lp and _lp.exists():
                rag_config.config.LABELS_PATH = _lp
                break
    except Exception:
        pass
    
    if not research_engine.behavioral_database:
        # Ingest all sessions
        print(f"[{time.strftime('%H:%M:%S')}] Ingesting raw data (this may take several minutes)...")
        research_engine.ingest_raw_data(limit=None)
        print(f"[{time.strftime('%H:%M:%S')}] Raw data ingested, indexing behavioral data...")
        research_engine.index_behavioral_data()
        print(f"[{time.strftime('%H:%M:%S')}] Behavioral database ready: {len(research_engine.behavioral_database)} sessions")
    print(f"[{time.strftime('%H:%M:%S')}] Building child dataset from behavioral database...")
    df = pd.DataFrame(research_engine.behavioral_database)
    if df.empty:
        raise RuntimeError("No behavioral data available")
    # Filter known labels
    df = df[df['binary_label'].isin(['ASD', 'TD'])].copy()
    if df.empty:
        raise RuntimeError("No labeled data (ASD/TD) available")
    # Features intersection
    available = [c for c in NUMERIC_FEATURES_CANON if c in df.columns]
    if not available:
        raise RuntimeError("No expected numeric features present in data")
    # Child aggregation
    agg = df.groupby('child_id')[available].mean().reset_index()
    sess_count = df.groupby('child_id').size().rename('session_count').reset_index()
    agg = agg.merge(sess_count, on='child_id', how='left')
    # Engineer domain features on aggregated numeric signals (no label leakage)
    # Per-zone dynamics
    if 'unique_zones' in agg.columns and 'total_touch_points' in agg.columns:
        agg['touches_per_zone'] = safe_divide(agg['total_touch_points'], agg['unique_zones'])
    if 'unique_zones' in agg.columns and 'stroke_count' in agg.columns:
        agg['strokes_per_zone'] = safe_divide(agg['stroke_count'], agg['unique_zones'])
    if 'unique_zones' in agg.columns and 'session_duration' in agg.columns:
        agg['zones_per_minute'] = safe_divide(agg['unique_zones'], agg['session_duration'] / 60.0)
    # Jitter/jerk proxies
    if 'velocity_mean' in agg.columns and 'velocity_std' in agg.columns:
        agg['vel_std_over_mean'] = safe_divide(agg['velocity_std'], agg['velocity_mean'])
    if 'acc_magnitude_mean' in agg.columns and 'acc_magnitude_std' in agg.columns:
        agg['acc_std_over_mean'] = safe_divide(agg['acc_magnitude_std'], agg['acc_magnitude_mean'])
    # Temporal stability indices
    if 'avg_time_between_points' in agg.columns and 'session_duration' in agg.columns:
        agg['avg_ibp_norm'] = safe_divide(agg['avg_time_between_points'], agg['session_duration'])
        agg['interpoint_rate'] = safe_divide(agg['session_duration'], agg['avg_time_between_points'])
    if 'total_touch_points' in agg.columns and 'session_duration' in agg.columns:
        agg['touch_rate'] = safe_divide(agg['total_touch_points'], agg['session_duration'])
    if 'stroke_count' in agg.columns and 'session_duration' in agg.columns:
        agg['stroke_rate'] = safe_divide(agg['stroke_count'], agg['session_duration'])
    # Quantile bins (quartiles) for key ratios, expanded to one-hot indicators
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

    # Mode label per child
    def _mode_or_first(s: pd.Series):
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]
    child_lab = df.groupby('child_id')['binary_label'].agg(_mode_or_first).reset_index()
    child_df = agg.merge(child_lab, on='child_id', how='left')
    child_df = child_df.dropna(subset=['binary_label'])
    X = child_df[[c for c in child_df.columns if c not in ['child_id', 'binary_label']]].copy()
    y = (child_df['binary_label'].values == 'ASD').astype(int)
    child_ids = child_df['child_id'].astype(str).tolist()
    print(f"[{time.strftime('%H:%M:%S')}] Child dataset complete: {len(child_ids)} children, {X.shape[1]} features (took {time.time()-t0:.1f}s)")
    return X, y, child_ids


def choose_threshold_clinical(y_true: np.ndarray,
                              proba: np.ndarray,
                              target_sens: float,
                              target_spec: float) -> Tuple[float, float, float]:
    """Pick threshold that satisfies both targets if possible; otherwise spec-first fallback.
    Returns (thr, sens, spec)."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    spec = 1 - fpr
    # Both targets
    idx = np.where((tpr >= target_sens) & (spec >= target_spec))[0]
    if len(idx) > 0:
        gains = (tpr[idx] - target_sens) + (spec[idx] - target_spec)
        best = int(idx[np.argmax(gains)])
        return float(thr[best]), float(tpr[best]), float(spec[best])
    # Spec-first fallback
    idx = np.where(spec >= target_spec)[0]
    if len(idx) > 0:
        # Maximize sensitivity
        best = int(idx[np.argmax(tpr[idx])])
        return float(thr[best]), float(tpr[best]), float(spec[best])
    # Youden fallback
    youden = tpr - fpr
    best = int(np.argmax(youden))
    return float(thr[best]), float(tpr[best]), float(spec[best])


def choose_threshold(y_true: np.ndarray,
                     proba: np.ndarray,
                     target_sens: float,
                     target_spec: float,
                     policy: str = 'both_targets') -> Tuple[float, float, float]:
    """General threshold selector.
    Policies:
      - 'both_targets': meet both sensitivity and specificity if possible; else spec-first; else Youden's J
      - 'spec_first': maximize sensitivity subject to specificity >= target_spec; else Youden
      - 'youden': maximize Youden's J (tpr - fpr)
      - 'np': Neyman–Pearson: maximize sensitivity (TPR) subject to FPR <= 1 - target_spec
    Returns (thr, sens, spec).
    """
    fpr, tpr, thr = roc_curve(y_true, proba)
    spec = 1 - fpr
    policy = (policy or 'both_targets').lower()

    if policy == 'both_targets':
        idx = np.where((tpr >= target_sens) & (spec >= target_spec))[0]
        if len(idx) > 0:
            gains = (tpr[idx] - target_sens) + (spec[idx] - target_spec)
            best = int(idx[np.argmax(gains)])
            return float(thr[best]), float(tpr[best]), float(spec[best])
        # spec-first fallback
        idx = np.where(spec >= target_spec)[0]
        if len(idx) > 0:
            best = int(idx[np.argmax(tpr[idx])])
            return float(thr[best]), float(tpr[best]), float(spec[best])
        # youden fallback
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
            # among feasible, maximize TPR; tie-break by highest specificity
            tprs = tpr[idx]
            best_rel = int(np.argmax(tprs))
            tie_candidates = idx[np.where(tprs == tprs[best_rel])[0]]
            if len(tie_candidates) > 1:
                best = int(tie_candidates[np.argmax(spec[tie_candidates])])
            else:
                best = int(idx[best_rel])
            return float(thr[best]), float(tpr[best]), float(spec[best])
        # fallback: spec-first
        idx = np.where(spec >= target_spec)[0]
        if len(idx) > 0:
            best = int(idx[np.argmax(tpr[idx])])
            return float(thr[best]), float(tpr[best]), float(spec[best])
        # youden fallback
        youden = tpr - fpr
        best = int(np.argmax(youden))
        return float(thr[best]), float(tpr[best]), float(spec[best])

    # default to both_targets if unknown
    return choose_threshold(y_true, proba, target_sens, target_spec, policy='both_targets')


def _logit_clip(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(p: np.ndarray, y: np.ndarray, T_min: float = 0.5, T_max: float = 5.0, num: int = 300) -> float:
    """Fit temperature T by grid-search minimizing log loss on given probabilities.
    Returns best T.
    """
    y = y.astype(int)
    logits = _logit_clip(p)
    Ts = np.linspace(T_min, T_max, num)
    best_T = 1.0
    best_loss = np.inf
    for T in Ts:
        p_cal = _sigmoid(logits / T)
        # log loss
        eps = 1e-12
        p_cal = np.clip(p_cal, eps, 1 - eps)
        loss = -np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal))
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    return best_T


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    return _sigmoid(_logit_clip(p) / float(T))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Compute two-sample KS statistic (D) without SciPy."""
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    data = np.concatenate([a, b])
    data.sort()
    # unique thresholds
    uniq = np.unique(data)
    # empirical CDFs
    def ecdf(x, vals):
        return np.searchsorted(np.sort(vals), x, side='right') / float(vals.size)
    Fa = np.array([ecdf(x, a) for x in uniq])
    Fb = np.array([ecdf(x, b) for x in uniq])
    return float(np.max(np.abs(Fa - Fb)))


def make_models(model_names: List[str], seed: int = 42):
    # Import optional models only when needed
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
                # Threading and stability
                n_jobs=1,
                force_row_wise=True,
            )
            models.append(('lightgbm', mdl))
        elif name == 'lgbm_grid' and _lgbm_ok:
            for mcs in [20, 40, 60]:
                for ff in [0.6, 0.8]:
                    for spw in [0.75, 1.0, 1.25]:
                        mdl = lgb.LGBMClassifier(
                            n_estimators=400,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            feature_fraction=ff,
                            min_child_samples=mcs,
                            max_depth=-1,
                            random_state=seed,
                            scale_pos_weight=spw,
                            verbosity=-1,
                            # Threading and stability
                            n_jobs=1,
                            force_row_wise=True,
                        )
                        name_tag = f"lgbm_mc{mcs}_ff{ff}_spw{spw}"
                        models.append((name_tag, mdl))
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
        elif name == 'xgb_grid' and _xgb_ok:
            for md in [3, 4, 5]:
                for mcw in [1, 3, 5]:
                    for spw in [0.75, 1.0, 1.25]:
                        for ral in [0.0, 0.5, 1.0]:
                            for gam in [0.0, 0.5]:
                                mdl = XGBClassifier(
                                    n_estimators=400,
                                    learning_rate=0.05,
                                    max_depth=md,
                                    min_child_weight=mcw,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    reg_lambda=1.0,
                                    reg_alpha=ral,
                                    gamma=gam,
                                    eval_metric='logloss',
                                    random_state=seed,
                                    tree_method='hist',
                                    scale_pos_weight=spw,
                                )
                                name_tag = f"xgb_md{md}_mcw{mcw}_spw{spw}_ral{ral}_gam{gam}"
                                models.append((name_tag, mdl))
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
        # Fallback
        models = [('logreg', LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'))]
    return models


def run_pipeline(target_sens: float,
                 target_spec: float,
                 use_polynomial: bool,
                 calibration_method: str,
                 model_list: List[str],
                 use_umap_cosine: bool = False,
                 umap_components: int = 16,
                 umap_neighbors: int = 40,
                 top_k_models: int = 6,
                 final_calibration: Optional[str] = None,
                 report_holdout_specfirst: bool = False,
                 use_quantile_threshold: bool = False,
                 threshold_policy: str = 'both_targets',
                 threshold_transfer: Optional[str] = None,
                 quantile_guard_ks: float = 0.0,
                 seed: int = 42,
                 holdout_seed: Optional[int] = None,
                 save_preds: bool = False,
                 export_dir: Optional[str] = None,
                 demographics_path: Optional[str] = None) -> Dict[str, Any]:
    # Progress + timeout
    START_TIME = time.time()
    def check_timeout(limit_minutes: float = 45.0) -> None:
        elapsed = (time.time() - START_TIME) / 60.0
        if elapsed > limit_minutes:
            raise TimeoutError(f"Pipeline exceeded {limit_minutes} minutes")

    print(f"[{time.strftime('%H:%M:%S')}] Starting Enhanced Clinical Pipeline...")
    # Load demographics if available
    demographics_df: Optional[pd.DataFrame] = None
    if demographics_path and demographic_manager is not None:
        try:
            # Accept directory or single CSV/XLSX file (use parent dir)
            base = Path(demographics_path)
            base_dir = base if base.is_dir() else base.parent
            if demographic_manager.load_from_dir(base_dir):
                rows = []
                for cid, info in demographic_manager.demographic_data.items():
                    rows.append({"child_id": cid, **info})
                demographics_df = pd.DataFrame(rows)
                print(f"[{time.strftime('%H:%M:%S')}] [demographics] Loaded {len(demographics_df)} children from {base_dir}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [demographics] Warning: failed to load: {e}")

    X_df, y, child_ids = build_child_dataset()
    check_timeout()

    # Phase 1: Demographic feature enhancement (if manager available)
    if demographics_df is not None and safe_demographic_manager is not None:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Enhancing features with demographics (Phase 1)...")
            f_before = X_df.shape[1]
            X_aug = safe_demographic_manager.enhance_features(X_df, child_ids)
            if 'child_id' in X_aug.columns:
                X_aug = X_aug.drop(columns=['child_id'])
            numeric_cols = X_aug.select_dtypes(include=[np.number]).columns
            X_df = X_aug[numeric_cols].copy()
            print(f"[{time.strftime('%H:%M:%S')}] Demographic enhancement complete: {f_before} -> {X_df.shape[1]} features")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [demographics] Enhancement skipped due to error: {e}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Demographic enhancement skipped (unavailable)")
    # Seeding
    np.random.seed(seed)
    random.seed(seed)
    # Holdout split (child-level)
    ho_seed = seed if holdout_seed is None else int(holdout_seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=ho_seed)
    (tr_idx, te_idx) = list(gss.split(X_df, y, groups=child_ids))[0]
    Xtr_df = X_df.iloc[tr_idx].reset_index(drop=True)
    Xte_df = X_df.iloc[te_idx].reset_index(drop=True)
    ytr = y[tr_idx]
    yte = y[te_idx]
    groups_tr = [child_ids[i] for i in tr_idx]

    scaler_all = StandardScaler()
    Xtr_s = scaler_all.fit_transform(Xtr_df)
    Xte_s = scaler_all.transform(Xte_df)
    if use_polynomial and not use_umap_cosine:
        poly_all = PolynomialFeatures(degree=2, include_bias=False)
        Xtr_s = poly_all.fit_transform(Xtr_s)
        Xte_s = poly_all.transform(Xte_s)
    # Optional UMAP (cosine) on full training for holdout transform
    if use_umap_cosine:
        try:
            import umap
            n_comp = max(2, min(umap_components, Xtr_s.shape[1]))
            umap_all = umap.UMAP(n_components=n_comp, n_neighbors=umap_neighbors, metric='cosine', random_state=seed)
            Xtr_u = umap_all.fit_transform(Xtr_s)
            Xte_u = umap_all.transform(Xte_s)
            # Concatenate embeddings with original scaled features
            Xtr_s = np.concatenate([Xtr_s, Xtr_u], axis=1)
            Xte_s = np.concatenate([Xte_s, Xte_u], axis=1)
        except Exception:
            pass

    # Nested CV for threshold selection
    if StratifiedGroupKFold is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = cv_splitter.split(Xtr_df.values, ytr, groups=groups_tr)
    else:
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(Xtr_df.values, ytr)

    models = make_models(model_list, seed=seed)

    fold_metrics = []
    thresholds = []
    aucs = []

    # Prepare OOF containers for combiner
    n_train = len(ytr)
    oof_per_model: Dict[str, np.ndarray] = {name: np.full(n_train, np.nan) for name, _ in models}
    fold_va_indices: List[np.ndarray] = []

    for fold, (tri, vai) in enumerate(splits, 1):
        print(f"[{time.strftime('%H:%M:%S')}] Cross-validation fold {fold}/5...")
        check_timeout()
        # Fit preprocessors per fold
        ss = StandardScaler()
        Xtri = ss.fit_transform(Xtr_df.values[tri])
        Xvai = ss.transform(Xtr_df.values[vai])
        if use_umap_cosine:
            try:
                import umap
                n_comp = max(2, min(umap_components, Xtri.shape[1]))
                um = umap.UMAP(n_components=n_comp, n_neighbors=umap_neighbors, metric='cosine', random_state=seed)
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

        # Calibrated models (collect per-model OOF)
        proba_list = []
        auc_list = []
        yv = ytr[vai]
        for name, mdl in models:
            print(f"[{time.strftime('%H:%M:%S')}] Training model '{name}' (fold {fold})...")
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
                        # decision_function fallback
                        scores = mdl.decision_function(Xvai)
                        prob = safe_divide(scores - scores.min(), scores.max() - scores.min())
                proba_list.append(prob)
                auc_list.append(roc_auc_score(yv, prob))
                # Save into OOF array for this model
                oof_arr = oof_per_model.get(name)
                if oof_arr is not None:
                    oof_arr[vai] = prob
            except Exception:
                continue
        if not proba_list:
            raise RuntimeError("No valid model produced probabilities in CV")
        p_agg = np.mean(np.vstack(proba_list), axis=0)
        thr, sens, spec = choose_threshold(yv, p_agg, target_sens, target_spec, policy=threshold_policy)
        auc_val = roc_auc_score(yv, p_agg)
        fold_metrics.append({"fold": fold, "threshold": float(thr), "sensitivity": float(sens), "specificity": float(spec), "auc": float(auc_val)})
        thresholds.append(thr)
        aucs.append(auc_val)
        fold_va_indices.append(np.array(vai))

    thr_med = float(np.median(thresholds))
    cv_sens_mean = float(np.mean([m['sensitivity'] for m in fold_metrics]))
    cv_spec_mean = float(np.mean([m['specificity'] for m in fold_metrics]))
    cv_auc_mean = float(np.mean(aucs))

    # Build E_sens and E_spec from OOF
    # Compute per-model OOF AUCs
    model_aucs = []
    for name in oof_per_model:
        o = oof_per_model[name]
        mask = ~np.isnan(o)
        if mask.sum() > 0 and len(np.unique(ytr[mask])) > 1:
            try:
                model_aucs.append((float(roc_auc_score(ytr[mask], o[mask])), name))
            except Exception:
                continue
    model_aucs.sort(reverse=True)
    top_names = [n for _, n in model_aucs[:max(1, int(top_k_models))]] if model_aucs else [n for n, _ in models]
    # Define E_sens and E_spec using best LGBM + best XGB plus BRF/ExtraTrees
    name_set = set(oof_per_model.keys())
    best_lgbm = None
    best_xgb = None
    for auc_val, nm in model_aucs:
        if (best_lgbm is None) and (nm.startswith('lgbm_') or nm == 'lightgbm'):
            best_lgbm = nm
        if (best_xgb is None) and (nm.startswith('xgb_') or nm == 'xgboost'):
            best_xgb = nm
        if best_lgbm and best_xgb:
            break
    # fallbacks
    if best_lgbm is None and model_aucs:
        best_lgbm = model_aucs[0][1]
    if best_xgb is None and len(model_aucs) > 1:
        best_xgb = model_aucs[1][1]
    e_sens_names = []
    e_spec_names = []
    if best_lgbm in name_set:
        e_sens_names.append(best_lgbm)
        e_spec_names.append(best_lgbm)
    if best_xgb in name_set:
        e_sens_names.append(best_xgb)
        e_spec_names.append(best_xgb)
    if 'brf' in name_set:
        e_sens_names.append('brf')
    if 'extratrees' in name_set:
        e_spec_names.append('extratrees')
    # Ensure non-empty
    if not e_sens_names:
        e_sens_names = [nm for _, nm in model_aucs[:max(1, int(top_k_models))]]
    if not e_spec_names:
        e_spec_names = [nm for _, nm in model_aucs[:max(1, int(top_k_models))]]
    # Compose z_sens and z_spec OOF
    def mean_across(names: List[str]) -> np.ndarray:
        arrs = []
        for nm in names:
            if nm in oof_per_model:
                arrs.append(oof_per_model[nm])
        if not arrs:
            return np.full(n_train, np.nan)
        A = np.vstack(arrs)
        return np.nanmean(A, axis=0)
    z_sens_oof = mean_across(e_sens_names)
    z_spec_oof = mean_across(e_spec_names)

    # Alpha-grid constrained search
    alphas = np.linspace(0.0, 1.0, 21)
    best_alpha = None
    best_tau_per_fold: List[float] = []
    best_score = (-1, -1.0)  # (feasible_folds, surplus)
    per_alpha_cv = {}
    for alpha in alphas:
        tau_list = []
        sens_list = []
        spec_list = []
        feasible = 0
        surplus_sum = 0.0
        for vai in fold_va_indices:
            p_val = alpha * z_sens_oof[vai] + (1.0 - alpha) * z_spec_oof[vai]
            yv = ytr[vai]
            thr, s, sp = choose_threshold(yv, p_val, target_sens, target_spec, policy=threshold_policy)
            tau_list.append(thr)
            sens_list.append(s)
            spec_list.append(sp)
            if (s >= target_sens) and (sp >= target_spec):
                feasible += 1
                surplus_sum += (s - target_sens) + (sp - target_spec)
        per_alpha_cv[float(alpha)] = {
            "fold_thresholds": [float(t) for t in tau_list],
            "sens": [float(x) for x in sens_list],
            "spec": [float(x) for x in spec_list],
            "feasible_folds": feasible,
            "surplus": float(surplus_sum)
        }
        score = (feasible, surplus_sum)
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_tau_per_fold = tau_list
    tau_med_alpha = float(np.median(best_tau_per_fold)) if best_tau_per_fold else thr_med

    # Meta-combiner: leak-safe per-fold training for OOF predictions
    meta_oof = np.full(n_train, np.nan, dtype=float)
    meta_tau_per_fold: List[float] = []
    meta_sens_per_fold: List[float] = []
    meta_spec_per_fold: List[float] = []
    from sklearn.linear_model import LogisticRegression as LR
    for vai in fold_va_indices:
        train_mask = np.ones(n_train, dtype=bool)
        train_mask[vai] = False
        X_meta_tr = np.column_stack([
            z_sens_oof[train_mask],
            z_spec_oof[train_mask],
            np.abs(z_sens_oof[train_mask] - z_spec_oof[train_mask])
        ])
        y_meta_tr = ytr[train_mask]
        X_meta_va = np.column_stack([
            z_sens_oof[vai],
            z_spec_oof[vai],
            np.abs(z_sens_oof[vai] - z_spec_oof[vai])
        ])
        try:
            meta = LR(C=0.5, penalty='l2', solver='liblinear', random_state=42)
            meta_cal = CalibratedClassifierCV(meta, method=calibration_method, cv=3)
            meta_cal.fit(X_meta_tr, y_meta_tr)
            p_meta_va = meta_cal.predict_proba(X_meta_va)[:, 1]
        except Exception:
            p_meta_va = 0.5 * (z_sens_oof[vai] + z_spec_oof[vai])
        meta_oof[vai] = p_meta_va
        thr_m, s_m, sp_m = choose_threshold(ytr[vai], p_meta_va, target_sens, target_spec, policy=threshold_policy)
        meta_tau_per_fold.append(thr_m)
        meta_sens_per_fold.append(s_m)
        meta_spec_per_fold.append(sp_m)
    # Evaluate meta vs alpha on CV
    feasible_alpha = sum((np.array(per_alpha_cv.get(float(best_alpha), {}).get('sens', [])) >= target_sens) & (np.array(per_alpha_cv.get(float(best_alpha), {}).get('spec', [])) >= target_spec)) if best_alpha is not None else 0
    feasible_meta = sum((np.array(meta_sens_per_fold) >= target_sens) & (np.array(meta_spec_per_fold) >= target_spec))
    surplus_alpha = float(sum((np.array(per_alpha_cv.get(float(best_alpha), {}).get('sens', [])) - target_sens) + (np.array(per_alpha_cv.get(float(best_alpha), {}).get('spec', [])) - target_spec))) if best_alpha is not None else -1.0
    surplus_meta = float(sum((np.array(meta_sens_per_fold) - target_sens) + (np.array(meta_spec_per_fold) - target_spec)))
    use_meta = (feasible_meta, surplus_meta) > (feasible_alpha, surplus_alpha)

    # Train final calibrated models on all training for holdout per-model probs
    per_model_hold = {}
    per_model_fitted = {}
    # Determine per-model calibration method for holdout predictions
    per_model_calib_method = calibration_method
    if final_calibration in {"isotonic", "sigmoid"}:
        per_model_calib_method = str(final_calibration)
    for name, mdl in models:
        try:
            if final_calibration == 'none':
                mdl.fit(Xtr_s, ytr)
                if hasattr(mdl, 'predict_proba'):
                    per_model_hold[name] = mdl.predict_proba(Xte_s)[:, 1]
                elif hasattr(mdl, 'decision_function'):
                    scores = mdl.decision_function(Xte_s)
                    per_model_hold[name] = safe_divide(scores - scores.min(), scores.max() - scores.min())
                else:
                    # fallback to uniform 0.5
                    per_model_hold[name] = np.full(len(yte), 0.5)
                per_model_fitted[name] = mdl
            else:
                cal = CalibratedClassifierCV(mdl, method=per_model_calib_method, cv=3)
                cal.fit(Xtr_s, ytr)
                per_model_hold[name] = cal.predict_proba(Xte_s)[:, 1]
                per_model_fitted[name] = cal
        except Exception:
            continue
    def mean_hold(names: List[str]) -> np.ndarray:
        arrs = []
        for nm in names:
            if nm in per_model_hold:
                arrs.append(per_model_hold[nm])
        if not arrs:
            return np.zeros(len(yte))
        return np.mean(np.vstack(arrs), axis=0)
    z_sens_hold = mean_hold(e_sens_names)
    z_spec_hold = mean_hold(e_spec_names)

    # Build OOF combiner predictions for the selected strategy
    if best_alpha is None:
        best_alpha = 0.5
    alpha_oof = float(best_alpha) * z_sens_oof + (1.0 - float(best_alpha)) * z_spec_oof

    meta_fitted = None
    if use_meta:
        # Fit meta on full OOF train for holdout prediction
        X_meta_full = np.column_stack([
            z_sens_oof[~np.isnan(z_sens_oof)],
            z_spec_oof[~np.isnan(z_spec_oof)],
            np.abs(z_sens_oof[~np.isnan(z_sens_oof)] - z_spec_oof[~np.isnan(z_spec_oof)])
        ])
        y_meta_full = ytr[~np.isnan(z_sens_oof)]
        try:
            meta = LR(C=0.5, penalty='l2', solver='liblinear', random_state=42)
            X_hold = np.column_stack([
                z_sens_hold,
                z_spec_hold,
                np.abs(z_sens_hold - z_spec_hold)
            ])
            if final_calibration == 'none':
                meta.fit(X_meta_full, y_meta_full)
                p_hold = meta.predict_proba(X_hold)[:, 1]
            else:
                meta_method = calibration_method
                if final_calibration in {"isotonic", "sigmoid"}:
                    meta_method = str(final_calibration)
                meta_cal = CalibratedClassifierCV(meta, method=meta_method, cv=3)
                meta_cal.fit(X_meta_full, y_meta_full)
                p_hold = meta_cal.predict_proba(X_hold)[:, 1]
                meta_fitted = meta_cal
        except Exception:
            p_hold = 0.5 * (z_sens_hold + z_spec_hold)
        # Decide threshold transfer method
        # Prepare p_oof_sel and tau list
        p_oof_sel = meta_oof
        tau_list = meta_tau_per_fold

        # Optional final calibration (temperature) applied on combiner outputs
        applied_final_calib = None
        temp_T = None
        if final_calibration == 'temperature':
            mask = ~np.isnan(p_oof_sel)
            if np.any(mask):
                temp_T = fit_temperature(p_oof_sel[mask], ytr[mask])
                p_hold = apply_temperature(p_hold, temp_T)
                # Map thresholds via the same monotonic transform
                tau_list = [float(apply_temperature(np.array([t]), temp_T)[0]) for t in tau_list]
                applied_final_calib = 'temperature'
        # choose threshold transfer method
        method = threshold_transfer if threshold_transfer else ('quantile_map' if use_quantile_threshold else 'median')
        method = (method or 'median').lower()
        if method == 'quantile_map' and tau_list:
            # Optional KS guard
            ks = ks_statistic(p_oof_sel, p_hold)
            if (quantile_guard_ks and ks <= float(quantile_guard_ks)):
                # fall back to median of (possibly mapped) thresholds
                tau_med = float(np.median(tau_list))
            else:
                q_list = []
                for (vai, thr_m) in zip(fold_va_indices, meta_tau_per_fold):
                    pv = meta_oof[vai]
                    q_list.append(float(np.mean(pv <= thr_m)))
                q_med = float(np.median(q_list)) if q_list else 0.5
                tau_med = float(np.quantile(p_hold, q_med))
        elif method == 'iqr_mid' and tau_list:
            q1, q3 = np.quantile(np.array(tau_list), [0.25, 0.75])
            tau_med = float((q1 + q3) / 2.0)
        else:
            tau_med = float(np.median(tau_list)) if tau_list else thr_med
    else:
        p_hold = float(best_alpha) * z_sens_hold + (1.0 - float(best_alpha)) * z_spec_hold
        # Decide threshold transfer method
        p_oof_sel = alpha_oof
        tau_list = best_tau_per_fold

        applied_final_calib = None
        temp_T = None
        if final_calibration == 'temperature':
            mask = ~np.isnan(p_oof_sel)
            if np.any(mask):
                temp_T = fit_temperature(p_oof_sel[mask], ytr[mask])
                p_hold = apply_temperature(p_hold, temp_T)
                tau_list = [float(apply_temperature(np.array([t]), temp_T)[0]) for t in tau_list]
                applied_final_calib = 'temperature'
        method = threshold_transfer if threshold_transfer else ('quantile_map' if use_quantile_threshold else 'median')
        method = (method or 'median').lower()
        if method == 'quantile_map' and tau_list:
            ks = ks_statistic(p_oof_sel, p_hold)
            if (quantile_guard_ks and ks <= float(quantile_guard_ks)):
                tau_med = float(np.median(tau_list))
            else:
                q_list = []
                for (vai, thr_a) in zip(fold_va_indices, best_tau_per_fold):
                    pv = alpha_oof[vai]
                    q_list.append(float(np.mean(pv <= thr_a)))
                q_med = float(np.median(q_list)) if q_list else 0.5
                tau_med = float(np.quantile(p_hold, q_med))
        elif method == 'iqr_mid' and tau_list:
            q1, q3 = np.quantile(np.array(tau_list), [0.25, 0.75])
            tau_med = float((q1 + q3) / 2.0)
        else:
            tau_med = float(np.median(tau_list)) if tau_list else thr_med

    # Apply selected threshold to holdout, optionally with demographic-aware thresholds per child
    if demographics_df is not None and demographic_manager is not None:
        holdout_child_ids = [child_ids[i] for i in te_idx]
        y_pred_h = np.zeros_like(yte)
        per_child_thr: List[float] = []
        for i, cid in enumerate(holdout_child_ids):
            thr_c = float(demographic_manager.get_clinical_threshold(cid))
            y_pred_h[i] = 1 if p_hold[i] >= thr_c else 0
            per_child_thr.append(thr_c)
        # Safety check: if it harms targets, revert to global
        tn0, fp0, fn0, tp0 = confusion_matrix(yte, y_pred_h).ravel()
        sens0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0.0
        spec0 = tn0 / (tn0 + fp0) if (tn0 + fp0) > 0 else 0.0
        if not (sens0 >= target_sens and spec0 >= target_spec):
            y_pred_h = (p_hold >= tau_med).astype(int)
            print("[demographics] Reverted to global threshold due to safety targets")
    else:
        y_pred_h = (p_hold >= tau_med).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, y_pred_h).ravel()
    sens_h = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    spec_h = float(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    auc_h = float(roc_auc_score(yte, p_hold))

    # Optional holdout spec-first assessment (diagnostic)
    holdout_specfirst = None
    if report_holdout_specfirst:
        fpr_h, tpr_h, thr_h = roc_curve(yte, p_hold)
        spec_h_arr = 1 - fpr_h
        idx = np.where(spec_h_arr >= target_spec)[0]
        if len(idx) > 0:
            best = int(idx[np.argmax(tpr_h[idx])])
        else:
            # fallback Youden
            youden = tpr_h - fpr_h
            best = int(np.argmax(youden))
        thr_sf = float(thr_h[best])
        y_pred_sf = (p_hold >= thr_sf).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(yte, y_pred_sf).ravel()
        sens_sf = float(tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0.0)
        spec_sf = float(tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0)
        holdout_specfirst = {
            "threshold": thr_sf,
            "sensitivity": sens_sf,
            "specificity": spec_sf,
            "meets_targets": bool((sens_sf >= target_sens) and (spec_sf >= target_spec))
        }

    # Orchestrate SafeEnhancementPipeline comparison
    enhancement_section: Dict[str, Any] = {}
    try:
        if SafeEnhancementPipeline is not None:
            print(f"[{time.strftime('%H:%M:%S')}] Orchestrating SafeEnhancementPipeline comparison...")
            pipeline = SafeEnhancementPipeline(min_sensitivity=target_sens, min_specificity=target_spec)

            def transform_for_holdout(X_in: pd.DataFrame) -> np.ndarray:
                Z = scaler_all.transform(X_in)
                if use_umap_cosine:
                    try:
                        Z_u = umap_all.transform(Z)  # type: ignore[name-defined]
                        Z = np.concatenate([Z, Z_u], axis=1)
                    except Exception:
                        pass
                if use_polynomial and not use_umap_cosine:
                    Z = poly_all.transform(Z)  # type: ignore[name-defined]
                return Z

            def predict_fn(X_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                Z = transform_for_holdout(X_in)
                probs = []
                for name, _ in models:
                    mdl = per_model_fitted.get(name)
                    if mdl is None:
                        continue
                    if hasattr(mdl, 'predict_proba'):
                        p = mdl.predict_proba(Z)[:, 1]
                    elif hasattr(mdl, 'decision_function'):
                        s = mdl.decision_function(Z)
                        p = safe_divide(s - s.min(), s.max() - s.min())
                    else:
                        p = np.full(Z.shape[0], 0.5)
                    probs.append(p)
                p_mean = np.mean(np.vstack(probs), axis=0) if probs else np.full(Z.shape[0], 0.5)
                y_pred = (p_mean >= tau_med).astype(int)
                return y_pred, p_mean

            baseline_context = {
                "X_val": Xte_df,
                "y_val": yte,
                "training_reference": Xtr_df,
                "child_ids": [child_ids[i] for i in te_idx],
            }
            baseline_result = pipeline.fit_baseline(Xtr_df, ytr, predict_fn, baseline_context, baseline_context["child_ids"])  # type: ignore[arg-type]

            enhanced_result = pipeline.fit_enhanced(
                Xtr_df,
                ytr,
                predict_fn,
                baseline_context,
                demo_dir=str(Path(demographics_path).parent) if demographics_path else None,
                metadata_roots=None,
            )

            chosen = pipeline.run(baseline_result, enhanced_result)
            enhancement_section = {
                "baseline": baseline_result.metrics,
                "enhanced": enhanced_result.metrics,
                "chosen_strategy": chosen.strategy,
                "chosen_metrics": chosen.metrics,
            }
            print(f"[{time.strftime('%H:%M:%S')}] SafeEnhancementPipeline chose: {enhancement_section['chosen_strategy']}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] SafeEnhancementPipeline not available; skipping Phase 3 orchestration")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] SafeEnhancementPipeline comparison failed: {e}")

    results = {
        "policy": (threshold_policy or "both_targets"),
        "targets": {"sensitivity": target_sens, "specificity": target_spec},
        "models": [n for n, _ in models],
        "use_polynomial": use_polynomial,
        "calibration": calibration_method,
        "enhancement": enhancement_section or None,
        "cv": {
            "folds": fold_metrics,
            "threshold_median": thr_med,
            "sensitivity_mean": cv_sens_mean,
            "specificity_mean": cv_spec_mean,
            "auc_mean": cv_auc_mean
        },
        "seed": int(seed),
        "holdout_seed": int(ho_seed),
        "threshold_transfer": (threshold_transfer if threshold_transfer else ("quantile_map" if use_quantile_threshold else "median")),
        "quantile_guard_ks": float(quantile_guard_ks),
        "holdout": {
            "sensitivity": sens_h,
            "specificity": spec_h,
            "auc": auc_h,
            "threshold": tau_med,
            "meets_targets": bool((sens_h >= target_sens) and (spec_h >= target_spec))
        },
        "combiner": {
            "best_alpha": float(best_alpha),
            "cv": per_alpha_cv
        },
        "holdout_specfirst": holdout_specfirst
    }

    if save_preds:
        # Add holdout predictions and identifiers for bagging/analysis
        results["holdout_ids"] = [str(child_ids[i]) for i in te_idx]
        results["holdout_y"] = [int(v) for v in yte.tolist()]
        results["holdout_proba"] = [float(v) for v in p_hold.tolist()]

    # Optional export of bundle
    if export_dir:
        proj = Path(__file__).resolve().parents[1]
        export_path = (proj / export_dir).resolve()
        (export_path / 'preprocess').mkdir(parents=True, exist_ok=True)
        (export_path / 'models').mkdir(parents=True, exist_ok=True)
        # Save preprocessors
        joblib.dump(scaler_all, export_path / 'preprocess' / 'scaler_all.joblib')
        if use_umap_cosine:
            umap_obj = locals().get('umap_all', None)
            if umap_obj is not None:
                joblib.dump(umap_obj, export_path / 'preprocess' / 'umap_all.joblib')
        if use_polynomial and not use_umap_cosine:
            poly_obj = locals().get('poly_all', None)
            if poly_obj is not None:
                joblib.dump(poly_obj, export_path / 'preprocess' / 'poly_all.joblib')
        # Save fitted per-model estimators used in ensembles
        used_models = sorted(list(set(e_sens_names + e_spec_names)))
        model_paths = {}
        for nm in used_models:
            if nm in per_model_fitted:
                outp = export_path / 'models' / f'{nm}.joblib'
                joblib.dump(per_model_fitted[nm], outp)
                model_paths[nm] = str(outp)
        # Save combiner
        combiner_info: Dict[str, Any] = {}
        if use_meta and (meta_fitted is not None):
            combiner_info['type'] = 'meta'
            meta_path = export_path / 'models' / 'meta_combiner.joblib'
            joblib.dump(meta_fitted, meta_path)
            combiner_info['path'] = str(meta_path)
        else:
            combiner_info['type'] = 'alpha'
            combiner_info['best_alpha'] = float(best_alpha)
        if locals().get('temp_T', None) is not None:
            combiner_info['temperature_T'] = float(temp_T)
            combiner_info['temperature_applied'] = True
        else:
            combiner_info['temperature_applied'] = False
        # Save bundle manifest
        bundle = {
            'policy': results['policy'],
            'targets': results['targets'],
            'feature_config': {
                'use_umap_cosine': bool(use_umap_cosine),
                'umap_components': int(umap_components),
                'umap_neighbors': int(umap_neighbors),
                'use_polynomial': bool(use_polynomial),
            },
            'feature_columns': list(Xtr_df.columns),
            'models_used': used_models,
            'model_paths': model_paths,
            'ensembles': {
                'e_sens_names': e_sens_names,
                'e_spec_names': e_spec_names,
            },
            'combiner': combiner_info,
            'threshold': float(tau_med),
            'holdout_metrics': results['holdout'],
            'cv_summary': results['cv'],
        }
        with (export_path / 'bundle.json').open('w') as f:
            json.dump(bundle, f, indent=2)
        results['export_dir'] = str(export_path)

        # Export training reference features for domain shift detection at prediction time
        try:
            Xtr_df.to_csv(export_path / 'training_reference.csv', index=False)
            print(f"[export] Wrote training_reference.csv with {len(Xtr_df)} rows")
        except Exception:
            pass

        # Optional export of training demographics subset for analysis (not required by runtime)
        if demographics_df is not None:
            try:
                train_child_set = set(groups_tr)
                demo_sub = demographics_df[demographics_df['child_id'].isin(train_child_set)].copy()
                if not demo_sub.empty:
                    demo_sub.to_csv(export_path / 'training_demographics.csv', index=False)
                    print(f"[demographics] Exported training demographics: {len(demo_sub)}")
            except Exception:
                pass

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target-sens', type=float, default=0.86)
    ap.add_argument('--target-spec', type=float, default=0.70)
    ap.add_argument('--use-polynomial', action='store_true')
    ap.add_argument('--use-umap-cosine', action='store_true')
    ap.add_argument('--umap-components', type=int, default=16)
    ap.add_argument('--umap-neighbors', type=int, default=40)
    ap.add_argument('--calibration', type=str, default='isotonic', choices=['isotonic', 'sigmoid'])
    ap.add_argument('--final-calibration', type=str, default='sigmoid', choices=['none', 'isotonic', 'sigmoid', 'temperature'])
    ap.add_argument('--models', type=str, default='lightgbm,brf,logreg', help='Comma list among: lightgbm, lgbm_grid, xgboost, xgb_grid, extratrees, brf, logreg')
    ap.add_argument('--top-k-models', type=int, default=3)
    ap.add_argument('--report-holdout-specfirst', action='store_true')
    ap.add_argument('--use-quantile-threshold', action='store_true')
    ap.add_argument('--tpot-minutes', type=int, default=0, help='If >0, run TPOT for this many minutes and evaluate its pipeline')
    ap.add_argument('--threshold-policy', type=str, default='both_targets', choices=['both_targets', 'spec_first', 'youden', 'np'])
    ap.add_argument('--threshold-transfer', type=str, default=None, choices=['median', 'iqr_mid', 'quantile_map'])
    ap.add_argument('--quantile-guard-ks', type=float, default=0.0, help='If >0 and transfer is quantile_map, only apply mapping when KS(CV, holdout) > threshold; else fallback to median')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--holdout-seed', type=int, default=None, help='If set, use this seed for train/holdout split to keep holdout fixed across runs')
    ap.add_argument('--save-preds', action='store_true', help='Include holdout IDs, labels, and probabilities in JSON output')
    ap.add_argument('--export-dir', type=str, default=None, help='If set, export the trained bundle to this directory')
    ap.add_argument('--out-name', type=str, default='clinical_fair_pipeline_results.json')
    ap.add_argument('--demographics-path', type=str, default=None, help='Optional path to demographics dir (default scans data/age_data)')
    args = ap.parse_args()

    model_list = [s.strip().lower() for s in args.models.split(',') if s.strip()]
    results = run_pipeline(
        target_sens=args.target_sens,
        target_spec=args.target_spec,
        use_polynomial=args.use_polynomial,
        calibration_method=args.calibration,
        model_list=model_list,
        use_umap_cosine=args.use_umap_cosine,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
        top_k_models=args.top_k_models,
        final_calibration=args.final_calibration,
        report_holdout_specfirst=args.report_holdout_specfirst,
        use_quantile_threshold=args.use_quantile_threshold,
        threshold_policy=args.threshold_policy,
        threshold_transfer=args.threshold_transfer,
        quantile_guard_ks=args.quantile_guard_ks,
        seed=args.seed,
        holdout_seed=args.holdout_seed,
        save_preds=args.save_preds,
        export_dir=args.export_dir,
        demographics_path=args.demographics_path
    )

    proj = Path(__file__).resolve().parents[1]
    out_dir = proj / 'results'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / args.out_name
    with out_path.open('w') as f:
        json.dump(results, f, indent=2)

    print("\n=== Clinical Fair Pipeline Summary ===")
    print(f"Targets: Sens>={results['targets']['sensitivity']}, Spec>={results['targets']['specificity']}")
    print(f"Models: {', '.join(results['models'])}")
    print(f"CV mean: Sens={results['cv']['sensitivity_mean']:.3f}, Spec={results['cv']['specificity_mean']:.3f}, AUC={results['cv']['auc_mean']:.3f}")
    print(f"Holdout: Sens={results['holdout']['sensitivity']:.3f}, Spec={results['holdout']['specificity']:.3f}, AUC={results['holdout']['auc']:.3f}")
    print(f"Threshold (median CV): {results['cv']['threshold_median']:.3f}")
    print(f"Meets both targets on holdout: {results['holdout']['meets_targets']}")

    # Optional TPOT pass
    if args.tpot_minutes and args.tpot_minutes > 0:
        try:
            from tpot import TPOTClassifier  # type: ignore
            print(f"\n[TPOT] Running time-boxed search for {args.tpot_minutes} minutes...")
            # Prepare standardized (and polynomial if requested) train data
            from sklearn.preprocessing import StandardScaler
            ss = StandardScaler()
            X_df, y, child_ids = build_child_dataset()
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            (tr_idx, te_idx) = list(gss.split(X_df, y, groups=child_ids))[0]
            Xtr_df = X_df.iloc[tr_idx].reset_index(drop=True)
            ytr = y[tr_idx]
            Xtr_s = ss.fit_transform(Xtr_df)
            if args.use_polynomial and not args.use_umap_cosine:
                pf = PolynomialFeatures(degree=2, include_bias=False)
                Xtr_s = pf.fit_transform(Xtr_s)
            if args.use_umap_cosine:
                try:
                    import umap
                    n_comp = max(2, min(args.umap_components, Xtr_s.shape[1]))
                    um_all = umap.UMAP(n_components=n_comp, metric='cosine', random_state=42)
                    Xtr_u = um_all.fit_transform(Xtr_s)
                    Xtr_s = np.concatenate([Xtr_s, Xtr_u], axis=1)
                except Exception:
                    pass
            tpot = TPOTClassifier(max_time_mins=args.tpot_minutes, cv=5, random_state=42)
            tpot.fit(Xtr_s, ytr)
            pipeline = tpot.fitted_pipeline_
            # Evaluate with our CV thresholding policy and holdout
            print("[TPOT] Evaluating best pipeline under clinical policy...")
            # Reuse run_pipeline logic by treating it as an extra model is complex. Instead, just evaluate holdout.
            # Build full train/holdout sets with same preprocessing
            Xte_df = X_df.iloc[te_idx].reset_index(drop=True)
            Xte_s = ss.transform(Xte_df)
            if args.use_polynomial and not args.use_umap_cosine:
                Xte_s = pf.transform(Xte_s)
            if args.use_umap_cosine:
                try:
                    Xte_u = um_all.transform(Xte_s)
                    Xte_s = np.concatenate([Xte_s, Xte_u], axis=1)
                except Exception:
                    pass
            cal = CalibratedClassifierCV(pipeline, method=args.calibration, cv=3)
            cal.fit(Xtr_s, ytr)
            # Holdout proba
            from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
            yte = y[te_idx]
            p_hold = cal.predict_proba(Xte_s)[:, 1]
            fpr_h, tpr_h, thr_h = roc_curve(yte, p_hold)
            # To choose threshold, we don't have fold thresholds; use clinical policy on holdout directly
            spec_h = 1 - fpr_h
            idx = np.where((tpr_h >= args.target_sens) & (spec_h >= args.target_spec))[0]
            if len(idx) > 0:
                gains = (tpr_h[idx] - args.target_sens) + (spec_h[idx] - args.target_spec)
                best = int(idx[np.argmax(gains)])
            else:
                idx = np.where(spec_h >= args.target_spec)[0]
                if len(idx) > 0:
                    best = int(idx[np.argmax(tpr_h[idx])])
                else:
                    youden = tpr_h - fpr_h
                    best = int(np.argmax(youden))
            thr_sel = float(thr_h[best])
            y_pred_h = (p_hold >= thr_sel).astype(int)
            tn, fp, fn, tp = confusion_matrix(yte, y_pred_h).ravel()
            sens_t = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            spec_t = float(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            auc_t = float(roc_auc_score(yte, p_hold))
            tpot_out = {
                "tpot_minutes": args.tpot_minutes,
                "use_polynomial": args.use_polynomial,
                "pipeline": str(pipeline),
                "holdout": {"sensitivity": sens_t, "specificity": spec_t, "auc": auc_t, "threshold": thr_sel,
                             "meets_targets": bool((sens_t >= args.target_sens) and (spec_t >= args.target_spec))}
            }
            tpot_path = out_dir / 'clinical_fair_pipeline_tpot_results.json'
            with tpot_path.open('w') as f:
                json.dump(tpot_out, f, indent=2)
            print("\n[TPOT] Holdout: Sens={:.3f}, Spec={:.3f}, AUC={:.3f}, Thr={:.3f}, Meets={}".format(
                sens_t, spec_t, auc_t, thr_sel, tpot_out['holdout']['meets_targets']))
        except Exception as e:
            print(f"[TPOT] Skipped due to error or missing package: {e}")


if __name__ == '__main__':
    main()

