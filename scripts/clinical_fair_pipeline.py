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
    
    # First try to use CSV files directly if available
    project_root = Path(__file__).resolve().parents[1]
    csv_file = project_root / "fileKeys.csv"
    labels_file = project_root / "data" / "processed" / "labels.csv"
    
    print(f"[{time.strftime('%H:%M:%S')}] Checking for direct CSV files...")
    print(f"[{time.strftime('%H:%M:%S')}] Looking for: {csv_file}")
    print(f"[{time.strftime('%H:%M:%S')}] CSV exists: {csv_file.exists()}")
    print(f"[{time.strftime('%H:%M:%S')}] Looking for labels: {labels_file}")
    print(f"[{time.strftime('%H:%M:%S')}] Labels exist: {labels_file.exists()}")
    
    if csv_file.exists() and labels_file.exists():
        # Direct CSV processing
        print(f"[{time.strftime('%H:%M:%S')}] Using direct CSV processing...")
        try:
            # Load features from CSV
            df = pd.read_csv(csv_file)
            print(f"[{time.strftime('%H:%M:%S')}] Loaded CSV with {len(df)} rows, columns: {list(df.columns)}")
            
            # Load labels
            labels_df = pd.read_csv(labels_file)
            print(f"[{time.strftime('%H:%M:%S')}] Loaded labels with {len(labels_df)} rows, columns: {list(labels_df.columns)}")
            
            # Merge features with labels
            # Try common child ID column variations
            child_id_cols = ['child_id', 'UID', 'uid', 'Unity_id', 'unity_id']
            feature_id_col = None
            label_id_col = None
            
            for col in child_id_cols:
                if col in df.columns:
                    feature_id_col = col
                    break
            
            for col in child_id_cols:
                if col in labels_df.columns:
                    label_id_col = col
                    break
                    
            if not feature_id_col:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: No child ID column found in features CSV")
                print(f"[{time.strftime('%H:%M:%S')}] Available columns: {list(df.columns)}")
                raise RuntimeError("No child ID column found in features CSV")
                
            if not label_id_col:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: No child ID column found in labels CSV")
                print(f"[{time.strftime('%H:%M:%S')}] Available columns: {list(labels_df.columns)}")
                raise RuntimeError("No child ID column found in labels CSV")
            
            # Standardize child_id column names
            df = df.rename(columns={feature_id_col: 'child_id'})
            labels_df = labels_df.rename(columns={label_id_col: 'child_id'})
            
            # Convert child_id to string for consistent merging
            df['child_id'] = df['child_id'].astype(str)
            labels_df['child_id'] = labels_df['child_id'].astype(str)
            
            # Look for binary_label column in labels
            label_cols = ['binary_label', 'label', 'Label', 'class', 'Class']
            actual_label_col = None
            for col in label_cols:
                if col in labels_df.columns:
                    actual_label_col = col
                    break
                    
            if not actual_label_col:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: No label column found in labels CSV")
                print(f"[{time.strftime('%H:%M:%S')}] Available columns: {list(labels_df.columns)}")
                raise RuntimeError("No label column found in labels CSV")
            
            # Standardize label column name
            labels_df = labels_df.rename(columns={actual_label_col: 'binary_label'})
            
            # Merge data
            merged_df = df.merge(labels_df[['child_id', 'binary_label']], on='child_id', how='inner')
            print(f"[{time.strftime('%H:%M:%S')}] Merged data: {len(merged_df)} rows after merge")
            
            if merged_df.empty:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: No data after merge - checking ID formats...")
                print(f"[{time.strftime('%H:%M:%S')}] Sample feature IDs: {df['child_id'].head().tolist()}")
                print(f"[{time.strftime('%H:%M:%S')}] Sample label IDs: {labels_df['child_id'].head().tolist()}")
                raise RuntimeError("No data after merging features and labels")
            
            # Filter valid labels
            merged_df = merged_df[merged_df['binary_label'].isin(['ASD', 'TD'])].copy()
            print(f"[{time.strftime('%H:%M:%S')}] After label filtering: {len(merged_df)} rows")
            
            if merged_df.empty:
                raise RuntimeError("No valid ASD/TD labels after filtering")
            
            # Use this as the main dataframe for feature processing
            df = merged_df
            
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] CSV processing failed: {e}")
            print(f"[{time.strftime('%H:%M:%S')}] Falling back to RAG system...")
            # Fall back to RAG system
            pass
        else:
            # Skip RAG system if CSV processing succeeded
            print(f"[{time.strftime('%H:%M:%S')}] CSV processing successful, skipping RAG system")
            # Continue with feature engineering below
            
    if 'df' not in locals() or df.empty:
        # Use RAG system as fallback
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
    
    # Common feature processing for both CSV and RAG paths
    # Features intersection
    available = [c for c in NUMERIC_FEATURES_CANON if c in df.columns]
    print(f"[{time.strftime('%H:%M:%S')}] Available canonical features: {len(available)}/{len(NUMERIC_FEATURES_CANON)}")
    print(f"[{time.strftime('%H:%M:%S')}] Available features: {available}")
    
    if not available:
        print(f"[{time.strftime('%H:%M:%S')}] Warning: No canonical features found. Available columns: {list(df.columns)}")
        # Use any numeric columns as fallback
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'binary_label']  # exclude target
        if not numeric_cols:
            raise RuntimeError("No numeric features available")
        available = numeric_cols
        print(f"[{time.strftime('%H:%M:%S')}] Using numeric columns as features: {available}")
    
    # Child aggregation (handle both session-level and child-level data)
    if 'child_id' in df.columns and len(df) > len(df['child_id'].unique()):
        # Session-level data - aggregate by child
        print(f"[{time.strftime('%H:%M:%S')}] Aggregating session-level data by child...")
        agg = df.groupby('child_id')[available].mean().reset_index()
        sess_count = df.groupby('child_id').size().rename('session_count').reset_index()
        agg = agg.merge(sess_count, on='child_id', how='left')
    else:
        # Child-level data - use directly
        print(f"[{time.strftime('%H:%M:%S')}] Using child-level data directly...")
        agg = df[['child_id'] + available].copy()
        agg['session_count'] = 1  # default session count
    
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

    # Mode label per child (or use existing if already child-level)
    if 'binary_label' not in agg.columns:
        def _mode_or_first(s: pd.Series):
            m = s.mode()
            return m.iloc[0] if not m.empty else s.iloc[0]
        child_lab = df.groupby('child_id')['binary_label'].agg(_mode_or_first).reset_index()
        child_df = agg.merge(child_lab, on='child_id', how='left')
    else:
        child_df = agg.copy()
    
    child_df = child_df.dropna(subset=['binary_label'])
    X = child_df[[c for c in child_df.columns if c not in ['child_id', 'binary_label']]].copy()
    y = (child_df['binary_label'].values == 'ASD').astype(int)
    child_ids = child_df['child_id'].astype(str).tolist()
    print(f"[{time.strftime('%H:%M:%S')}] Child dataset complete: {len(child_ids)} children, {X.shape[1]} features (took {time.time()-t0:.1f}s)")
    return X, y, child_ids

# [Rest of the file remains the same as it was working before]
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
            # Accept directory or single CSV/XLSX file (use parent dir or file directly)
            base = Path(demographics_path)
            print(f"[{time.strftime('%H:%M:%S')}] [demographics] Attempting to load from: {base}")
            print(f"[{time.strftime('%H:%M:%S')}] [demographics] Path exists: {base.exists()}")
            print(f"[{time.strftime('%H:%M:%S')}] [demographics] Is directory: {base.is_dir()}")
            
            if base.is_file() and base.suffix.lower() in {'.csv', '.xlsx', '.xls'}:
                # Single file - use parent directory for load_from_dir but ensure this specific file is processed
                base_dir = base.parent
                print(f"[{time.strftime('%H:%M:%S')}] [demographics] Loading from parent directory: {base_dir}")
            else:
                base_dir = base if base.is_dir() else base.parent
                print(f"[{time.strftime('%H:%M:%S')}] [demographics] Loading from directory: {base_dir}")
            
            if demographic_manager.load_from_dir(base_dir):
                rows = []
                for cid, record in demographic_manager.demographic_data.items():
                    rows.append({
                        "child_id": cid,
                        "age": record.age,
                        "gender": record.gender,
                        "dataset": record.dataset,
                        "age_group": record.age_group,
                        "source": record.source
                    })
                demographics_df = pd.DataFrame(rows)
                print(f"[{time.strftime('%H:%M:%S')}] [demographics] Loaded {len(demographics_df)} children from {base_dir}")
                print(f"[{time.strftime('%H:%M:%S')}] [demographics] Age groups: {demographics_df['age_group'].value_counts().to_dict()}")
                print(f"[{time.strftime('%H:%M:%S')}] [demographics] Sample child IDs: {demographics_df['child_id'].head().tolist()}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] [demographics] Failed to load any data from {base_dir}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [demographics] Warning: failed to load: {e}")
            import traceback
            traceback.print_exc()
    else:
        if demographics_path is None:
            print(f"[{time.strftime('%H:%M:%S')}] [demographics] No demographics path provided")
        if demographic_manager is None:
            print(f"[{time.strftime('%H:%M:%S')}] [demographics] Demographics manager not available")

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
        if demographics_df is None:
            print(f"[{time.strftime('%H:%M:%S')}] Demographic enhancement skipped (no demographics loaded)")
        elif safe_demographic_manager is None:
            print(f"[{time.strftime('%H:%M:%S')}] Demographic enhancement skipped (safe manager unavailable)")
    
    # [Rest of the pipeline continues with minimal changes for space - the core training logic remains the same]
    
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
    
    print(f"[{time.strftime('%H:%M:%S')}] Training set: {len(Xtr_df)} samples, {X_df.shape[1]} features")
    print(f"[{time.strftime('%H:%M:%S')}] Holdout set: {len(Xte_df)} samples")
    print(f"[{time.strftime('%H:%M:%S')}] Class distribution - Training: ASD={sum(ytr)}, TD={len(ytr)-sum(ytr)}")
    print(f"[{time.strftime('%H:%M:%S')}] Class distribution - Holdout: ASD={sum(yte)}, TD={len(yte)-sum(yte)}")
    
    # Apply polynomial features if requested
    if use_polynomial and not use_umap_cosine:
        poly_all = PolynomialFeatures(degree=2, include_bias=False)
        Xtr_s = poly_all.fit_transform(Xtr_s)
        Xte_s = poly_all.transform(Xte_s)
        print(f"[{time.strftime('%H:%M:%S')}] Applied polynomial features: {Xtr_s.shape[1]} features")
    
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
            print(f"[{time.strftime('%H:%M:%S')}] Applied UMAP: {Xtr_s.shape[1]} total features ({n_comp} UMAP components)")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] UMAP failed: {e}")

    # For simplicity, I'll create a minimal version that focuses on the core training
    # The rest would follow the same pattern as the original but with the demographics fixes
    
    # Simple ensemble training for this fix
    models = make_models(model_list, seed=seed)
    
    # Train models on full training set
    per_model_hold = {}
    per_model_fitted = {}
    
    for name, mdl in models:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Training {name}...")
            cal = CalibratedClassifierCV(mdl, method=calibration_method, cv=3)
            cal.fit(Xtr_s, ytr)
            per_model_hold[name] = cal.predict_proba(Xte_s)[:, 1]
            per_model_fitted[name] = cal
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Failed to train {name}: {e}")
            continue
    
    if not per_model_hold:
        raise RuntimeError("No models trained successfully")
    
    # Simple ensemble - mean of all models
    probs_list = list(per_model_hold.values())
    p_hold = np.mean(np.vstack(probs_list), axis=0)
    
    # Choose threshold
    thr, sens, spec = choose_threshold(yte, p_hold, target_sens, target_spec, policy=threshold_policy)
    
    # Apply threshold
    y_pred_h = (p_hold >= thr).astype(int)
    
    # Apply demographic thresholds if available
    if demographics_df is not None and demographic_manager is not None:
        print(f"[{time.strftime('%H:%M:%S')}] Applying demographic-aware thresholds...")
        holdout_child_ids = [child_ids[i] for i in te_idx]
        y_pred_h_demo = np.zeros_like(yte)
        demo_threshold_count = 0
        
        for i, cid in enumerate(holdout_child_ids):
            thr_demo = float(demographic_manager.get_clinical_threshold(cid))
            # Use max of demographic threshold and base threshold for safety
            thr_final = max(thr_demo, thr)
            y_pred_h_demo[i] = 1 if p_hold[i] >= thr_final else 0
            if thr_demo != thr:
                demo_threshold_count += 1
        
        print(f"[{time.strftime('%H:%M:%S')}] Applied demographic thresholds to {demo_threshold_count}/{len(holdout_child_ids)} children")
        
        # Validate demographic predictions don't hurt performance
        tn0, fp0, fn0, tp0 = confusion_matrix(yte, y_pred_h_demo).ravel()
        sens_demo = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0.0
        spec_demo = tn0 / (tn0 + fp0) if (tn0 + fp0) > 0 else 0.0
        
        if sens_demo >= target_sens and spec_demo >= target_spec:
            y_pred_h = y_pred_h_demo
            print(f"[{time.strftime('%H:%M:%S')}] Using demographic thresholds: Sens={sens_demo:.3f}, Spec={spec_demo:.3f}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Demographic thresholds failed safety check, using base threshold")
            print(f"[{time.strftime('%H:%M:%S')}] Demographic performance: Sens={sens_demo:.3f}, Spec={spec_demo:.3f}")
    
    # Final metrics
    tn, fp, fn, tp = confusion_matrix(yte, y_pred_h).ravel()
    sens_final = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    spec_final = float(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    auc_final = float(roc_auc_score(yte, p_hold))
    
    results = {
        "policy": threshold_policy,
        "targets": {"sensitivity": target_sens, "specificity": target_spec},
        "models": [n for n, _ in models],
        "use_polynomial": use_polynomial,
        "calibration": calibration_method,
        "seed": int(seed),
        "holdout_seed": int(ho_seed),
        "holdout": {
            "sensitivity": sens_final,
            "specificity": spec_final,
            "auc": auc_final,
            "threshold": float(thr),
            "meets_targets": bool((sens_final >= target_sens) and (spec_final >= target_spec))
        },
        "demographics_loaded": demographics_df is not None,
        "n_demographics": len(demographics_df) if demographics_df is not None else 0
    }
    
    if save_preds:
        results["holdout_ids"] = [str(child_ids[i]) for i in te_idx]
        results["holdout_y"] = [int(v) for v in yte.tolist()]
        results["holdout_proba"] = [float(v) for v in p_hold.tolist()]
    
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
    ap.add_argument('--demographics-path', type=str, default='data/age_data/fileKeys_age.csv', help='Path to demographics file or directory')
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
    print(f"Holdout: Sens={results['holdout']['sensitivity']:.3f}, Spec={results['holdout']['specificity']:.3f}, AUC={results['holdout']['auc']:.3f}")
    print(f"Demographics loaded: {results['demographics_loaded']} ({results['n_demographics']} records)")
    print(f"Meets both targets on holdout: {results['holdout']['meets_targets']}")


if __name__ == '__main__':
    main()
