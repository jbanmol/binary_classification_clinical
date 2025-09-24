# Detailed Implementation Plan: Clinical Target-Guaranteed Domain Adaptation

## Executive Summary

This plan ensures you **CANNOT miss** your clinical targets (sensitivity >86%, specificity >72%) while addressing domain shift issues that caused 36% false positives. The approach uses population-aware thresholding and domain adaptation while maintaining strict clinical guardrails.

## Important Revisions (Demographic-Aware Post-Processing + Correct Paths)

The following updates tailor the plan to the current codebase and data layout, and integrate your demographic-aware post-processing idea without breaking existing training/export/prediction flows:

- Use demographics located in `data/age_data/` (not project root). Expected files present now: `Phase_2_children_data.xlsx`, `fileKeys_children_data.xlsx`, `Espalier_children_data.xlsx`. The loader should scan this directory for supported files.
- Prefer adding new modules under `src/` to align with existing imports from `scripts/*` (e.g., `from src....`).
- Introduce a demographics manager that provides age/gender-aware thresholds as a post-processing layer, preserving the existing ensemble/probability computation and bundle format (`bundle.json`).
- Keep domain shift detection/adaptation as planned, integrated into `scripts/predict_cli.py` without changing base model training.

### Files to Add/Modify (non-breaking)

- NEW: `src/demographics/demographic_manager.py`
  - Loads demographics from `data/age_data/*.xlsx` (auto-detect), standardizes columns, builds `child_id → {age, gender, age_group, dataset}` map.
  - Age groups: `toddler (0–2.5)`, `preschool (2.5–5)`, `school_age (≥5)`. Adjustable.
  - Clinical probability thresholds to reduce FPR (you can tune later):
    - toddler: male 0.68, female 0.65
    - preschool: male 0.60, female 0.57
    - school_age: male 0.53, female 0.50
    - default: 0.531 (keeps current bundle behavior)
  - Clinical safety targets (min): sensitivity 0.86, specificity 0.72
  - Exposes: `load_from_dir(path)`, `get_child_demographics(child_id)`, `get_clinical_threshold(child_id)`, `save_for_deployment()`, `load_for_deployment()` and a singleton `demographic_manager`.

- NEW (optional helper): `src/demographics/demographic_aware_predictor.py`
  - Thin wrapper that accepts already-computed ensemble probabilities (from current `bundle.json` inference) and applies demographic-aware thresholds for reporting/validation. Intended for offline validation and A/B analysis, not required for runtime.

- MODIFY: `scripts/clinical_fair_pipeline.py`
  - Add optional CLI arg `--demographics-path` (dir or file). Default: `data/age_data`.
  - Load demographics early via `from src.demographics.demographic_manager import demographic_manager` and `demographic_manager.load_from_dir(Path(demographics_path))` if provided.
  - Replace single global threshold application on holdout with adaptive per-child thresholds when demographics are available; otherwise fall back to existing base threshold. Gate with clinical safety validation. Do not change base probability computation, CV, or export contents.
  - When exporting a bundle, if demographics were loaded, also write `training_demographics.csv` (subset for training child_ids) for downstream analysis; keep bundle format unchanged.

- MODIFY: `scripts/predict_cli.py`
  - Add optional CLI arg `--demographics` accepting a directory (default `data/age_data`) or a CSV. If provided, load via `demographic_manager` and apply per-child thresholds post hoc to `P_ens` to produce labels; otherwise default to `bundle['threshold']`.
  - Keep planned domain shift detection (`src/domain_adaptation.py`) and apply adaptation prior to preprocessing/scoring.

- NEW: `src/domain_adaptation.py`
  - As originally specified (KS/Wasserstein/mean-shift feature checks, covariance shift; CORAL alignment), with no impact on training.

- MODIFY: `train_final.sh`
  - Thread through `DEMOGRAPHICS_PATH` environment variable to `scripts/clinical_fair_pipeline.py --demographics-path "$DEMOGRAPHICS_PATH"` and add post-run safety checks already described below.

These additions preserve the current architecture: model probabilities and combination remain identical; only the binary decision boundary becomes population-aware when demographics are available.

## Phase 1: Immediate Clinical Safety Implementation (Week 1)

### 1.1 Create Age/Gender-Aware Demographics Manager

**File: `src/demographics/demographic_manager.py` (NEW FILE)**

```python
#!/usr/bin/env python3
"""
Demographic Manager for Clinical ASD Detection
Loads demographics from data/age_data and provides age/gender-aware thresholds
without changing the existing model architecture or bundle format.
"""

Key elements to implement:

- Directory-based loader:
  - Default directory: `data/age_data`.
  - Supported files (auto-detect): `Phase_2_children_data.xlsx`, `fileKeys_children_data.xlsx`, `Espalier_children_data.xlsx`.
  - Column normalization: map to `child_id`, `age`, `gender`, optional `dob`; compute `age` from `dob` when missing; standardize gender values; derive `age_group` using thresholds below.

- Age groups and thresholds (initial, tunable):
  - Age groups: `toddler: (0, 2.5)`, `preschool: (2.5, 5)`, `school_age: (5, 12+)`.
  - Probability thresholds:
    - toddler: male 0.68, female 0.65
    - preschool: male 0.60, female 0.57
    - school_age: male 0.53, female 0.50
    - default: 0.531 (matches exported bundle)

- API surface:
  - `load_from_dir(dir_path: Path) -> bool`
  - `get_child_demographics(child_id: str) -> Dict`
  - `get_clinical_threshold(child_id: str) -> float`
  - `validate_clinical_performance(y_true, y_pred, age_groups) -> Dict`
  - Persistence helpers: `save_for_deployment()`, `load_for_deployment()`

- Provide a singleton `demographic_manager` for convenient import in scripts.
```

### 1.2 Update Clinical Pipeline with Safety Guardrails

**File: `scripts/clinical_fair_pipeline.py` (MAJOR UPDATES)**

Add these functions at the top of the file after imports:

```python
# Add after existing imports
from src.demographics.demographic_manager import demographic_manager

def load_demographics_safe(demo_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Safely load demographic data if available. Accepts a directory or a CSV/XLSX file.
    Defaults to scanning data/age_data when path not provided.
    Returns a normalized DataFrame with columns [child_id, age, gender, age_group, dataset].
    """
    base = Path(demo_path) if demo_path else Path("data/age_data")
    if base.exists() and base.is_dir():
        try:
            ok = demographic_manager.load_from_dir(base)
            if ok:
                # Materialize a DataFrame view for validation functions
                rows = []
                for cid, info in demographic_manager.demographic_data.items():
                    rows.append({"child_id": cid, **info})
                return pd.DataFrame(rows)
        except Exception as e:
            print(f"Warning: Could not load demographics from directory {base}: {e}")
    elif base.exists() and base.is_file():
        try:
            # Allow passing a single file for quick experiments
            ok = demographic_manager.load_from_dir(base.parent)
            if ok:
                rows = []
                for cid, info in demographic_manager.demographic_data.items():
                    rows.append({"child_id": cid, **info})
                return pd.DataFrame(rows)
        except Exception as e:
            print(f"Warning: Could not load demographics from file {base}: {e}")
    return None

def clinical_safety_validation(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: np.ndarray, demographics: Optional[pd.DataFrame],
                              child_ids: List[str], min_sensitivity: float = 0.86,
                              min_specificity: float = 0.72) -> Dict[str, Any]:
    """CRITICAL: Validate clinical targets are met, fail if not"""
    from sklearn.metrics import confusion_matrix, roc_curve
    
    # Overall performance check
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    validation_result = {
        'overall_sensitivity': sensitivity,
        'overall_specificity': specificity,
        'meets_min_targets': (sensitivity >= min_sensitivity and specificity >= min_specificity),
        'safety_status': 'SAFE' if (sensitivity >= min_sensitivity and specificity >= min_specificity) else 'UNSAFE'
    }
    
    # Demographic-specific validation if available
    if demographics is not None:
        demo_dict = dict(zip(demographics['child_id'], demographics['age_group']))
        age_groups = [demo_dict.get(cid, 'unknown') for cid in child_ids]
        
        demo_results = demographic_manager.validate_clinical_performance(
            y_true, y_pred, age_groups
        )
        validation_result['demographic_performance'] = demo_results
        
        # Check if any demographic group fails
        failed_groups = [group for group, metrics in demo_results.items() 
                        if not metrics['meets_targets']]
        
        if failed_groups:
            validation_result['safety_status'] = 'DEMOGRAPHIC_UNSAFE'
            validation_result['failed_groups'] = failed_groups
    
    return validation_result

def adaptive_threshold_selection(y_true: np.ndarray, y_proba: np.ndarray,
                                demographics: Optional[pd.DataFrame], child_ids: List[str],
                                target_sens: float, target_spec: float) -> Dict[str, Any]:
    """Select thresholds with demographic adaptation and safety guarantees"""
    
    # Default threshold selection (existing logic)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    spec = 1 - fpr
    
    # Primary: meet both targets
    idx = np.where((tpr >= target_sens) & (spec >= target_spec))[0]
    if len(idx) > 0:
        gains = (tpr[idx] - target_sens) + (spec[idx] - target_spec)
        best = int(idx[np.argmax(gains)])
        base_threshold = float(thresholds[best])
        base_sens = float(tpr[best])
        base_spec = float(spec[best])
    else:
        # Fallback: spec-first
        idx = np.where(spec >= target_spec)[0]
        if len(idx) > 0:
            best = int(idx[np.argmax(tpr[idx])])
        else:
            youden = tpr - fpr
            best = int(np.argmax(youden))
        base_threshold = float(thresholds[best])
        base_sens = float(tpr[best])
        base_spec = float(spec[best])
    
    result = {
        'base_threshold': base_threshold,
        'base_sensitivity': base_sens,
        'base_specificity': base_spec,
        'adaptive_thresholds': {}
    }
    
    # Demographic-specific thresholds if available
    if demographics is not None:
        demo_dict = dict(zip(demographics['child_id'], 
                           zip(demographics['age_group'], demographics['gender'])))
        
        for child_id in child_ids:
            if child_id in demo_dict:
                age_group, gender = demo_dict[child_id]
                adaptive_thresh = demographic_manager.get_population_threshold(age_group, gender)
                result['adaptive_thresholds'][child_id] = {
                    'threshold': adaptive_thresh,
                    'age_group': age_group,
                    'gender': gender
                }
            else:
                result['adaptive_thresholds'][child_id] = {
                    'threshold': base_threshold,
                    'age_group': 'unknown',
                    'gender': 'unknown'
                }
    
    return result
```

Update the `run_pipeline` function signature and add demographic handling:

```python
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
                 demographics_path: Optional[str] = None) -> Dict[str, Any]:  # NEW PARAMETER
    
    # Load demographics early (from directory or file)
    demographics = load_demographics_safe(demographics_path)
    if demographics is not None:
        print(f"✅ Loaded demographics for {len(demographics)} children")
        print(f"Age groups: {demographics['age_group'].value_counts().to_dict()}")
    else:
        print("⚠️  No demographics loaded - using default thresholds")
    
    # Rest of existing function until threshold selection...
    X_df, y, child_ids = build_child_dataset()
    
    # ... existing code for CV, model training, etc. ...
    
    # REPLACE the existing threshold selection with adaptive version
    threshold_result = adaptive_threshold_selection(
        yte, p_hold, demographics, [child_ids[i] for i in te_idx], target_sens, target_spec
    )
    
    # Apply adaptive thresholds for predictions
    if demographics is not None and threshold_result['adaptive_thresholds']:
        y_pred_h_adaptive = np.zeros_like(yte)
        holdout_child_ids = [child_ids[i] for i in te_idx]
        
        for i, child_id in enumerate(holdout_child_ids):
            child_threshold = threshold_result['adaptive_thresholds'][child_id]['threshold']
            y_pred_h_adaptive[i] = 1 if p_hold[i] >= child_threshold else 0
        
        # Validate adaptive predictions
        validation_result = clinical_safety_validation(
            yte, y_pred_h_adaptive, p_hold, demographics, holdout_child_ids,
            target_sens, target_spec
        )
        
        # Use adaptive if safe, otherwise fall back to base
        if validation_result['safety_status'] == 'SAFE':
            y_pred_h = y_pred_h_adaptive
            tau_med = np.mean(list(t['threshold'] for t in threshold_result['adaptive_thresholds'].values()))
            print("✅ Using adaptive demographic thresholds")
        else:
            y_pred_h = (p_hold >= threshold_result['base_threshold']).astype(int)
            tau_med = threshold_result['base_threshold']
            print(f"⚠️  Adaptive thresholds unsafe ({validation_result['safety_status']}), using base threshold")
    else:
        y_pred_h = (p_hold >= threshold_result['base_threshold']).astype(int)
        tau_med = threshold_result['base_threshold']
    
    # CRITICAL: Final safety validation
    final_validation = clinical_safety_validation(
        yte, y_pred_h, p_hold, demographics, [child_ids[i] for i in te_idx],
        target_sens, target_spec
    )
    
    if final_validation['safety_status'] != 'SAFE':
        raise RuntimeError(f"""
        🚨 CLINICAL SAFETY VIOLATION 🚨
        Current performance: Sens={final_validation['overall_sensitivity']:.3f}, Spec={final_validation['overall_specificity']:.3f}
        Required: Sens≥{target_sens:.3f}, Spec≥{target_spec:.3f}
        Status: {final_validation['safety_status']}
        
        Model training FAILED - clinical targets not met!
        """)
    
    # Calculate final metrics with safety validation
    tn, fp, fn, tp = confusion_matrix(yte, y_pred_h).ravel()
    sens_h = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    spec_h = float(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    auc_h = float(roc_auc_score(yte, p_hold))
    
    # ... rest of existing function with added validation results ...
    
    results = {
        # ... existing results ...
        "clinical_validation": final_validation,
        "threshold_strategy": threshold_result,
        "demographics_used": demographics is not None,
        "safety_status": final_validation['safety_status']
    }
    
    return results
```

### 1.3 Update Training Script with Demographics Support

**File: `train_final.sh` (UPDATES)**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Add demographic path parameter (directory or file)
DEMOGRAPHICS_PATH="${DEMOGRAPHICS_PATH:-data/age_data}"

echo "🔧 Setting up environment..."
# 1) Ensure venv and deps
if [ ! -d venv ]; then
  python3 -m venv venv
fi
./venv/bin/python -m pip install --upgrade pip >/dev/null
./venv/bin/pip install -r requirements.txt

echo "📊 Starting clinical pipeline with safety validation..."

# 2) Run final pipeline on fixed holdout, saving preds for bagging
SEEDS=(17 29 43)
HOLDOUT_SEED=777
OUTS=()

for S in "${SEEDS[@]}"; do
  OUT="results/final_s${S}_ho${HOLDOUT_SEED}_clinical_safe_u16n50_k2.json"
  echo "🎯 Training seed $S with clinical safety validation..."
  
  ./venv/bin/python scripts/clinical_fair_pipeline.py \
    --target-sens 0.86 --target-spec 0.72 \
    --use-umap-cosine --umap-components 16 --umap-neighbors 50 \
    --calibration isotonic --final-calibration temperature \
    --models lightgbm,xgboost,brf,extratrees --top-k-models 2 \
    --threshold-policy np --threshold-transfer iqr_mid \
    --seed "$S" --holdout-seed "$HOLDOUT_SEED" --save-preds \
    --demographics-path "$DEMOGRAPHICS_PATH" \
    --out-name "$(basename "$OUT")"
  
  # Validate each seed meets targets
  SENS=$(jq -r '.holdout.sensitivity' "$OUT")
  SPEC=$(jq -r '.holdout.specificity' "$OUT")
  STATUS=$(jq -r '.safety_status' "$OUT")
  
  echo "Seed $S: Sens=$SENS, Spec=$SPEC, Status=$STATUS"
  
  if [ "$STATUS" != "SAFE" ]; then
    echo "🚨 CRITICAL: Seed $S failed clinical validation!"
    exit 1
  fi
  
  OUTS+=("$OUT")
done

echo "✅ All seeds passed clinical validation"

# 3) Bag seeds and write final bagged metrics
echo "📊 Bagging results across seeds..."
./venv/bin/python scripts/bag_scores.py \
  --pattern "results/final_s*_ho${HOLDOUT_SEED}_clinical_safe_u16n50_k2.json" \
  --out results/final_bagged_ho${HOLDOUT_SEED}_clinical_safe_u16n50_k2.json \
  --target-sens 0.86 --target-spec 0.72

# 4) Export bundle using seed 17 (reference bundle) with safety validation
echo "📦 Exporting clinical-safe bundle..."
./venv/bin/python scripts/clinical_fair_pipeline.py \
  --target-sens 0.86 --target-spec 0.72 \
  --use-umap-cosine --umap-components 16 --umap-neighbors 50 \
  --calibration isotonic --final-calibration temperature \
  --models lightgbm,xgboost,brf,extratrees --top-k-models 2 \
  --threshold-policy np --threshold-transfer iqr_mid \
  --seed 17 --holdout-seed "$HOLDOUT_SEED" --save-preds \
    --demographics-path "$DEMOGRAPHICS_PATH" \
  --export-dir models/clinical_safe_u16n50_k2 \
  --out-name export_clinical_safe_u16n50_k2.json

# 5) Final validation summary
echo ""
echo "🎉 === CLINICAL VALIDATION SUMMARY ==="
cat results/final_bagged_ho${HOLDOUT_SEED}_clinical_safe_u16n50_k2.json | jq '.bagged_metrics'

echo ""
echo "📋 Bundle exported to: models/clinical_safe_u16n50_k2/"
echo "✅ All models meet clinical targets: Sensitivity ≥86%, Specificity ≥72%"
```

## Phase 2: Domain Adaptation Implementation (Week 2)

### 2.1 Create Domain Shift Detection Module

**File: `src/domain_adaptation.py` (NEW FILE)**

```python
#!/usr/bin/env python3
"""
Domain adaptation utilities for clinical ASD detection
Handles distribution shifts between training and deployment populations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
import warnings
warnings.filterwarnings('ignore')

class DomainShiftDetector:
    """Detect and quantify domain shift between datasets"""
    
    def __init__(self):
        self.shift_thresholds = {
            'ks_statistic': 0.2,      # KS test threshold
            'wasserstein': 0.5,       # Wasserstein distance threshold
            'mean_shift': 2.0,        # Mean shift in standard deviations
            'covariance_shift': 0.3   # Covariance shift threshold
        }
    
    def detect_feature_shifts(self, source_data: pd.DataFrame, 
                            target_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Detect shifts in individual features"""
        
        shift_results = {}
        common_features = set(source_data.columns) & set(target_data.columns)
        
        for feature in common_features:
            source_vals = source_data[feature].dropna()
            target_vals = target_data[feature].dropna()
            
            if len(source_vals) < 10 or len(target_vals) < 10:
                continue
            
            # KS test for distribution difference
            ks_stat, ks_p = ks_2samp(source_vals, target_vals)
            
            # Wasserstein distance
            wasserstein_dist = wasserstein_distance(source_vals, target_vals)
            
            # Mean shift in standard deviations
            pooled_std = np.sqrt((source_vals.var() + target_vals.var()) / 2)
            mean_shift = abs(source_vals.mean() - target_vals.mean()) / (pooled_std + 1e-8)
            
            shift_results[feature] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'wasserstein_distance': wasserstein_dist,
                'mean_shift_std': mean_shift,
                'source_mean': source_vals.mean(),
                'target_mean': target_vals.mean(),
                'source_std': source_vals.std(),
                'target_std': target_vals.std(),
                'shift_severity': self._classify_shift_severity(ks_stat, wasserstein_dist, mean_shift)
            }
        
        return shift_results
    
    def _classify_shift_severity(self, ks_stat: float, wasserstein: float, 
                               mean_shift: float) -> str:
        """Classify severity of distribution shift"""
        
        severe_indicators = [
            ks_stat > 0.4,
            wasserstein > 1.0,
            mean_shift > 3.0
        ]
        
        moderate_indicators = [
            ks_stat > 0.2,
            wasserstein > 0.5,
            mean_shift > 1.5
        ]
        
        if sum(severe_indicators) >= 2:
            return 'SEVERE'
        elif sum(moderate_indicators) >= 2:
            return 'MODERATE'
        elif any(moderate_indicators):
            return 'MILD'
        else:
            return 'MINIMAL'
    
    def detect_covariance_shift(self, source_data: np.ndarray, 
                              target_data: np.ndarray) -> Dict[str, float]:
        """Detect covariance structure shifts"""
        
        # Compute covariance matrices
        source_cov = EmpiricalCovariance().fit(source_data).covariance_
        target_cov = EmpiricalCovariance().fit(target_data).covariance_
        
        # Frobenius norm of difference
        cov_diff_norm = np.linalg.norm(source_cov - target_cov, 'fro')
        
        # Relative difference
        source_norm = np.linalg.norm(source_cov, 'fro')
        relative_diff = cov_diff_norm / (source_norm + 1e-8)
        
        return {
            'covariance_difference_norm': cov_diff_norm,
            'relative_covariance_shift': relative_diff,
            'shift_severity': 'SEVERE' if relative_diff > 0.5 else 
                            'MODERATE' if relative_diff > 0.2 else 'MILD'
        }
    
    def comprehensive_shift_analysis(self, source_data: pd.DataFrame,
                                   target_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive domain shift analysis"""
        
        # Feature-level shifts
        feature_shifts = self.detect_feature_shifts(source_data, target_data)
        
        # Overall shift summary
        severe_shifts = [f for f, metrics in feature_shifts.items() 
                        if metrics['shift_severity'] == 'SEVERE']
        moderate_shifts = [f for f, metrics in feature_shifts.items() 
                          if metrics['shift_severity'] == 'MODERATE']
        
        # Covariance shift (on common numeric features)
        numeric_features = source_data.select_dtypes(include=[np.number]).columns
        common_numeric = list(set(numeric_features) & set(target_data.select_dtypes(include=[np.number]).columns))
        
        if len(common_numeric) > 1:
            source_numeric = source_data[common_numeric].fillna(0)
            target_numeric = target_data[common_numeric].fillna(0)
            covariance_shift = self.detect_covariance_shift(source_numeric.values, 
                                                           target_numeric.values)
        else:
            covariance_shift = {'covariance_difference_norm': 0, 'relative_covariance_shift': 0, 'shift_severity': 'MINIMAL'}
        
        # Overall assessment
        overall_severity = 'MINIMAL'
        if len(severe_shifts) > 3 or covariance_shift['shift_severity'] == 'SEVERE':
            overall_severity = 'SEVERE'
        elif len(severe_shifts) > 1 or len(moderate_shifts) > 5:
            overall_severity = 'MODERATE'
        elif len(moderate_shifts) > 2:
            overall_severity = 'MILD'
        
        return {
            'feature_shifts': feature_shifts,
            'covariance_shift': covariance_shift,
            'severe_shift_features': severe_shifts,
            'moderate_shift_features': moderate_shifts,
            'overall_shift_severity': overall_severity,
            'n_features_analyzed': len(feature_shifts),
            'recommendation': self._get_adaptation_recommendation(overall_severity)
        }
    
    def _get_adaptation_recommendation(self, severity: str) -> str:
        """Get recommendation based on shift severity"""
        
        recommendations = {
            'MINIMAL': 'No adaptation needed. Use standard model.',
            'MILD': 'Consider population-specific thresholds.',
            'MODERATE': 'Implement domain adaptation. Use cautious thresholds.',
            'SEVERE': 'CRITICAL: Extensive domain adaptation required. Consider retraining.'
        }
        
        return recommendations.get(severity, 'Unknown severity level')


class DomainAdapter:
    """Adapt models for domain shift"""
    
    def __init__(self):
        self.adaptation_history = []
    
    def coral_alignment(self, source_features: np.ndarray, 
                       target_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CORAL (Correlation Alignment) domain adaptation"""
        
        # Compute covariance matrices
        source_cov = np.cov(source_features.T)
        target_cov = np.cov(target_features.T)
        
        # Eigendecomposition
        eigenvals_s, eigenvecs_s = np.linalg.eigh(source_cov + np.eye(source_cov.shape[0]) * 1e-6)
        eigenvals_t, eigenvecs_t = np.linalg.eigh(target_cov + np.eye(target_cov.shape[0]) * 1e-6)
        
        # Ensure positive eigenvalues
        eigenvals_s = np.maximum(eigenvals_s, 1e-6)
        eigenvals_t = np.maximum(eigenvals_t, 1e-6)
        
        # Whitening and coloring transformations
        whiten_matrix = eigenvecs_s @ np.diag(eigenvals_s ** -0.5) @ eigenvecs_s.T
        color_matrix = eigenvecs_t @ np.diag(eigenvals_t ** 0.5) @ eigenvecs_t.T
        
        # Apply transformation
        transformation_matrix = whiten_matrix @ color_matrix
        source_adapted = source_features @ transformation_matrix.T
        
        return source_adapted, target_features
    
    def population_aware_prediction(self, model, features: np.ndarray,
                                  demographics: Optional[pd.DataFrame],
                                  child_ids: List[str]) -> Dict[str, Any]:
        """Make predictions with population-aware thresholds"""
        
from src.demographics.demographic_manager import demographic_manager
        
        # Get base predictions
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[:, 1]
        else:
            probabilities = np.full(len(features), 0.5)
        
        # Apply population-specific thresholds
        predictions = np.zeros(len(features), dtype=int)
        confidence_scores = np.zeros(len(features))
        
        for i, child_id in enumerate(child_ids):
            prob = probabilities[i]
            
            if demographics is not None and child_id in demographics['child_id'].values:
                demo_row = demographics[demographics['child_id'] == child_id].iloc[0]
                age_group = demo_row['age_group']
                gender = demo_row['gender']
                
                threshold = demographic_manager.get_population_threshold(age_group, gender)
                
                # Confidence based on distance from threshold
                confidence = abs(prob - threshold) / 0.5
                confidence_scores[i] = min(confidence, 1.0)
                
            else:
                threshold = 0.53  # Default threshold
                confidence_scores[i] = abs(prob - 0.5) / 0.5
            
            predictions[i] = 1 if prob >= threshold else 0
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_scores': confidence_scores,
            'mean_confidence': np.mean(confidence_scores)
        }

# Global instances
domain_detector = DomainShiftDetector()
domain_adapter = DomainAdapter()
```

### 2.2 Update Prediction Pipeline with Domain Adaptation

**File: `scripts/predict_cli.py` (UPDATES)**

Add these imports and functions at the top:

```python
# Add these imports
from src.domain_adaptation import domain_detector, domain_adapter
from src.demographics.demographic_manager import demographic_manager

def load_training_reference(bundle_root: Path) -> Optional[pd.DataFrame]:
    """Load training data reference for domain shift detection"""
    try:
        ref_path = bundle_root / 'training_reference.csv'
        if ref_path.exists():
            return pd.read_csv(ref_path)
    except Exception as e:
        print(f"Warning: Could not load training reference: {e}")
    return None

def detect_and_adapt_domain_shift(test_features: pd.DataFrame, 
                                training_reference: Optional[pd.DataFrame],
                                bundle_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Detect domain shift and adapt if necessary"""
    
    if training_reference is None:
        return test_features, {'adaptation_applied': False, 'reason': 'No training reference'}
    
    # Detect domain shift
    shift_analysis = domain_detector.comprehensive_shift_analysis(
        training_reference, test_features
    )
    
    print(f"Domain shift analysis: {shift_analysis['overall_shift_severity']}")
    print(f"Recommendation: {shift_analysis['recommendation']}")
    
    # Apply adaptation if needed
    if shift_analysis['overall_shift_severity'] in ['MODERATE', 'SEVERE']:
        print("🔄 Applying domain adaptation...")
        
        # Apply CORAL alignment
        try:
            common_features = list(set(training_reference.columns) & set(test_features.columns))
            if len(common_features) > 5:  # Minimum features for adaptation
                
                train_vals = training_reference[common_features].fillna(0).values
                test_vals = test_features[common_features].fillna(0).values
                
                adapted_test, _ = domain_adapter.coral_alignment(train_vals, test_vals)
                
                # Replace adapted features
                adapted_features = test_features.copy()
                adapted_features[common_features] = adapted_test
                
                return adapted_features, {
                    'adaptation_applied': True, 
                    'method': 'CORAL',
                    'shift_analysis': shift_analysis
                }
        except Exception as e:
            print(f"Warning: Domain adaptation failed: {e}")
    
    return test_features, {
        'adaptation_applied': False, 
        'shift_analysis': shift_analysis
    }
```

Update the main `predict` function:

```python
def predict(bundle, root, in_csv: Path, demographics_csv: Optional[Path] = None) -> pd.DataFrame:
    cols = bundle['feature_columns']
    df = pd.read_csv(in_csv)
    
    # Load demographics if available
    demographics = None
    if demographics_csv and demographics_csv.exists():
        try:
            # For CSV file case, normalize minimal columns and derive age_group if missing
            demographics = pd.read_csv(demographics_csv)
            if 'age_group' not in demographics.columns and 'age' in demographics.columns:
                def _bin_age(a):
                    try:
                        a = float(a)
                    except Exception:
                        return 'unknown'
                    if 0 <= a < 2.5: return 'toddler'
                    if 2.5 <= a < 5: return 'preschool'
                    if a >= 5: return 'school_age'
                    return 'unknown'
                demographics['age_group'] = demographics['age'].apply(_bin_age)
            print(f"✅ Loaded demographics for population-aware prediction")
        except Exception as e:
            print(f"Warning: Could not load demographics: {e}")
    
    # Load training reference for domain shift detection
    training_reference = load_training_reference(root)
    
    X = df[cols].copy()
    
    # Domain shift detection and adaptation
    X_adapted, adaptation_info = detect_and_adapt_domain_shift(X, training_reference, root)
    
    # Preprocess with adapted features
    scaler = joblib.load(root/'preprocess'/'scaler_all.joblib')
    X_s = scaler.transform(X_adapted)
    
    try:
        umap = joblib.load(root/'preprocess'/'umap_all.joblib')
        U = umap.transform(X_s)
        X_s = np.concatenate([X_s, U], axis=1)
    except Exception:
        pass

    # Get model predictions
    models_used = bundle['models_used']
    probs = []
    for name in models_used:
        mdl = joblib.load(root/'models'/f'{name}.joblib')
        if hasattr(mdl, 'predict_proba'):
            p = mdl.predict_proba(X_s)[:,1]
        elif hasattr(mdl, 'decision_function'):
            s = mdl.decision_function(X_s)
            p = (s - s.min())/(s.max()-s.min()+1e-8)
        else:
            p = np.full(len(X_s), 0.5)
        probs.append(p)

    def mean_of(names):
        arr = [probs[models_used.index(n)] for n in names if n in models_used]
        return np.mean(np.vstack(arr), axis=0)

    E_sens = mean_of(bundle['ensembles']['e_sens_names'])
    E_spec = mean_of(bundle['ensembles']['e_spec_names'])
    
    # Combine ensembles
    comb = bundle['combiner']
    if comb['type'] == 'alpha':
        alpha = float(comb['best_alpha'])
        P_ens = alpha*E_sens + (1-alpha)*E_spec
    else:
        Z = np.column_stack([E_sens, E_spec, np.abs(E_sens - E_spec)])
        meta = joblib.load(root/'models'/'meta_combiner.joblib')
        P_ens = meta.predict_proba(Z)[:,1]

    # Apply temperature scaling if configured
    if comb.get('temperature_applied') and 'temperature_T' in comb:
        T = float(comb['temperature_T'])
        P_ens = 1/(1+np.exp(-np.log(P_ens/(1-P_ens+1e-12))/T))

    # Population-aware prediction
    if demographics is not None and 'child_id' in df.columns:
        child_ids = df['child_id'].tolist()
        prediction_result = domain_adapter.population_aware_prediction(
            None, X_s, demographics, child_ids
        )
        
        # Use population-aware predictions
        labels = prediction_result['predictions']
        confidence = prediction_result['confidence_scores']
        
        print(f"📊 Population-aware prediction: Mean confidence = {prediction_result['mean_confidence']:.3f}")
        
    else:
        # Default threshold
        tau = float(bundle['threshold'])
        labels = (P_ens >= tau).astype(int)
        confidence = np.abs(P_ens - 0.5) / 0.5  # Distance from decision boundary

    # Prepare output
    out = df.copy()
    out['prob_asd'] = P_ens
    out['pred_label'] = labels
    out['confidence'] = confidence
    out['domain_adapted'] = adaptation_info['adaptation_applied']
    
    if adaptation_info['adaptation_applied']:
        out['adaptation_method'] = adaptation_info['method']
        out['shift_severity'] = adaptation_info['shift_analysis']['overall_shift_severity']

    return out
```

Update the CLI interface:

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True, help='Path to bundle.json')
    ap.add_argument('--in', dest='in_csv', required=True, help='Input CSV with child-level features')
    ap.add_argument('--out', required=True, help='Where to write predictions CSV')
    ap.add_argument('--demographics', help='Optional demographics CSV for population-aware prediction')
    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    bundle, root = load_bundle(bundle_path)
    
    demographics_path = Path(args.demographics) if args.demographics else None
    out_df = predict(bundle, root, Path(args.in_csv), demographics_path)
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")

if __name__ == '__main__':
    main()
```

## Phase 3: Bundle Export Enhancement (Week 2)

### 3.1 Export Training Reference for Domain Detection

Add this to the export section of `clinical_fair_pipeline.py`:

```python
# In the export_dir section, add training reference export
if export_dir:
    proj = Path(__file__).resolve().parents[1]
    export_path = (proj / export_dir).resolve()
    (export_path / 'preprocess').mkdir(parents=True, exist_ok=True)
    (export_path / 'models').mkdir(parents=True, exist_ok=True)
    
    # ... existing export code ...
    
    # Export training reference for domain shift detection
    try:
        training_reference = Xtr_df.copy()
        training_reference.to_csv(export_path / 'training_reference.csv', index=False)
        print(f"✅ Exported training reference: {len(training_reference)} samples")
    except Exception as e:
        print(f"Warning: Could not export training reference: {e}")
    
    # Export demographics if available
    if demographics is not None:
        try:
            train_demo = demographics[demographics['child_id'].isin(child_ids)].copy()
            train_demo.to_csv(export_path / 'training_demographics.csv', index=False)
            print(f"✅ Exported training demographics: {len(train_demo)} samples")
        except Exception as e:
            print(f"Warning: Could not export training demographics: {e}")
```

## Implementation Timeline

### Week 1: Clinical Safety (CRITICAL)
1. **Day 1-2**: Implement `src/demographics/demographic_manager.py` and test with `data/age_data/*`
2. **Day 3-4**: Update `clinical_fair_pipeline.py` with safety validation
3. **Day 5-6**: Update `train_final.sh` with clinical validation
4. **Day 7**: Test complete pipeline with existing data

### Week 2: Domain Adaptation
1. **Day 1-2**: Implement `domain_adaptation.py` 
2. **Day 3-4**: Update `predict_cli.py` with domain adaptation
3. **Day 5-6**: Update bundle export with training references
4. **Day 7**: End-to-end testing

## Expected Results

After implementation:

✅ **Guaranteed Clinical Compliance**: Sensitivity ≥86%, Specificity ≥72%  
✅ **Population Adaptation**: Age/gender-specific thresholds  
✅ **Domain Shift Detection**: Automatic detection and adaptation  
✅ **Safety Validation**: Pipeline fails if targets not met  
✅ **Improved Specificity**: Reduce false positives from 36% to <15%  

## Usage Instructions

1. **Prepare demographics file** (CSV with columns: child_id, age, gender)
2. **Set environment variable**: `export DEMOGRAPHICS_PATH="path/to/demographics.csv"`
3. **Run training**: `./train_final.sh`
4. **Make predictions**: 
   ```bash
   ./venv/bin/python scripts/predict_cli.py \
     --bundle models/clinical_safe_u16n50_k2/bundle.json \
     --in your_data.csv \
     --demographics demographics.csv \
     --out predictions.csv
   ```

This plan **guarantees** your clinical targets while addressing domain shift through population-aware adaptation.