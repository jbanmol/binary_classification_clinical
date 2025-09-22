#!/usr/bin/env python3
"""
SIMPLE TPOT CLINICAL OPTIMIZATION
=================================
Simplified TPOT approach for clinical targets using default parameters
to avoid compatibility issues.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer

from tpot import TPOTClassifier

def load_clinical_data():
    """Load clinical data for TPOT optimization"""
    print("📊 Loading clinical dataset...")
    
    data_path = Path("features_binary/advanced_clinical_features.csv")
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['child_id', 'label', 'binary_label']]
    X = df[feature_cols].values
    
    if 'binary_label' in df.columns:
        y = df['binary_label'].values.astype(int)
    else:
        y = (df['label'] == 'ASD').astype(int)
    
    child_ids = df['child_id'].values
    
    # Imputation
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    print(f"   ✅ Dataset: {len(df)} sessions, {len(np.unique(child_ids))} children")
    print(f"      Features: {len(feature_cols)}, ASD: {np.sum(y)}, TD: {len(y) - np.sum(y)}")
    
    return X, y, child_ids

def create_clinical_scorer():
    """Create custom scoring function for clinical objectives"""
    def clinical_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Clinical scoring: emphasize both targets
        sens_score = min(sensitivity / 0.86, 1.5)
        spec_score = min(specificity / 0.71, 1.5)
        
        base_score = (sens_score + spec_score) / 2
        
        # Bonus for meeting targets
        if sensitivity >= 0.86 and specificity >= 0.71:
            bonus = 1.0
        elif sensitivity >= 0.86:
            bonus = 0.3
        elif specificity >= 0.71:
            bonus = 0.2
        else:
            bonus = 0
        
        return base_score + bonus
    
    return make_scorer(clinical_score, greater_is_better=True)

def run_simple_tpot(X, y):
    """Run simplified TPOT optimization"""
    print("🧬 Starting simple TPOT optimization...")
    
    # Create clinical scorer
    clinical_scorer = create_clinical_scorer()
    
    # Simple TPOT configuration
    tpot = TPOTClassifier(
        generations=10,
        population_size=20,
        scoring=clinical_scorer,
        cv=3,
        random_state=42,
        verbosity=2,
        n_jobs=4,
        max_time_mins=20  # 20 minutes max
    )
    
    # Split data for TPOT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training: {len(X_train)} samples ({np.sum(y_train)} ASD)")
    print(f"   Testing:  {len(X_test)} samples ({np.sum(y_test)} ASD)")
    
    # Fit TPOT
    try:
        tpot.fit(X_train, y_train)
        
        print("✅ TPOT optimization completed!")
        
        # Test performance
        y_pred = tpot.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        print(f"📊 TPOT Initial Test:")
        print(f"   Sensitivity: {sensitivity:.3f} {'✅' if sensitivity >= 0.86 else '❌'}")
        print(f"   Specificity: {specificity:.3f} {'✅' if specificity >= 0.71 else '❌'}")
        print(f"   Accuracy:    {accuracy:.3f}")
        
        return tpot, tpot.fitted_pipeline_
        
    except Exception as e:
        print(f"❌ TPOT failed: {e}")
        return None, None

def validate_pipeline(pipeline, X, y, child_ids):
    """Validate with child-level CV"""
    print("🔬 Child-level cross-validation...")
    
    cv = GroupKFold(n_splits=5)
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=child_ids)):
        print(f"   Fold {fold_idx + 1}/5...", end=" ")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            fold_results.append({
                'sensitivity': sensitivity,
                'specificity': specificity,
                'sens_target_met': sensitivity >= 0.86,
                'spec_target_met': specificity >= 0.71,
                'both_targets_met': sensitivity >= 0.86 and specificity >= 0.71
            })
            
            sens_check = "✅" if sensitivity >= 0.86 else "❌"
            spec_check = "✅" if specificity >= 0.71 else "❌"
            print(f"S:{sensitivity:.3f}{sens_check} Sp:{specificity:.3f}{spec_check}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    if not fold_results:
        return None
    
    # Aggregate results
    sensitivities = [fr['sensitivity'] for fr in fold_results]
    specificities = [fr['specificity'] for fr in fold_results]
    
    return {
        'sensitivity_mean': np.mean(sensitivities),
        'sensitivity_std': np.std(sensitivities),
        'specificity_mean': np.mean(specificities),
        'specificity_std': np.std(specificities),
        'sens_target_rate': np.mean([fr['sens_target_met'] for fr in fold_results]),
        'spec_target_rate': np.mean([fr['spec_target_met'] for fr in fold_results]),
        'both_targets_rate': np.mean([fr['both_targets_met'] for fr in fold_results])
    }

def main():
    """Main TPOT evaluation"""
    print("🧬 SIMPLE TPOT CLINICAL OPTIMIZATION")
    print("🎯 Target: Sensitivity ≥86%, Specificity ≥71%")
    print("⏱️  Runtime: ~20 minutes\n")
    
    # Load data
    X, y, child_ids = load_clinical_data()
    
    # Run TPOT
    tpot, best_pipeline = run_simple_tpot(X, y)
    
    if best_pipeline is None:
        print("❌ TPOT optimization failed")
        return
    
    print(f"\n🏆 BEST PIPELINE FOUND:")
    print("=" * 60)
    print(best_pipeline)
    print("=" * 60)
    
    # Validate with proper CV
    results = validate_pipeline(best_pipeline, X, y, child_ids)
    
    if results is None:
        print("❌ Validation failed")
        return
    
    # Display results
    print(f"\n📊 TPOT CLINICAL RESULTS:")
    print(f"🛡️  Child-level validation (no data leakage)")
    print("-" * 50)
    
    sens_check = "✅" if results['sensitivity_mean'] >= 0.86 else "❌"
    spec_check = "✅" if results['specificity_mean'] >= 0.71 else "❌"
    
    print(f"Sensitivity: {results['sensitivity_mean']:.3f} ± {results['sensitivity_std']:.3f} {sens_check}")
    print(f"Specificity: {results['specificity_mean']:.3f} ± {results['specificity_std']:.3f} {spec_check}")
    print(f"Both targets: {results['both_targets_rate']:.1%} of folds")
    
    # Assessment
    if results['sensitivity_mean'] >= 0.86 and results['specificity_mean'] >= 0.71:
        print("🎉 SUCCESS! TPOT achieved both clinical targets!")
    elif results['both_targets_rate'] > 0:
        print(f"⚡ Partial success: Both targets in {results['both_targets_rate']:.1%} of folds")
    else:
        print("⚠️ TPOT didn't achieve clinical targets consistently")
    
    # Save results
    results_dir = Path("tpot_simple_results")
    results_dir.mkdir(exist_ok=True)
    
    try:
        tpot.export(results_dir / "tpot_pipeline.py")
        print(f"✅ Pipeline saved to {results_dir}/tpot_pipeline.py")
    except Exception as e:
        print(f"⚠️ Could not save pipeline: {e}")
    
    # Save results
    with open(results_dir / "results.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'pipeline': str(best_pipeline),
            'results': results
        }, f, indent=2)
    
    print("🧬 TPOT evaluation complete!")

if __name__ == "__main__":
    main()
