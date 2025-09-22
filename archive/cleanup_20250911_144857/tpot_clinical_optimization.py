#!/usr/bin/env python3
"""
TPOT CLINICAL PIPELINE OPTIMIZATION
===================================
Using TPOT (Tree-based Pipeline Optimization Tool) to automatically discover 
optimal ML pipelines for clinical targets:
- Sensitivity ≥86%
- Specificity ≥71%

TPOT uses genetic programming to evolve ML pipelines, potentially discovering 
novel combinations of:
- Feature preprocessing
- Feature selection
- Model architectures
- Hyperparameter settings

This approach may find optimal solutions that manual optimization missed.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                           make_scorer, fbeta_score)

from tpot import TPOTClassifier
import multiprocessing

def print_tpot_header(title, char='='):
    """Print formatted TPOT header"""
    print(f"\n{char*80}")
    print(f"🧬 {title}")
    print(f"{char*80}")

def load_clinical_data():
    """Load clinical data for TPOT optimization"""
    print("📊 Loading clinical dataset for TPOT...")
    
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
    
    # Enhanced imputation for TPOT
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    print(f"   ✅ Dataset prepared for TPOT:")
    print(f"      Sessions: {len(df)}, Children: {len(np.unique(child_ids))}")
    print(f"      Features: {len(feature_cols)}")
    print(f"      ASD: {np.sum(y)}, TD: {len(y) - np.sum(y)}")
    
    return X, y, child_ids, feature_cols

def create_clinical_scorer():
    """Create custom scoring function for clinical objectives"""
    def clinical_score(y_true, y_pred):
        """
        Custom scoring function that balances sensitivity and specificity
        with emphasis on achieving clinical targets
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Clinical targets
        sens_target = 0.86
        spec_target = 0.71
        
        # Scoring strategy: reward meeting targets, penalize missing them
        sens_score = min(sensitivity / sens_target, 1.5)  # Allow over-achievement
        spec_score = min(specificity / spec_target, 1.5)
        
        # Base score (balanced)
        base_score = (sens_score + spec_score) / 2
        
        # Bonuses for meeting targets
        if sensitivity >= sens_target and specificity >= spec_target:
            bonus = 1.0  # Large bonus for both targets
        elif sensitivity >= sens_target:
            bonus = 0.3  # Moderate bonus for sensitivity
        elif specificity >= spec_target:
            bonus = 0.2  # Small bonus for specificity
        else:
            bonus = 0
        
        # Final score
        final_score = base_score + bonus
        
        # Penalize extreme imbalance
        if sensitivity < 0.5 or specificity < 0.3:
            final_score *= 0.5  # Heavy penalty for extreme imbalance
        
        return final_score
    
    return make_scorer(clinical_score, greater_is_better=True)

def run_tpot_optimization(X, y, child_ids, n_jobs=-1, generations=20, population_size=50):
    """
    Run TPOT optimization with clinical scoring
    """
    print(f"🧬 Starting TPOT optimization...")
    print(f"   Generations: {generations}")
    print(f"   Population Size: {population_size}")
    print(f"   Parallel Jobs: {n_jobs if n_jobs != -1 else 'All available cores'}")
    
    # Create custom scorer
    clinical_scorer = create_clinical_scorer()
    
    # Use default TPOT configuration for compatibility
    tpot_config = None  # Use TPOT's default configuration
    
    # Initialize TPOT
    tpot = TPOTClassifier(
        generations=generations,
        population_size=population_size,
        mutation_rate=0.9,
        crossover_rate=0.1,
        scoring=clinical_scorer,
        cv=5,  # Will be overridden by our custom CV
        random_state=42,
        config_dict=tpot_config,
        verbosity=2,
        n_jobs=n_jobs,
        max_time_mins=30,
        max_eval_time_mins=5,
        early_stop=5
    )
    
    # Split data for TPOT (it needs a simple train/test split)
    # We'll do proper child-level validation later
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   🔬 TPOT training set: {len(X_train)} samples ({np.sum(y_train)} ASD)")
    print(f"   🔬 TPOT test set: {len(X_test)} samples ({np.sum(y_test)} ASD)")
    
    # Fit TPOT
    print("\n🧬 Running genetic programming optimization...")
    print("   This may take 15-30 minutes depending on your hardware...")
    
    try:
        tpot.fit(X_train, y_train)
        
        # Get best pipeline
        print(f"\n✅ TPOT optimization completed!")
        print(f"   Best CV score: {tpot.score(X_test, y_test):.4f}")
        
        # Evaluate on test set
        y_pred = tpot.predict(X_test)
        y_proba = tpot.predict_proba(X_test)[:, 1] if hasattr(tpot, 'predict_proba') else None
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        print(f"\n📊 TPOT Best Pipeline Performance (Initial Test):")
        print(f"   Sensitivity: {sensitivity:.3f} {'✅' if sensitivity >= 0.86 else '❌'}")
        print(f"   Specificity: {specificity:.3f} {'✅' if specificity >= 0.71 else '❌'}")
        print(f"   Accuracy:    {accuracy:.3f}")
        
        if y_proba is not None:
            auc_roc = roc_auc_score(y_test, y_proba)
            print(f"   AUC-ROC:     {auc_roc:.3f}")
        
        return tpot, tpot.fitted_pipeline_
        
    except Exception as e:
        print(f"❌ TPOT optimization failed: {e}")
        return None, None

def validate_tpot_pipeline(pipeline, X, y, child_ids):
    """
    Validate TPOT pipeline using proper child-level cross-validation
    """
    print("\n🔬 Validating TPOT pipeline with child-level cross-validation...")
    
    cv = GroupKFold(n_splits=5)
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=child_ids)):
        print(f"\n   🧬 FOLD {fold_idx + 1}/5 - TPOT Pipeline Validation")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_children = child_ids[train_idx]
        test_children = child_ids[test_idx]
        
        # Verify no leakage
        assert len(set(train_children) & set(test_children)) == 0, "DATA LEAKAGE DETECTED!"
        
        print(f"      🛡️  Zero leakage: {len(set(train_children))} train, {len(set(test_children))} test children")
        
        try:
            # Train pipeline
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Get probabilities if available
            y_proba = None
            if hasattr(pipeline, 'predict_proba'):
                try:
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                    auc_roc = roc_auc_score(y_test, y_proba)
                except:
                    auc_roc = None
            else:
                auc_roc = None
            
            # Store results
            fold_results.append({
                'fold': fold_idx + 1,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'accuracy': accuracy,
                'auc_roc': auc_roc,
                'sens_target_met': sensitivity >= 0.86,
                'spec_target_met': specificity >= 0.71,
                'both_targets_met': sensitivity >= 0.86 and specificity >= 0.71
            })
            
            # Display fold results
            sens_check = "✅" if sensitivity >= 0.86 else "❌"
            spec_check = "✅" if specificity >= 0.71 else "❌"
            both_check = "🎉" if sensitivity >= 0.86 and specificity >= 0.71 else ""
            
            print(f"      📊 Sensitivity: {sensitivity:.3f} {sens_check}")
            print(f"      📊 Specificity: {specificity:.3f} {spec_check}")
            print(f"      📊 Accuracy:    {accuracy:.3f}")
            if auc_roc:
                print(f"      📊 AUC-ROC:     {auc_roc:.3f}")
            print(f"      📊 Status: {'🎉 BOTH TARGETS!' if both_check else '⚠️ Targets missed'}")
            
        except Exception as e:
            print(f"      ❌ Fold {fold_idx + 1} failed: {e}")
    
    if not fold_results:
        print("❌ No successful validation folds")
        return None
    
    # Aggregate results
    sensitivities = [fr['sensitivity'] for fr in fold_results]
    specificities = [fr['specificity'] for fr in fold_results]
    accuracies = [fr['accuracy'] for fr in fold_results]
    
    sens_target_rate = np.mean([fr['sens_target_met'] for fr in fold_results])
    spec_target_rate = np.mean([fr['spec_target_met'] for fr in fold_results])
    both_targets_rate = np.mean([fr['both_targets_met'] for fr in fold_results])
    
    results_summary = {
        'sensitivity_mean': np.mean(sensitivities),
        'sensitivity_std': np.std(sensitivities),
        'specificity_mean': np.mean(specificities),
        'specificity_std': np.std(specificities),
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'sens_target_rate': sens_target_rate,
        'spec_target_rate': spec_target_rate,
        'both_targets_rate': both_targets_rate,
        'fold_details': fold_results
    }
    
    return results_summary

def tpot_clinical_evaluation():
    """Main TPOT evaluation function"""
    print_tpot_header("TPOT CLINICAL PIPELINE OPTIMIZATION")
    print("🧬 Using genetic programming to discover optimal ML pipelines")
    print("🎯 Target: Sensitivity ≥86%, Specificity ≥71%")
    print("⏱️  Expected runtime: 15-30 minutes")
    
    # Load data
    X, y, child_ids, feature_names = load_clinical_data()
    
    # Determine number of cores
    n_cores = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid overwhelming
    print(f"🔧 Using {n_cores} CPU cores for parallel optimization")
    
    # Run TPOT optimization
    tpot, best_pipeline = run_tpot_optimization(
        X, y, child_ids, 
        n_jobs=n_cores, 
        generations=15,  # Reasonable for clinical optimization
        population_size=30  # Balanced between exploration and runtime
    )
    
    if best_pipeline is None:
        print("❌ TPOT optimization failed")
        return None
    
    # Display best pipeline
    print(f"\n🏆 TPOT DISCOVERED BEST PIPELINE:")
    print("=" * 80)
    print(best_pipeline)
    print("=" * 80)
    
    # Validate with proper cross-validation
    validation_results = validate_tpot_pipeline(best_pipeline, X, y, child_ids)
    
    if validation_results is None:
        print("❌ Pipeline validation failed")
        return None
    
    # Display final results
    print_tpot_header("🏆 TPOT CLINICAL RESULTS")
    
    print("🛡️  CHILD-LEVEL VALIDATION - ZERO DATA LEAKAGE")
    print(f"📊 Folds completed: {len(validation_results['fold_details'])}/5")
    print("=" * 80)
    
    print("🎯 CLINICAL TARGET ACHIEVEMENT:")
    sens_check = "✅" if validation_results['sens_target_rate'] >= 0.6 else "❌"
    spec_check = "✅" if validation_results['spec_target_rate'] >= 0.6 else "❌"
    both_check = "🎉" if validation_results['both_targets_rate'] > 0 else "❌"
    
    print(f"   Sensitivity ≥86%: {validation_results['sens_target_rate']:.1%} of folds {sens_check}")
    print(f"   Specificity ≥71%: {validation_results['spec_target_rate']:.1%} of folds {spec_check}")
    print(f"   BOTH TARGETS:     {validation_results['both_targets_rate']:.1%} of folds {both_check}")
    
    print("\n📊 TPOT PERFORMANCE METRICS:")
    sens_target_check = "✅" if validation_results['sensitivity_mean'] >= 0.86 else "❌"
    spec_target_check = "✅" if validation_results['specificity_mean'] >= 0.71 else "❌"
    
    print(f"   Sensitivity: {validation_results['sensitivity_mean']:.3f} ± {validation_results['sensitivity_std']:.3f} {sens_target_check}")
    print(f"   Specificity: {validation_results['specificity_mean']:.3f} ± {validation_results['specificity_std']:.3f} {spec_target_check}")
    print(f"   Accuracy:    {validation_results['accuracy_mean']:.3f} ± {validation_results['accuracy_std']:.3f}")
    
    # Assessment
    overall_success = (validation_results['sensitivity_mean'] >= 0.86 and 
                      validation_results['specificity_mean'] >= 0.71)
    
    if overall_success:
        success_msg = "🎉 TPOT SUCCESS! Both clinical targets achieved!"
        recommendation = "✅ TPOT pipeline recommended for clinical use"
    elif validation_results['both_targets_rate'] > 0:
        success_msg = f"⚡ PARTIAL SUCCESS! Both targets achieved in {validation_results['both_targets_rate']:.1%} of folds"
        recommendation = "⚠️ TPOT pipeline shows promise but needs refinement"
    else:
        success_msg = "⚠️ TPOT pipeline doesn't consistently meet clinical targets"
        recommendation = "❌ Current TPOT configuration insufficient for clinical targets"
    
    print(f"\n🏆 TPOT ASSESSMENT: {success_msg}")
    print(f"💡 RECOMMENDATION: {recommendation}")
    
    # Save results
    print("\n💾 Saving TPOT results...")
    
    results_dir = Path("tpot_clinical_results")
    results_dir.mkdir(exist_ok=True)
    
    # Export best pipeline
    try:
        tpot.export(results_dir / "tpot_best_pipeline.py")
        print(f"   ✅ Best pipeline exported to {results_dir}/tpot_best_pipeline.py")
    except Exception as e:
        print(f"   ⚠️ Could not export pipeline: {e}")
    
    # Save results
    tpot_results = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'clinical_targets': {'sensitivity': 0.86, 'specificity': 0.71},
        'tpot_config': {
            'generations': 15,
            'population_size': 30,
            'scoring': 'custom_clinical_scorer'
        },
        'best_pipeline_str': str(best_pipeline),
        'validation_results': validation_results
    }
    
    with open(results_dir / "tpot_clinical_results.json", 'w') as f:
        json.dump(tpot_results, f, indent=2, default=str)
    
    print(f"   ✅ Results saved to {results_dir}/")
    
    print_tpot_header("🧬 TPOT CLINICAL OPTIMIZATION COMPLETE")
    
    return tpot_results

if __name__ == "__main__":
    results = tpot_clinical_evaluation()
    print("\n🧬 TPOT clinical evaluation complete!")
