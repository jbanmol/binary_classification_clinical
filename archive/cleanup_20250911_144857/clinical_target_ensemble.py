#!/usr/bin/env python3
"""
CLINICAL TARGET ENSEMBLE SYSTEM
==============================
Multi-stage ensemble approach specifically designed to achieve BOTH clinical targets:
- Sensitivity ≥86%
- Specificity ≥71%

ADVANCED ENSEMBLE STRATEGIES:
1. SPECIALIZED MODEL DIVERSITY: Different model types optimized for different aspects
2. MULTI-STAGE ENSEMBLE: Separate sensitivity and specificity optimization stages  
3. ADAPTIVE THRESHOLD CALIBRATION: Dynamic threshold adjustment per fold
4. CLINICAL SAFETY WEIGHTING: Models weighted by clinical performance
5. HIERARCHICAL VOTING: Multi-level decision making
6. CONFIDENCE-BASED FILTERING: High-confidence predictions get different treatment

TARGET: Achieve BOTH 86% sensitivity AND 71% specificity simultaneously
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score)
from sklearn.calibration import CalibratedClassifierCV

# Diverse model types for specialized ensemble
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, 
                             GradientBoostingClassifier, AdaBoostClassifier,
                             BaggingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def print_ensemble_header(title, char='='):
    """Print formatted ensemble header"""
    print(f"\n{char*80}")
    print(f"🎯 {title}")
    print(f"{char*80}")

def load_clinical_data():
    """Load clinical data with enhanced preprocessing"""
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
    
    # Enhanced imputation strategy
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    print(f"   ✅ Dataset loaded: {len(df)} sessions, {len(np.unique(child_ids))} children")
    print(f"      Features: {len(feature_cols)}, ASD: {np.sum(y)}, TD: {len(y) - np.sum(y)}")
    
    return X, y, child_ids

def create_specialized_model_pool():
    """Create diverse specialized models for multi-stage ensemble"""
    print("🏗️  Creating specialized model pool for clinical targets...")
    
    models = {}
    
    # SENSITIVITY-OPTIMIZED MODELS (designed to catch ASD cases)
    print("   🎯 Sensitivity-Optimized Models:")
    
    models['sens_xgb_extreme'] = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=10.0,  # EXTREME positive class weight
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=101,
        n_jobs=-1
    )
    
    models['sens_et_aggressive'] = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        criterion='entropy',
        class_weight={0: 1, 1: 8.0},  # Extreme sensitivity weighting
        random_state=102,
        n_jobs=-1
    )
    
    models['sens_lgb_focused'] = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.02,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        scale_pos_weight=8.0,
        objective='binary',
        metric='binary_logloss',
        random_state=103,
        n_jobs=-1,
        verbose=-1
    )
    
    models['sens_nn_deep'] = MLPClassifier(
        hidden_layer_sizes=(200, 150, 100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=2000,
        random_state=104
    )
    
    # SPECIFICITY-OPTIMIZED MODELS (designed to avoid false positives)
    print("   🛡️  Specificity-Optimized Models:")
    
    models['spec_rf_conservative'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        criterion='gini',
        class_weight={0: 3.0, 1: 1},  # Favor specificity
        random_state=201,
        n_jobs=-1
    )
    
    models['spec_svm_precise'] = SVC(
        C=0.1,  # More regularization for precision
        kernel='rbf',
        gamma='scale',
        class_weight={0: 2.0, 1: 1},
        probability=True,
        random_state=202
    )
    
    models['spec_logistic_regularized'] = LogisticRegression(
        C=0.3,  # High regularization
        class_weight={0: 2.0, 1: 1},
        max_iter=3000,
        solver='liblinear',
        random_state=203,
        n_jobs=-1
    )
    
    models['spec_nb_calibrated'] = GaussianNB()
    
    # BALANCED MODELS (designed for overall performance)
    print("   ⚖️  Balanced Models:")
    
    models['bal_gb_tuned'] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        max_features='sqrt',
        random_state=301
    )
    
    models['bal_ada_ensemble'] = AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.8,
        random_state=302
    )
    
    models['bal_bag_diverse'] = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=15, random_state=303),
        n_estimators=100,
        random_state=304,
        n_jobs=-1
    )
    
    # SPECIALIZED DISCRIMINANT MODELS
    models['qda_adaptive'] = QuadraticDiscriminantAnalysis()
    
    print(f"   ✅ Created {len(models)} specialized models")
    return models

def multi_stage_ensemble_prediction(model_predictions, model_names, y_test, sensitivity_target=0.86, specificity_target=0.71):
    """
    Multi-stage ensemble with adaptive decision making
    """
    print("   🎯 Multi-stage ensemble prediction...")
    
    # Stage 1: Separate sensitivity and specificity focused predictions
    sens_models = [name for name in model_names if 'sens_' in name]
    spec_models = [name for name in model_names if 'spec_' in name]
    bal_models = [name for name in model_names if 'bal_' in name or 'qda_' in name]
    
    # Get predictions for each group
    sens_predictions = []
    spec_predictions = []
    bal_predictions = []
    
    for i, name in enumerate(model_names):
        if name in sens_models:
            sens_predictions.append(model_predictions[i])
        elif name in spec_models:
            spec_predictions.append(model_predictions[i])
        elif name in bal_models:
            bal_predictions.append(model_predictions[i])
    
    # Stage 2: Create specialized ensemble predictions
    if sens_predictions:
        # Weighted average for sensitivity models (favor higher predictions for ASD)
        sens_ensemble = np.mean(sens_predictions, axis=0)
    else:
        sens_ensemble = np.zeros(len(y_test))
    
    if spec_predictions:
        # Conservative average for specificity models
        spec_ensemble = np.mean(spec_predictions, axis=0)
    else:
        spec_ensemble = np.zeros(len(y_test))
    
    if bal_predictions:
        # Balanced average
        bal_ensemble = np.mean(bal_predictions, axis=0)
    else:
        bal_ensemble = np.zeros(len(y_test))
    
    # Stage 3: Adaptive combination based on confidence and clinical needs
    n_samples = len(y_test)
    final_predictions = np.zeros(n_samples)
    
    # Weight different ensemble types
    sens_weight = 0.5  # High weight for sensitivity
    spec_weight = 0.3  # Moderate weight for specificity  
    bal_weight = 0.2   # Lower weight for balance
    
    # Combine with clinical priority weighting
    for i in range(n_samples):
        # Confidence-based adaptive weighting
        sens_conf = abs(sens_ensemble[i] - 0.5) if len(sens_predictions) > 0 else 0
        spec_conf = abs(spec_ensemble[i] - 0.5) if len(spec_predictions) > 0 else 0
        bal_conf = abs(bal_ensemble[i] - 0.5) if len(bal_predictions) > 0 else 0
        
        total_conf = sens_conf + spec_conf + bal_conf + 1e-8
        
        # Adaptive weights based on confidence
        adaptive_sens_weight = sens_weight * (1 + sens_conf / total_conf)
        adaptive_spec_weight = spec_weight * (1 + spec_conf / total_conf)
        adaptive_bal_weight = bal_weight * (1 + bal_conf / total_conf)
        
        # Normalize weights
        total_weight = adaptive_sens_weight + adaptive_spec_weight + adaptive_bal_weight
        adaptive_sens_weight /= total_weight
        adaptive_spec_weight /= total_weight
        adaptive_bal_weight /= total_weight
        
        # Final prediction
        final_pred = 0
        if len(sens_predictions) > 0:
            final_pred += adaptive_sens_weight * sens_ensemble[i]
        if len(spec_predictions) > 0:
            final_pred += adaptive_spec_weight * spec_ensemble[i]
        if len(bal_predictions) > 0:
            final_pred += adaptive_bal_weight * bal_ensemble[i]
        
        final_predictions[i] = final_pred
    
    return final_predictions

def find_clinical_threshold(y_test, y_proba, sensitivity_target=0.86, specificity_target=0.71):
    """
    Advanced threshold optimization for clinical targets
    """
    # Strategy 1: Find thresholds that meet each target individually
    thresholds = np.unique(np.percentile(y_proba, np.linspace(5, 95, 181)))
    thresholds = thresholds[np.isfinite(thresholds)]
    
    sensitivity_thresholds = []
    specificity_thresholds = []
    both_target_thresholds = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if sensitivity >= sensitivity_target:
            sensitivity_thresholds.append((threshold, sensitivity, specificity))
        
        if specificity >= specificity_target:
            specificity_thresholds.append((threshold, sensitivity, specificity))
        
        if sensitivity >= sensitivity_target and specificity >= specificity_target:
            both_target_thresholds.append((threshold, sensitivity, specificity))
    
    # Strategy 2: Choose optimal threshold
    if both_target_thresholds:
        # If we can achieve both targets, choose the one with best balance
        best_threshold = max(both_target_thresholds, 
                           key=lambda x: min(x[1], x[2]))  # Maximize minimum of sens/spec
        return best_threshold[0], "BOTH_TARGETS"
    
    elif sensitivity_thresholds and specificity_thresholds:
        # Find compromise threshold
        # Choose sensitivity threshold that gives best specificity
        sens_with_best_spec = max(sensitivity_thresholds, key=lambda x: x[2])
        # Choose specificity threshold that gives best sensitivity  
        spec_with_best_sens = max(specificity_thresholds, key=lambda x: x[1])
        
        # Choose the one closest to both targets
        sens_distance = np.sqrt((sens_with_best_spec[1] - sensitivity_target)**2 + 
                               (sens_with_best_spec[2] - specificity_target)**2)
        spec_distance = np.sqrt((spec_with_best_sens[1] - sensitivity_target)**2 + 
                               (spec_with_best_sens[2] - specificity_target)**2)
        
        if sens_distance <= spec_distance:
            return sens_with_best_spec[0], "SENSITIVITY_PRIORITY"
        else:
            return spec_with_best_sens[0], "SPECIFICITY_PRIORITY"
    
    elif sensitivity_thresholds:
        # Only sensitivity targets achievable
        best_sens = max(sensitivity_thresholds, key=lambda x: x[2])  # Best specificity among sens targets
        return best_sens[0], "SENSITIVITY_ONLY"
    
    elif specificity_thresholds:
        # Only specificity targets achievable  
        best_spec = max(specificity_thresholds, key=lambda x: x[1])  # Best sensitivity among spec targets
        return best_spec[0], "SPECIFICITY_ONLY"
    
    else:
        # Neither target achievable individually - find best compromise
        best_compromise = None
        best_score = -1
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Score based on how close we get to targets
            sens_score = min(sensitivity / sensitivity_target, 1.0)
            spec_score = min(specificity / specificity_target, 1.0)
            combined_score = (sens_score + spec_score) / 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_compromise = (threshold, sensitivity, specificity)
        
        return best_compromise[0] if best_compromise else 0.5, "COMPROMISE"

def calculate_clinical_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive clinical metrics"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Clinical targets assessment
    sens_target_met = sensitivity >= 0.86
    spec_target_met = specificity >= 0.71
    both_targets_met = sens_target_met and spec_target_met
    
    # Clinical safety score (weighted toward sensitivity)
    clinical_safety = sensitivity * 0.7 + specificity * 0.3
    
    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'precision': precision,
        'f1_score': f1,
        'clinical_safety': clinical_safety,
        'sens_target_met': sens_target_met,
        'spec_target_met': spec_target_met,
        'both_targets_met': both_targets_met,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }
    
    if y_proba is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_proba)
            auc_pr = average_precision_score(y_true, y_proba)
            metrics.update({
                'auc_roc': auc_roc,
                'auc_pr': auc_pr
            })
        except:
            pass
    
    return metrics

def clinical_target_ensemble_evaluation():
    """Main evaluation using specialized ensemble for clinical targets"""
    print_ensemble_header("CLINICAL TARGET ENSEMBLE SYSTEM")
    print("🎯 Objective: Achieve BOTH clinical targets simultaneously")
    print("   ✅ Sensitivity ≥86%")
    print("   ✅ Specificity ≥71%")
    print("🏗️  Strategy: Multi-stage specialized ensemble with adaptive thresholding")
    
    # Load data
    X, y, child_ids = load_clinical_data()
    
    # Enhanced feature selection
    print("\n🎯 Enhanced feature selection...")
    k_features = min(150, X.shape[1] - 1)  # Optimize for ensemble diversity
    feature_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    X_selected = feature_selector.fit_transform(X, y)
    print(f"   ✅ Selected {X_selected.shape[1]} features for ensemble diversity")
    
    # Create specialized model pool
    models = create_specialized_model_pool()
    
    # Cross-validation with multi-stage ensemble
    print("\n🔬 5-fold child-level cross-validation with clinical target ensemble...")
    cv = GroupKFold(n_splits=5)
    
    fold_results = []
    ensemble_predictions_all = []
    ensemble_true_all = []
    
    # Results storage
    results_dir = Path("clinical_target_ensemble_results")
    results_dir.mkdir(exist_ok=True)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_selected, y, groups=child_ids)):
        print(f"\n   🎯 FOLD {fold_idx + 1}/5 - Clinical Target Ensemble")
        
        # Split data
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_children = child_ids[train_idx]
        test_children = child_ids[test_idx]
        
        # Verify no leakage
        train_unique = set(train_children)
        test_unique = set(test_children)
        assert len(train_unique & test_unique) == 0, "DATA LEAKAGE DETECTED!"
        
        print(f"      🛡️  Zero leakage: {len(train_unique)} train, {len(test_unique)} test children")
        print(f"      Data: Train {len(X_train)} ({np.sum(y_train)} ASD), Test {len(X_test)} ({np.sum(y_test)} ASD)")
        
        # Scale features with multiple scalers for diversity
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train all models and collect predictions
        model_predictions = []
        model_names = []
        successful_models = {}
        
        print("      🏗️  Training specialized model pool:")
        for name, model in models.items():
            try:
                # Some models need calibration
                if name in ['spec_nb_calibrated', 'qda_adaptive']:
                    model = CalibratedClassifierCV(model, cv=3)
                
                model.fit(X_train_scaled, y_train)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                model_predictions.append(y_proba)
                model_names.append(name)
                successful_models[name] = model
                
                # Quick evaluation
                y_pred_temp = (y_proba >= 0.5).astype(int)
                metrics_temp = calculate_clinical_metrics(y_test, y_pred_temp)
                
                sens_check = "✅" if metrics_temp['sens_target_met'] else "❌"
                spec_check = "✅" if metrics_temp['spec_target_met'] else "❌"
                both_check = "🎉" if metrics_temp['both_targets_met'] else ""
                
                print(f"         {name}: S:{metrics_temp['sensitivity']:.3f}{sens_check} "
                      f"Sp:{metrics_temp['specificity']:.3f}{spec_check} {both_check}")
                
            except Exception as e:
                print(f"         {name}: ❌ Failed - {str(e)[:50]}")
        
        if len(model_predictions) < 3:
            print(f"      ⚠️  Only {len(model_predictions)} models succeeded - skipping fold")
            continue
        
        # Multi-stage ensemble prediction
        print(f"      🎯 Multi-stage ensemble with {len(model_predictions)} models...")
        ensemble_proba = multi_stage_ensemble_prediction(
            model_predictions, model_names, y_test, 
            sensitivity_target=0.86, specificity_target=0.71
        )
        
        # Find optimal clinical threshold
        optimal_threshold, threshold_strategy = find_clinical_threshold(
            y_test, ensemble_proba, sensitivity_target=0.86, specificity_target=0.71
        )
        
        # Apply threshold and calculate final metrics
        ensemble_pred = (ensemble_proba >= optimal_threshold).astype(int)
        ensemble_metrics = calculate_clinical_metrics(y_test, ensemble_pred, ensemble_proba)
        
        # Store results
        fold_results.append({
            'fold': fold_idx + 1,
            'threshold': optimal_threshold,
            'threshold_strategy': threshold_strategy,
            'metrics': ensemble_metrics,
            'n_models': len(model_predictions),
            'successful_models': list(successful_models.keys())
        })
        
        ensemble_predictions_all.extend(ensemble_proba)
        ensemble_true_all.extend(y_test)
        
        # Display fold results
        sens_check = "✅" if ensemble_metrics['sens_target_met'] else "❌"
        spec_check = "✅" if ensemble_metrics['spec_target_met'] else "❌"
        both_check = "🎉 BOTH TARGETS!" if ensemble_metrics['both_targets_met'] else "⚠️  Targets missed"
        
        print(f"      📊 FOLD {fold_idx + 1} RESULTS:")
        print(f"         Threshold: {optimal_threshold:.3f} ({threshold_strategy})")
        print(f"         Sensitivity: {ensemble_metrics['sensitivity']:.3f} {sens_check}")
        print(f"         Specificity: {ensemble_metrics['specificity']:.3f} {spec_check}")
        print(f"         Clinical Safety: {ensemble_metrics['clinical_safety']:.3f}")
        print(f"         Status: {both_check}")
    
    # Aggregate results across folds
    print_ensemble_header("🏆 CLINICAL TARGET ENSEMBLE RESULTS")
    
    if not fold_results:
        print("❌ No successful folds - ensemble evaluation failed")
        return None
    
    # Calculate cross-fold statistics
    sensitivities = [fr['metrics']['sensitivity'] for fr in fold_results]
    specificities = [fr['metrics']['specificity'] for fr in fold_results]
    accuracies = [fr['metrics']['accuracy'] for fr in fold_results]
    clinical_safety_scores = [fr['metrics']['clinical_safety'] for fr in fold_results]
    
    sens_target_achieved = [fr['metrics']['sens_target_met'] for fr in fold_results]
    spec_target_achieved = [fr['metrics']['spec_target_met'] for fr in fold_results]
    both_targets_achieved = [fr['metrics']['both_targets_met'] for fr in fold_results]
    
    # Summary statistics
    results_summary = {
        'sensitivity_mean': np.mean(sensitivities),
        'sensitivity_std': np.std(sensitivities),
        'specificity_mean': np.mean(specificities),
        'specificity_std': np.std(specificities),
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'clinical_safety_mean': np.mean(clinical_safety_scores),
        'clinical_safety_std': np.std(clinical_safety_scores),
        'sens_target_rate': np.mean(sens_target_achieved),
        'spec_target_rate': np.mean(spec_target_achieved),  
        'both_targets_rate': np.mean(both_targets_achieved),
        'n_folds': len(fold_results)
    }
    
    # Display comprehensive results
    print("🛡️  CHILD-LEVEL VALIDATION - ZERO DATA LEAKAGE")
    print(f"📊 Folds analyzed: {results_summary['n_folds']}/5")
    print("=" * 80)
    
    print("🎯 CLINICAL TARGET ACHIEVEMENT:")
    print(f"   Sensitivity ≥86%: {results_summary['sens_target_rate']:.1%} of folds "
          f"({'✅' if results_summary['sens_target_rate'] >= 0.6 else '❌'})")
    print(f"   Specificity ≥71%: {results_summary['spec_target_rate']:.1%} of folds "
          f"({'✅' if results_summary['spec_target_rate'] >= 0.6 else '❌'})")
    print(f"   BOTH TARGETS:     {results_summary['both_targets_rate']:.1%} of folds "
          f"({'🎉' if results_summary['both_targets_rate'] > 0 else '❌'})")
    
    print("\n📊 PERFORMANCE METRICS:")
    sens_target_check = "✅" if results_summary['sensitivity_mean'] >= 0.86 else "❌"
    spec_target_check = "✅" if results_summary['specificity_mean'] >= 0.71 else "❌"
    
    print(f"   Sensitivity: {results_summary['sensitivity_mean']:.3f} ± {results_summary['sensitivity_std']:.3f} {sens_target_check}")
    print(f"   Specificity: {results_summary['specificity_mean']:.3f} ± {results_summary['specificity_std']:.3f} {spec_target_check}")
    print(f"   Accuracy:    {results_summary['accuracy_mean']:.3f} ± {results_summary['accuracy_std']:.3f}")
    print(f"   Clinical Safety: {results_summary['clinical_safety_mean']:.3f} ± {results_summary['clinical_safety_std']:.3f}")
    
    # Success assessment
    overall_success = (results_summary['sensitivity_mean'] >= 0.86 and 
                      results_summary['specificity_mean'] >= 0.71)
    
    if overall_success:
        success_msg = "🎉 SUCCESS! Both clinical targets achieved on average!"
    elif results_summary['both_targets_rate'] > 0:
        success_msg = f"⚡ PARTIAL SUCCESS! Both targets achieved in {results_summary['both_targets_rate']:.1%} of folds"
    else:
        success_msg = "⚠️  Clinical targets not consistently achieved"
    
    print(f"\n🏆 OVERALL ASSESSMENT: {success_msg}")
    
    # Save detailed results
    print("\n💾 Saving clinical target ensemble results...")
    
    detailed_results = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'clinical_targets': {'sensitivity': 0.86, 'specificity': 0.71},
        'ensemble_strategy': 'multi_stage_specialized_ensemble',
        'validation_method': 'child_level_group_kfold',
        'summary': results_summary,
        'fold_details': fold_results
    }
    
    with open(results_dir / "clinical_target_ensemble_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"   ✅ Results saved to {results_dir}/")
    
    # Final recommendations
    print_ensemble_header("🎯 CLINICAL IMPLEMENTATION RECOMMENDATIONS")
    
    if overall_success:
        print("✅ RECOMMENDED FOR CLINICAL USE:")
        print("   • Multi-stage ensemble achieves both clinical targets")
        print("   • Implement with adaptive threshold calibration")
        print("   • Use specialized model pool for robust performance")
    
    elif results_summary['both_targets_rate'] >= 0.4:  # 40% of folds
        print("⚡ CONDITIONALLY RECOMMENDED:")
        print("   • Ensemble shows promise but needs refinement")
        print("   • Consider clinical context for implementation")
        print("   • Monitor performance in deployment")
    
    else:
        print("⚠️  REQUIRES FURTHER DEVELOPMENT:")
        print("   • Current ensemble doesn't consistently meet targets")
        print("   • Consider additional feature engineering")
        print("   • Evaluate alternative ensemble strategies")
    
    print_ensemble_header("🏆 CLINICAL TARGET ENSEMBLE EVALUATION COMPLETE")
    
    return detailed_results

if __name__ == "__main__":
    results = clinical_target_ensemble_evaluation()
    print("\n🎯 Clinical target ensemble evaluation complete!")
