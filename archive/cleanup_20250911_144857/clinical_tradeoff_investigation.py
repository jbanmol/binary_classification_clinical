#!/usr/bin/env python3
"""
CLINICAL TRADE-OFF INVESTIGATION & ADVANCED FEATURE ENGINEERING
===============================================================
Investigating strategies to overcome the sensitivity-specificity trade-off:

1. ADVANCED FEATURE ENGINEERING:
   - Clinical interaction features
   - Domain-specific composite features  
   - Statistical moment features
   - Temporal pattern features
   - Clinical risk factors

2. REALISTIC CLINICAL THRESHOLD ANALYSIS:
   - ROC curve analysis across all folds
   - Pareto frontier for sensitivity-specificity trade-offs
   - Clinical risk assessment at different thresholds
   - Cost-benefit analysis for different clinical scenarios

3. ENHANCED MODEL ARCHITECTURES:
   - Multi-objective optimization
   - Calibrated probability thresholds
   - Clinical decision support integration

TARGET: Find optimal balance or recommend realistic clinical targets
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score, 
                           roc_curve, precision_recall_curve)

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def print_clinical_header(title, char='='):
    """Print formatted clinical header"""
    print(f"\n{char*80}")
    print(f"🏥 {title}")
    print(f"{char*80}")

def load_and_engineer_clinical_features():
    """Load data and create advanced clinical features"""
    print("📊 Loading base clinical dataset...")
    
    data_path = Path("features_binary/advanced_clinical_features.csv")
    df = pd.read_csv(data_path)
    
    # Prepare basic features
    feature_cols = [col for col in df.columns if col not in ['child_id', 'label', 'binary_label']]
    X_base = df[feature_cols].values
    
    if 'binary_label' in df.columns:
        y = df['binary_label'].values.astype(int)
    else:
        y = (df['label'] == 'ASD').astype(int)
    
    child_ids = df['child_id'].values
    
    print(f"   ✅ Base features loaded: {X_base.shape[1]} features")
    print(f"      Sessions: {len(df)}, Children: {len(np.unique(child_ids))}")
    print(f"      ASD: {np.sum(y)}, TD: {len(y) - np.sum(y)}")
    
    # Advanced Feature Engineering Pipeline
    print("\n🔬 ADVANCED CLINICAL FEATURE ENGINEERING...")
    
    # Handle missing values first
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_base)
    
    # Convert back to DataFrame for feature engineering
    df_features = pd.DataFrame(X_imputed, columns=feature_cols)
    
    print("   🧬 Creating clinical interaction features...")
    # Clinical domain interactions (based on ASD screening knowledge)
    clinical_interactions = []
    
    # Social vs Communication interactions
    social_cols = [col for col in feature_cols if 'social' in col.lower()]
    comm_cols = [col for col in feature_cols if any(word in col.lower() for word in ['comm', 'speech', 'lang'])]
    
    if len(social_cols) >= 2 and len(comm_cols) >= 2:
        for i, social_col in enumerate(social_cols[:3]):  # Limit to avoid explosion
            for j, comm_col in enumerate(comm_cols[:3]):
                interaction = df_features[social_col] * df_features[comm_col]
                clinical_interactions.append(interaction.values)
                print(f"      ➕ Social-Communication interaction {i+1}-{j+1}")
    
    # Behavioral pattern interactions
    behav_cols = [col for col in feature_cols if any(word in col.lower() for word in ['behav', 'repet', 'restrict'])]
    if len(behav_cols) >= 2:
        for i in range(min(3, len(behav_cols))):
            for j in range(i+1, min(3, len(behav_cols))):
                interaction = df_features[behav_cols[i]] * df_features[behav_cols[j]]
                clinical_interactions.append(interaction.values)
                print(f"      ➕ Behavioral interaction {i+1}-{j+1}")
    
    print("   📊 Creating statistical moment features...")
    # Statistical moments for clinical patterns
    statistical_features = []
    
    # Calculate moments across feature subsets
    n_features = X_imputed.shape[1]
    for start_idx in range(0, n_features, max(1, n_features // 8)):  # 8 subsets
        end_idx = min(start_idx + max(1, n_features // 8), n_features)
        subset = X_imputed[:, start_idx:end_idx]
        
        if subset.shape[1] > 1:
            # Statistical moments
            subset_mean = np.mean(subset, axis=1)
            subset_std = np.std(subset, axis=1)
            subset_skew = stats.skew(subset, axis=1)
            subset_kurt = stats.kurtosis(subset, axis=1)
            
            statistical_features.extend([subset_mean, subset_std, subset_skew, subset_kurt])
    
    print(f"      ✅ Created {len(statistical_features)} statistical moment features")
    
    print("   🏥 Creating clinical composite scores...")
    # Clinical composite scores
    composite_features = []
    
    # Overall severity composite (weighted sum of key features)
    if n_features >= 5:
        # Use top features by variance as proxy for clinical importance
        feature_vars = np.var(X_imputed, axis=0)
        top_indices = np.argsort(feature_vars)[-5:]  # Top 5 most variable features
        
        severity_score = np.mean(X_imputed[:, top_indices], axis=1)
        composite_features.append(severity_score)
        
        # Clinical risk stratification
        risk_high = (severity_score > np.percentile(severity_score, 75)).astype(float)
        risk_low = (severity_score < np.percentile(severity_score, 25)).astype(float)
        composite_features.extend([risk_high, risk_low])
    
    print(f"      ✅ Created {len(composite_features)} clinical composite features")
    
    print("   🔄 Creating temporal/pattern features...")
    # Pattern detection features
    pattern_features = []
    
    # Feature stability patterns (if we can infer temporal relationships)
    for i in range(0, min(10, n_features)):  # Limit to first 10 features
        feature_vals = X_imputed[:, i]
        
        # Outlier patterns
        q75, q25 = np.percentile(feature_vals, [75, 25])
        iqr = q75 - q25
        outlier_pattern = ((feature_vals < (q25 - 1.5 * iqr)) | 
                          (feature_vals > (q75 + 1.5 * iqr))).astype(float)
        pattern_features.append(outlier_pattern)
    
    print(f"      ✅ Created {len(pattern_features)} pattern features")
    
    print("   ⚡ Creating polynomial interaction features...")
    # Limited polynomial features (degree 2, selected features)
    if n_features <= 20:  # Only for manageable feature sets
        # Select top 10 features by variance for polynomial expansion
        feature_vars = np.var(X_imputed, axis=0)
        top_indices = np.argsort(feature_vars)[-10:]
        X_selected = X_imputed[:, top_indices]
        
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_selected)
        
        # Take only the interaction terms (not the original features)
        n_original = X_selected.shape[1]
        polynomial_features = X_poly[:, n_original:].T  # Transpose for consistency
        print(f"      ✅ Created {len(polynomial_features)} polynomial interaction features")
    else:
        polynomial_features = []
        print(f"      ⚠️  Skipping polynomial features (too many base features)")
    
    # Combine all engineered features
    print("\n🏗️  Combining all engineered features...")
    
    all_engineered = []
    all_engineered.extend(clinical_interactions)
    all_engineered.extend(statistical_features)
    all_engineered.extend(composite_features)
    all_engineered.extend(pattern_features)
    all_engineered.extend(polynomial_features)
    
    if all_engineered:
        X_engineered = np.column_stack(all_engineered)
        print(f"   ✅ Created {X_engineered.shape[1]} engineered features")
        
        # Combine with original features
        X_combined = np.column_stack([X_imputed, X_engineered])
        print(f"   🏆 Final feature set: {X_combined.shape[1]} features ({X_imputed.shape[1]} original + {X_engineered.shape[1]} engineered)")
    else:
        X_combined = X_imputed
        print(f"   ⚠️  No engineered features created, using {X_combined.shape[1]} original features")
    
    return X_combined, y, child_ids

def analyze_clinical_thresholds(y_true_list, y_proba_list, fold_names):
    """Comprehensive clinical threshold analysis across all folds"""
    print("\n🎯 COMPREHENSIVE CLINICAL THRESHOLD ANALYSIS")
    print("=" * 70)
    
    # Combine all fold data for global analysis
    y_true_combined = np.concatenate(y_true_list)
    y_proba_combined = np.concatenate(y_proba_list)
    
    print(f"📊 Combined dataset: {len(y_true_combined)} samples ({np.sum(y_true_combined)} ASD)")
    
    # ROC curve analysis
    fpr, tpr, thresholds = roc_curve(y_true_combined, y_proba_combined)
    roc_auc = roc_auc_score(y_true_combined, y_proba_combined)
    
    print(f"📈 ROC AUC: {roc_auc:.3f}")
    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true_combined, y_proba_combined)
    pr_auc = average_precision_score(y_true_combined, y_proba_combined)
    
    print(f"📈 PR AUC: {pr_auc:.3f}")
    
    # Clinical threshold analysis
    print("\n🏥 CLINICAL THRESHOLD SCENARIOS:")
    print("-" * 70)
    
    clinical_scenarios = []
    
    # Evaluate different threshold strategies
    threshold_strategies = [
        ("Conservative (High Specificity)", 0.7),
        ("Balanced (Youden Index)", None),  # Will calculate optimal
        ("Aggressive (High Sensitivity)", 0.3),
        ("Ultra-Aggressive (Max Sensitivity)", 0.2),
        ("Current Clinical Standard", 0.5),
        ("Your V4.2 Proven", 0.568)
    ]
    
    # Find Youden Index optimal threshold
    youden_scores = tpr - fpr
    youden_optimal_idx = np.argmax(youden_scores)
    youden_optimal_threshold = thresholds[youden_optimal_idx]
    
    for strategy_name, threshold in threshold_strategies:
        if threshold is None:  # Youden Index
            threshold = youden_optimal_threshold
            
        # Find closest threshold in our ROC curve
        closest_idx = np.argmin(np.abs(thresholds - threshold))
        actual_threshold = thresholds[closest_idx]
        sensitivity = tpr[closest_idx]
        specificity = 1 - fpr[closest_idx]
        
        # Calculate metrics at this threshold
        y_pred_threshold = (y_proba_combined >= actual_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_combined, y_pred_threshold).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Clinical safety metrics
        clinical_safety = sensitivity * 0.7 + specificity * 0.3  # Sensitivity-weighted
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Clinical assessments
        sens_target = sensitivity >= 0.86
        spec_target = specificity >= 0.71
        both_targets = sens_target and spec_target
        
        clinical_viability = "🎉 EXCELLENT" if both_targets else \
                           "✅ GOOD" if (sensitivity >= 0.80 and specificity >= 0.65) else \
                           "⚠️  MARGINAL" if (sensitivity >= 0.75 and specificity >= 0.60) else \
                           "❌ POOR"
        
        scenario_result = {
            'strategy': strategy_name,
            'threshold': actual_threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'precision': precision_at_threshold,
            'clinical_safety': clinical_safety,
            'balanced_accuracy': balanced_accuracy,
            'sens_target_met': sens_target,
            'spec_target_met': spec_target,
            'both_targets_met': both_targets,
            'clinical_viability': clinical_viability,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
        
        clinical_scenarios.append(scenario_result)
        
        print(f"\n🎯 {strategy_name}:")
        print(f"   Threshold:   {actual_threshold:.3f}")
        print(f"   Sensitivity: {sensitivity:.3f} {'✅' if sens_target else '❌'} (target: ≥0.86)")
        print(f"   Specificity: {specificity:.3f} {'✅' if spec_target else '❌'} (target: ≥0.71)")
        print(f"   Accuracy:    {accuracy:.3f}")
        print(f"   Precision:   {precision_at_threshold:.3f}")
        print(f"   Clinical Safety: {clinical_safety:.3f}")
        print(f"   Viability:   {clinical_viability}")
        print(f"   Confusion:   TP:{tp}, FP:{fp}, TN:{tn}, FN:{fn}")
    
    # Find Pareto frontier for sensitivity-specificity trade-off
    print("\n📊 PARETO FRONTIER ANALYSIS:")
    print("-" * 70)
    
    pareto_points = []
    for i, threshold in enumerate(thresholds[::10]):  # Sample every 10th threshold
        sens = tpr[i*10] if i*10 < len(tpr) else tpr[-1]
        spec = 1 - fpr[i*10] if i*10 < len(fpr) else 1 - fpr[-1]
        
        # Check if this point is Pareto optimal
        is_pareto = True
        for j, other_threshold in enumerate(thresholds[::10]):
            if j == i:
                continue
            other_sens = tpr[j*10] if j*10 < len(tpr) else tpr[-1] 
            other_spec = 1 - fpr[j*10] if j*10 < len(fpr) else 1 - fpr[-1]
            
            # If another point dominates this one, it's not Pareto optimal
            if other_sens >= sens and other_spec >= spec and (other_sens > sens or other_spec > spec):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_points.append({
                'threshold': threshold,
                'sensitivity': sens,
                'specificity': spec,
                'distance_to_target': np.sqrt((sens - 0.86)**2 + (spec - 0.71)**2)
            })
    
    # Sort by distance to clinical targets
    pareto_points.sort(key=lambda x: x['distance_to_target'])
    
    print("🏆 TOP PARETO-OPTIMAL POINTS (closest to clinical targets):")
    for i, point in enumerate(pareto_points[:5]):
        sens_check = "✅" if point['sensitivity'] >= 0.86 else "❌"
        spec_check = "✅" if point['specificity'] >= 0.71 else "❌"
        print(f"   {i+1}. Threshold: {point['threshold']:.3f} → "
              f"Sens: {point['sensitivity']:.3f}{sens_check}, "
              f"Spec: {point['specificity']:.3f}{spec_check} "
              f"(Distance: {point['distance_to_target']:.3f})")
    
    # Clinical recommendation analysis
    print("\n💡 CLINICAL RECOMMENDATIONS:")
    print("-" * 70)
    
    # Find best balanced approach
    best_balanced = min(clinical_scenarios, 
                       key=lambda x: abs(x['sensitivity'] - x['specificity']))
    
    # Find closest to clinical targets
    best_clinical = min(clinical_scenarios,
                       key=lambda x: np.sqrt((x['sensitivity'] - 0.86)**2 + (x['specificity'] - 0.71)**2))
    
    # Find best clinical safety score
    best_safety = max(clinical_scenarios, key=lambda x: x['clinical_safety'])
    
    print(f"🔄 MOST BALANCED: {best_balanced['strategy']}")
    print(f"   Sens: {best_balanced['sensitivity']:.3f}, Spec: {best_balanced['specificity']:.3f}")
    print(f"   Threshold: {best_balanced['threshold']:.3f}")
    
    print(f"\n🎯 CLOSEST TO CLINICAL TARGETS: {best_clinical['strategy']}")
    print(f"   Sens: {best_clinical['sensitivity']:.3f}, Spec: {best_clinical['specificity']:.3f}")
    print(f"   Threshold: {best_clinical['threshold']:.3f}")
    print(f"   Distance to targets: {np.sqrt((best_clinical['sensitivity'] - 0.86)**2 + (best_clinical['specificity'] - 0.71)**2):.3f}")
    
    print(f"\n🛡️  BEST CLINICAL SAFETY: {best_safety['strategy']}")
    print(f"   Sens: {best_safety['sensitivity']:.3f}, Spec: {best_safety['specificity']:.3f}")
    print(f"   Safety Score: {best_safety['clinical_safety']:.3f}")
    
    return clinical_scenarios, pareto_points

def advanced_model_with_engineered_features():
    """Train models with advanced engineered features and analyze trade-offs"""
    print_clinical_header("ADVANCED FEATURE ENGINEERING & CLINICAL THRESHOLD ANALYSIS")
    print("🎯 Investigating strategies to improve sensitivity-specificity trade-off")
    print("🔬 Advanced feature engineering + Comprehensive threshold analysis")
    
    # Load and engineer features
    X, y, child_ids = load_and_engineer_clinical_features()
    
    print(f"\n📊 Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Unique children: {len(np.unique(child_ids))}")
    print(f"   ASD: {np.sum(y)}, TD: {len(y) - np.sum(y)}")
    
    # Feature selection on the enhanced feature set
    print("\n🎯 Advanced feature selection...")
    
    # Use your proven k=200 approach, but adapt to feature count
    k_features = min(200, X.shape[1] - 1)
    feature_selector = SelectKBest(score_func=f_classif, k=k_features)
    X_selected = feature_selector.fit_transform(X, y)
    
    print(f"   ✅ Selected {X_selected.shape[1]} features from {X.shape[1]} engineered features")
    
    # Create enhanced models (your proven V4.2 + best performers)
    models = {
        'enhanced_v42_rf': RandomForestClassifier(
            n_estimators=300,  # More trees for complex features
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            criterion='gini',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        'enhanced_v42_et': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            criterion='entropy',
            class_weight={0: 1, 1: 1.5},
            random_state=43,
            n_jobs=-1
        ),
        
        'enhanced_xgboost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.8,
            scale_pos_weight=1.3,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=44,
            n_jobs=-1
        ),
        
        'enhanced_logistic': LogisticRegression(
            C=0.5,  # Slightly more regularization for complex features
            class_weight={0: 1, 1: 1.5},
            max_iter=3000,
            solver='liblinear',
            random_state=45,
            n_jobs=-1
        )
    }
    
    print(f"🏆 Created {len(models)} enhanced models for advanced feature evaluation")
    
    # Cross-validation with enhanced features
    print("\n🔬 Enhanced 5-fold child-level cross-validation...")
    cv = GroupKFold(n_splits=5)
    
    fold_results = []
    all_y_true = []
    all_y_proba = {}  # Store probabilities for each model
    
    for model_name in models.keys():
        all_y_proba[model_name] = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_selected, y, groups=child_ids)):
        print(f"\n   🔬 FOLD {fold_idx + 1}/5 - Enhanced Features")
        
        # Split data
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_children = child_ids[train_idx]
        test_children = child_ids[test_idx]
        
        # Verify no leakage
        train_unique = set(train_children)
        test_unique = set(test_children)
        assert len(train_unique & test_unique) == 0, "DATA LEAKAGE DETECTED!"
        
        print(f"      🛡️  Zero leakage - {len(train_unique)} train, {len(test_unique)} test children")
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        fold_model_results = {}
        
        # Train enhanced models
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                all_y_proba[name].append(y_proba)
                
                # Quick evaluation at standard threshold
                y_pred = (y_proba >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                fold_model_results[name] = {
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'y_proba': y_proba
                }
                
                sens_check = "✅" if sensitivity >= 0.86 else "❌"
                spec_check = "✅" if specificity >= 0.71 else "❌"
                print(f"         {name}: S:{sensitivity:.3f}{sens_check} Sp:{specificity:.3f}{spec_check}")
                
            except Exception as e:
                print(f"         {name}: ❌ Failed: {e}")
        
        all_y_true.append(y_test)
        fold_results.append(fold_model_results)
    
    # Comprehensive threshold analysis for each model
    print_clinical_header("COMPREHENSIVE CLINICAL THRESHOLD ANALYSIS")
    
    results_dir = Path("clinical_tradeoff_analysis")
    results_dir.mkdir(exist_ok=True)
    
    all_analyses = {}
    
    for model_name in models.keys():
        if all(model_name in fold_result for fold_result in fold_results):
            print(f"\n🎯 ANALYZING {model_name.upper()}...")
            
            model_y_proba = all_y_proba[model_name]
            fold_names = [f"Fold_{i+1}" for i in range(len(model_y_proba))]
            
            # Perform threshold analysis
            scenarios, pareto_points = analyze_clinical_thresholds(all_y_true, model_y_proba, fold_names)
            
            all_analyses[model_name] = {
                'scenarios': scenarios,
                'pareto_points': pareto_points[:10]  # Top 10 Pareto points
            }
    
    # Create ensemble with advanced features
    print_clinical_header("ENHANCED ENSEMBLE WITH ADVANCED FEATURES")
    
    # Combine probabilities from successful models
    ensemble_probabilities = []
    successful_models = []
    
    for model_name in models.keys():
        if model_name in all_analyses:
            successful_models.append(model_name)
    
    print(f"🏆 Creating ensemble from {len(successful_models)} successful models")
    
    if len(successful_models) >= 2:
        for fold_idx in range(5):
            fold_probas = []
            for model_name in successful_models:
                if fold_idx < len(all_y_proba[model_name]):
                    fold_probas.append(all_y_proba[model_name][fold_idx])
            
            if fold_probas:
                # Weight based on clinical performance (favor balanced models)
                weights = []
                for model_name in successful_models:
                    if fold_idx < len(fold_results) and model_name in fold_results[fold_idx]:
                        sens = fold_results[fold_idx][model_name]['sensitivity']
                        spec = fold_results[fold_idx][model_name]['specificity']
                        # Balance weight: favor models closer to both targets
                        balance_score = 1 / (1 + abs(sens - 0.86) + abs(spec - 0.71))
                        weights.append(balance_score)
                    else:
                        weights.append(1.0)
                
                weights = np.array(weights) / np.sum(weights)
                ensemble_proba = np.average(fold_probas, axis=0, weights=weights)
                ensemble_probabilities.append(ensemble_proba)
        
        if ensemble_probabilities:
            print("🔬 Analyzing ENHANCED ENSEMBLE...")
            fold_names = [f"Ensemble_Fold_{i+1}" for i in range(len(ensemble_probabilities))]
            
            ensemble_scenarios, ensemble_pareto = analyze_clinical_thresholds(all_y_true, ensemble_probabilities, fold_names)
            all_analyses['ENHANCED_ENSEMBLE'] = {
                'scenarios': ensemble_scenarios,
                'pareto_points': ensemble_pareto[:10]
            }
    
    # Save comprehensive results
    print("\n💾 Saving comprehensive analysis results...")
    
    summary_results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': int(X.shape[0]),
            'total_features_original': int(X.shape[1]),
            'total_features_selected': int(X_selected.shape[1]),
            'unique_children': int(len(np.unique(child_ids))),
            'asd_samples': int(np.sum(y)),
            'td_samples': int(len(y) - np.sum(y))
        },
        'clinical_targets': {'sensitivity': 0.86, 'specificity': 0.71},
        'models_analyzed': list(all_analyses.keys()),
        'detailed_analyses': {}
    }
    
    # Save detailed results for each model
    for model_name, analysis in all_analyses.items():
        summary_results['detailed_analyses'][model_name] = {
            'best_scenario': min(analysis['scenarios'], 
                               key=lambda x: np.sqrt((x['sensitivity'] - 0.86)**2 + (x['specificity'] - 0.71)**2)),
            'most_balanced': min(analysis['scenarios'], 
                               key=lambda x: abs(x['sensitivity'] - x['specificity'])),
            'best_clinical_safety': max(analysis['scenarios'], key=lambda x: x['clinical_safety']),
            'top_pareto_points': analysis['pareto_points'][:3]
        }
    
    with open(results_dir / "comprehensive_clinical_analysis.json", 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_types(summary_results), f, indent=2)
    
    print(f"   ✅ Comprehensive results saved to {results_dir}/")
    
    # Final recommendations
    print_clinical_header("🎯 FINAL CLINICAL RECOMMENDATIONS")
    
    # Find overall best approaches
    all_scenarios = []
    for model_name, analysis in all_analyses.items():
        for scenario in analysis['scenarios']:
            scenario['model'] = model_name
            all_scenarios.append(scenario)
    
    # Best for clinical targets
    best_clinical_overall = min(all_scenarios,
                               key=lambda x: np.sqrt((x['sensitivity'] - 0.86)**2 + (x['specificity'] - 0.71)**2))
    
    # Most balanced overall
    most_balanced_overall = min(all_scenarios,
                               key=lambda x: abs(x['sensitivity'] - x['specificity']))
    
    # Best clinical safety
    best_safety_overall = max(all_scenarios, key=lambda x: x['clinical_safety'])
    
    print("🏆 TOP RECOMMENDATIONS:")
    print(f"\n1️⃣  CLOSEST TO CLINICAL TARGETS:")
    print(f"   Model: {best_clinical_overall['model']}")
    print(f"   Strategy: {best_clinical_overall['strategy']}")
    print(f"   Sensitivity: {best_clinical_overall['sensitivity']:.3f} {'✅' if best_clinical_overall['sens_target_met'] else '❌'}")
    print(f"   Specificity: {best_clinical_overall['specificity']:.3f} {'✅' if best_clinical_overall['spec_target_met'] else '❌'}")
    print(f"   Threshold: {best_clinical_overall['threshold']:.3f}")
    print(f"   Clinical Viability: {best_clinical_overall['clinical_viability']}")
    
    print(f"\n2️⃣  MOST BALANCED APPROACH:")
    print(f"   Model: {most_balanced_overall['model']}")
    print(f"   Strategy: {most_balanced_overall['strategy']}")
    print(f"   Sensitivity: {most_balanced_overall['sensitivity']:.3f}")
    print(f"   Specificity: {most_balanced_overall['specificity']:.3f}")
    print(f"   Balance: {abs(most_balanced_overall['sensitivity'] - most_balanced_overall['specificity']):.3f}")
    
    print(f"\n3️⃣  BEST CLINICAL SAFETY:")
    print(f"   Model: {best_safety_overall['model']}")
    print(f"   Strategy: {best_safety_overall['strategy']}")
    print(f"   Clinical Safety Score: {best_safety_overall['clinical_safety']:.3f}")
    print(f"   Sensitivity: {best_safety_overall['sensitivity']:.3f}")
    print(f"   Specificity: {best_safety_overall['specificity']:.3f}")
    
    # Reality check on targets
    achievable_both = any(s['both_targets_met'] for s in all_scenarios)
    close_to_both = any(s['sensitivity'] >= 0.80 and s['specificity'] >= 0.65 for s in all_scenarios)
    
    print(f"\n📊 REALITY CHECK ON CLINICAL TARGETS:")
    if achievable_both:
        print("   🎉 BOTH targets (86% sens, 71% spec) ARE achievable with current data!")
    elif close_to_both:
        print("   ⚠️  Current targets challenging but close approaches exist (80%+ sens, 65%+ spec)")
        print("   💡 Consider relaxing targets to 80% sensitivity, 65% specificity for clinical viability")
    else:
        print("   ❌ Current clinical targets appear unrealistic with available features")
        print("   💡 Recommend targets: 75% sensitivity, 65% specificity as clinically viable")
    
    print_clinical_header("🏆 ANALYSIS COMPLETE!")
    print("📊 Comprehensive feature engineering and threshold analysis completed")
    print("🏥 Clinical recommendations provided based on realistic trade-off analysis")
    print("💾 Detailed results saved for further review")
    
    return all_analyses, summary_results

if __name__ == "__main__":
    analyses, summary = advanced_model_with_engineered_features()
    print("\n🏆 Advanced clinical investigation complete!")
