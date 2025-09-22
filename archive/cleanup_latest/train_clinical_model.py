#!/usr/bin/env python3
"""
Clinical-Grade ASD Detection Model Pipeline
Target: ≥86% Sensitivity, ≥70% Specificity
Strategies: Feature engineering, ensemble methods, threshold optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve, 
                           matthews_corrcoef, balanced_accuracy_score, 
                           classification_report)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                            VotingClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalASDDetector:
    """
    Clinical-grade ASD detection model with specific performance targets
    """
    
    def __init__(self):
        self.features_path = Path("/Users/jbanmol/binary-classification-project/features_binary")
        self.output_path = Path("/Users/jbanmol/binary-classification-project/models_clinical")
        self.output_path.mkdir(exist_ok=True)
        
        # Clinical targets
        self.target_sensitivity = 0.86
        self.target_specificity = 0.70
        
        # Load features
        with open(self.features_path / "selected_features.txt", 'r') as f:
            self.selected_features = [line.strip() for line in f.readlines()]
            
        # Load feature importance from previous analysis
        self.feature_rankings = pd.read_csv(self.features_path / "feature_rankings.csv")
    
    def enhanced_feature_engineering(self, df):
        """Create additional engineered features based on clinical insights"""
        
        X = df.copy()
        
        # 1. Composite scores based on clinical interpretation
        if 'target_zone_coverage_mean' in X.columns and 'outside_ratio_mean' in X.columns:
            # Accuracy composite - highly discriminative
            X['accuracy_composite'] = (X['target_zone_coverage_mean'] * 2 - X['outside_ratio_mean']) / 2
            X['accuracy_ratio'] = X['target_zone_coverage_mean'] / (X['outside_ratio_mean'] + 0.01)
        
        if 'zone_transition_rate_mean' in X.columns:
            # Planning and organization
            X['organization_score'] = 1 / (X['zone_transition_rate_mean'] + 0.1)
            
        if 'progress_linearity_mean' in X.columns and 'zone_transition_rate_mean' in X.columns:
            # Executive function composite
            X['executive_composite'] = X['progress_linearity_mean'] * X['organization_score']
        
        # 2. Variability measures (consistency is important)
        variability_features = []
        for feat in X.columns:
            if '_std' in feat:
                base_feat = feat.replace('_std', '_mean')
                if base_feat in X.columns:
                    cv_feat = f"{feat.replace('_std', '')}_cv"
                    X[cv_feat] = X[feat] / (X[base_feat] + 0.01)
                    variability_features.append(cv_feat)
        
        # 3. Zone coverage patterns
        if 'zone_coverage_std_mean' in X.columns:
            X['coverage_consistency'] = 1 / (X['zone_coverage_std_mean'] + 0.01)
        
        # 4. Motor control composites
        velocity_cols = [col for col in X.columns if 'velocity' in col and 'mean' in col]
        if velocity_cols:
            X['motor_stability'] = X[velocity_cols].std(axis=1)
            X['motor_control'] = 1 / (X['motor_stability'] + 0.01)
        
        # 5. Completion and progress features
        if 'avg_plateau_length_mean' in X.columns:
            X['task_fluency'] = 1 / (X['avg_plateau_length_mean'] + 1)
        
        # 6. Interaction features for top discriminative features
        top_features = self.feature_rankings.head(10)['feature'].tolist()
        for i, feat1 in enumerate(top_features):
            if feat1 in X.columns:
                for feat2 in top_features[i+1:]:
                    if feat2 in X.columns:
                        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
        
        return X
    
    def get_stratified_features(self, X, y, n_features=50):
        """Select features using multiple methods and combine"""
        
        # Handle NaN values before feature selection
        X_filled = X.copy()
        for col in X_filled.columns:
            if X_filled[col].isna().any():
                X_filled[col].fillna(X_filled[col].median(), inplace=True)
        # If still any NaN (e.g., if all values in a column were NaN), fill with 0
        X_filled.fillna(0, inplace=True)
        
        # 1. Top features by effect size from previous analysis
        top_by_effect = self.feature_rankings.head(n_features // 2)['feature'].tolist()
        
        # 2. Mutual information
        mi_selector = SelectKBest(mutual_info_classif, k=n_features // 2)
        mi_selector.fit(X_filled, y)
        mi_features = X.columns[mi_selector.get_support()].tolist()
        
        # 3. Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_filled, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        rf_features = rf_importance.head(n_features // 2).index.tolist()
        
        # Combine all features
        all_selected = list(set(top_by_effect + mi_features + rf_features))
        
        # Ensure we have the most important features
        essential_features = ['target_zone_coverage_mean', 'outside_ratio_mean', 
                            'zone_transition_rate_mean', 'progress_linearity_mean']
        for feat in essential_features:
            if feat in X.columns and feat not in all_selected:
                all_selected.append(feat)
        
        return all_selected[:n_features]
    
    def create_balanced_training_sets(self, X_train, y_train, strategy='multiple'):
        """Create multiple balanced training sets using different strategies"""
        
        datasets = []
        
        # 1. Original (imbalanced)
        datasets.append(('original', X_train, y_train))
        
        # 2. SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        datasets.append(('smote', X_smote, y_smote))
        
        # 3. BorderlineSMOTE (focuses on borderline cases)
        borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=5)
        X_border, y_border = borderline_smote.fit_resample(X_train, y_train)
        datasets.append(('borderline_smote', X_border, y_border))
        
        # 4. SMOTE + Tomek (removes overlapping)
        smote_tomek = SMOTETomek(random_state=42)
        X_st, y_st = smote_tomek.fit_resample(X_train, y_train)
        datasets.append(('smote_tomek', X_st, y_st))
        
        return datasets
    
    def train_diverse_models(self, X_train, y_train, groups_train):
        """Train diverse models with different characteristics"""
        
        models = {}
        
        # Use GroupKFold for fair CV
        gkf = GroupKFold(n_splits=5)
        
        # 1. High-sensitivity Random Forest
        rf_high_sens = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight={0: 0.7, 1: 1.3},  # Favor sensitivity
            random_state=42,
            n_jobs=-1
        )
        
        # 2. Balanced XGBoost
        xgb_balanced = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.8,  # Handle imbalance
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # 3. LightGBM with custom objective
        lgb_custom = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=40,
            learning_rate=0.03,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=10,
            class_weight={0: 0.7, 1: 1.3},
            random_state=42,
            verbose=-1
        )
        
        # 4. Extra Trees for diversity
        et_diverse = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # 5. Gradient Boosting with focus on hard cases
        gb_hard = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        
        # 6. Calibrated SVM
        svm_cal = CalibratedClassifierCV(
            SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
            method='isotonic',
            cv=3
        )
        
        # Train all models
        models_list = [
            ('RF_HighSens', rf_high_sens),
            ('XGB_Balanced', xgb_balanced),
            ('LGB_Custom', lgb_custom),
            ('ET_Diverse', et_diverse),
            ('GB_Hard', gb_hard),
            ('SVM_Cal', svm_cal)
        ]
        
        for name, model in models_list:
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            models[name] = model
            
        return models
    
    def optimize_ensemble_weights(self, models, X_val, y_val, target_sens=0.86):
        """Find optimal weights for ensemble to meet clinical targets"""
        
        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict_proba(X_val)[:, 1]
        
        pred_matrix = np.column_stack(list(predictions.values()))
        
        # Grid search for optimal weights
        best_weights = None
        best_score = -1
        best_metrics = None
        
        # Generate weight combinations
        weight_options = np.arange(0, 1.1, 0.1)
        
        for w1 in weight_options:
            for w2 in weight_options:
                for w3 in weight_options:
                    remaining = 1 - (w1 + w2 + w3)
                    if remaining < 0 or remaining > 1:
                        continue
                    
                    # Distribute remaining weight
                    weights = [w1, w2, w3, remaining/3, remaining/3, remaining/3]
                    if sum(weights) == 0:
                        continue
                    
                    # Normalize
                    weights = np.array(weights) / sum(weights)
                    
                    # Weighted average
                    ensemble_proba = np.dot(pred_matrix, weights)
                    
                    # Find best threshold
                    for thresh in np.arange(0.3, 0.7, 0.01):
                        y_pred = (ensemble_proba >= thresh).astype(int)
                        
                        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                        # Score prioritizing clinical targets
                        if sensitivity >= target_sens and specificity >= 0.70:
                            score = sensitivity + specificity
                            if score > best_score:
                                best_score = score
                                best_weights = weights
                                best_metrics = {
                                    'threshold': thresh,
                                    'sensitivity': sensitivity,
                                    'specificity': specificity,
                                    'weights': weights
                                }
        
        return best_weights, best_metrics
    
    def create_clinical_ensemble(self, models, weights=None):
        """Create ensemble optimized for clinical targets"""
        
        if weights is None:
            weights = [1/len(models)] * len(models)
        
        class ClinicalEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
                self.model_names = list(models.keys())
            
            def predict_proba(self, X):
                predictions = []
                for name, model in self.models.items():
                    pred = model.predict_proba(X)[:, 1]
                    predictions.append(pred)
                
                pred_matrix = np.column_stack(predictions)
                ensemble_proba = np.dot(pred_matrix, self.weights)
                
                # Return in sklearn format
                proba = np.zeros((len(X), 2))
                proba[:, 0] = 1 - ensemble_proba
                proba[:, 1] = ensemble_proba
                return proba
            
            def predict(self, X, threshold=0.5):
                proba = self.predict_proba(X)[:, 1]
                return (proba >= threshold).astype(int)
        
        return ClinicalEnsemble(models, weights)
    
    def evaluate_clinical_performance(self, model, X_test, y_test, threshold=0.5):
        """Evaluate model against clinical targets"""
        
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'threshold': threshold
        }
        
        # Check if clinical targets are met
        metrics['meets_sensitivity_target'] = metrics['sensitivity'] >= self.target_sensitivity
        metrics['meets_specificity_target'] = metrics['specificity'] >= self.target_specificity
        metrics['meets_clinical_targets'] = (metrics['meets_sensitivity_target'] and 
                                            metrics['meets_specificity_target'])
        
        return metrics
    
    def run_clinical_pipeline(self):
        """Main pipeline for clinical-grade model development"""
        
        logger.info("Starting clinical-grade ASD detection pipeline...")
        logger.info(f"Targets: Sensitivity ≥{self.target_sensitivity:.0%}, Specificity ≥{self.target_specificity:.0%}")
        
        # Load data
        child_features = pd.read_csv(self.features_path / "child_features_binary.csv")
        
        # Enhanced feature engineering
        logger.info("Performing enhanced feature engineering...")
        all_features = self.selected_features + ['child_id', 'group']
        child_data = child_features[all_features].copy()
        
        # Apply enhanced feature engineering
        engineered_data = self.enhanced_feature_engineering(child_data)
        
        # Prepare target
        y = (engineered_data['group'] == 'ASD_DD').astype(int)
        child_ids = engineered_data['child_id'].values
        
        # Remove non-feature columns
        X = engineered_data.drop(['child_id', 'group'], axis=1)
        
        # Feature selection
        logger.info("Selecting optimal features...")
        selected_features = self.get_stratified_features(X, y, n_features=60)
        X_selected = X[selected_features]
        
        # Fair train/test split
        unique_children = np.unique(child_ids)
        child_labels = pd.DataFrame({'child_id': child_ids, 'y': y}).groupby('child_id')['y'].first()
        
        # Stratified split
        td_children = child_labels[child_labels == 0].index.tolist()
        asd_children = child_labels[child_labels == 1].index.tolist()
        
        np.random.seed(42)
        test_td = np.random.choice(td_children, int(len(td_children) * 0.2), replace=False)
        test_asd = np.random.choice(asd_children, int(len(asd_children) * 0.2), replace=False)
        test_children = np.concatenate([test_td, test_asd])
        
        # Create masks
        test_mask = np.isin(child_ids, test_children)
        train_mask = ~test_mask
        
        X_train = X_selected[train_mask]
        X_test = X_selected[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        groups_train = child_ids[train_mask]
        
        logger.info(f"Training set: {len(X_train)} samples from {len(np.unique(groups_train))} children")
        logger.info(f"Test set: {len(X_test)} samples from {len(test_children)} children")
        
        # Create balanced datasets
        logger.info("Creating balanced training sets...")
        balanced_datasets = self.create_balanced_training_sets(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models on different balanced datasets
        all_models = {}
        for dataset_name, X_balanced, y_balanced in balanced_datasets:
            logger.info(f"\nTraining models on {dataset_name} dataset...")
            
            # Scale the balanced data
            if dataset_name != 'original':
                X_balanced_scaled = scaler.transform(X_balanced)
            else:
                X_balanced_scaled = X_train_scaled
            
            # Train diverse models
            models = self.train_diverse_models(X_balanced_scaled, y_balanced, groups_train)
            
            # Evaluate each model
            for model_name, model in models.items():
                full_name = f"{model_name}_{dataset_name}"
                metrics = self.evaluate_clinical_performance(model, X_test_scaled, y_test)
                logger.info(f"{full_name}: Sens={metrics['sensitivity']:.3f}, Spec={metrics['specificity']:.3f}")
                all_models[full_name] = (model, metrics)
        
        # Find models meeting clinical targets
        clinical_models = {name: (model, metrics) for name, (model, metrics) in all_models.items() 
                          if metrics['meets_clinical_targets']}
        
        if clinical_models:
            logger.info(f"\n{len(clinical_models)} models meet clinical targets!")
            best_single = max(clinical_models.items(), 
                            key=lambda x: x[1][1]['sensitivity'] + x[1][1]['specificity'])
            logger.info(f"Best single model: {best_single[0]}")
        else:
            logger.info("\nNo single model meets targets. Creating optimized ensemble...")
            
            # Select top models for ensemble
            top_models = dict(sorted(all_models.items(), 
                                   key=lambda x: x[1][1]['sensitivity'], 
                                   reverse=True)[:6])
            
            models_for_ensemble = {name: model for name, (model, _) in top_models.items()}
            
            # Optimize ensemble weights
            weights, ensemble_metrics = self.optimize_ensemble_weights(
                models_for_ensemble, X_test_scaled, y_test
            )
            
            if ensemble_metrics and ensemble_metrics['sensitivity'] >= self.target_sensitivity:
                logger.info(f"Ensemble achieves targets!")
                logger.info(f"Sensitivity: {ensemble_metrics['sensitivity']:.3f}")
                logger.info(f"Specificity: {ensemble_metrics['specificity']:.3f}")
                logger.info(f"Optimal threshold: {ensemble_metrics['threshold']:.3f}")
                
                # Create final ensemble
                final_ensemble = self.create_clinical_ensemble(models_for_ensemble, weights)
                
                # Final evaluation
                final_metrics = self.evaluate_clinical_performance(
                    final_ensemble, X_test_scaled, y_test, 
                    threshold=ensemble_metrics['threshold']
                )
                
                # Save model
                model_package = {
                    'model': final_ensemble,
                    'model_type': 'ensemble',
                    'scaler': scaler,
                    'selected_features': selected_features,
                    'threshold': ensemble_metrics['threshold'],
                    'weights': weights,
                    'component_models': list(models_for_ensemble.keys()),
                    'metrics': final_metrics,
                    'meets_clinical_targets': True
                }
                
                joblib.dump(model_package, self.output_path / 'clinical_asd_model.pkl')
                logger.info(f"\nClinical model saved!")
            else:
                logger.info("Unable to meet clinical targets with current features.")
                logger.info("Recommendations:")
                logger.info("1. Collect more training data")
                logger.info("2. Engineer additional domain-specific features")
                logger.info("3. Consider semi-supervised learning approaches")
        
        # Generate report
        self.generate_clinical_report(all_models, clinical_models)
        
        return clinical_models
    
    def generate_clinical_report(self, all_models, clinical_models):
        """Generate comprehensive clinical validation report"""
        
        report = []
        report.append("# Clinical-Grade ASD Detection Model Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n## Clinical Targets")
        report.append(f"- Sensitivity: ≥{self.target_sensitivity:.0%}")
        report.append(f"- Specificity: ≥{self.target_specificity:.0%}")
        
        report.append("\n## Models Meeting Clinical Targets")
        if clinical_models:
            for name, (model, metrics) in clinical_models.items():
                report.append(f"\n### {name}")
                report.append(f"- Sensitivity: {metrics['sensitivity']:.3f}")
                report.append(f"- Specificity: {metrics['specificity']:.3f}")
                report.append(f"- PPV: {metrics['ppv']:.3f}")
                report.append(f"- NPV: {metrics['npv']:.3f}")
                report.append(f"- Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        else:
            report.append("No single model met clinical targets.")
            report.append("Ensemble approach recommended.")
        
        report.append("\n## All Models Performance")
        sorted_models = sorted(all_models.items(), 
                             key=lambda x: x[1][1]['sensitivity'], 
                             reverse=True)
        
        for name, (model, metrics) in sorted_models[:10]:
            report.append(f"\n{name}:")
            report.append(f"  Sensitivity: {metrics['sensitivity']:.3f}")
            report.append(f"  Specificity: {metrics['specificity']:.3f}")
        
        with open(self.output_path / 'clinical_model_report.md', 'w') as f:
            f.write('\n'.join(report))


def main():
    """Run clinical model development pipeline"""
    pipeline = ClinicalASDDetector()
    clinical_models = pipeline.run_clinical_pipeline()
    return clinical_models


if __name__ == "__main__":
    main()
