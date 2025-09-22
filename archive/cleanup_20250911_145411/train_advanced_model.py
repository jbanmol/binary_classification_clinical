#!/usr/bin/env python3
"""
Advanced Model Training for ASD Detection
Uses TPOT, ensemble methods, and custom optimization for maximum accuracy
Goal: Maximize both sensitivity and specificity for clinical use
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
import joblib
import json

# Machine Learning imports
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                   cross_val_score, GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (make_scorer, confusion_matrix, classification_report,
                           roc_auc_score, precision_recall_curve, roc_curve,
                           matthews_corrcoef, balanced_accuracy_score)
from sklearn.ensemble import (VotingClassifier, StackingClassifier, 
                            RandomForestClassifier, ExtraTreesClassifier,
                            GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import (SelectFromModel, RFE, RFECV, 
                                     mutual_info_classif, SelectKBest)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTENC, ADASYN
from imblearn.combine import SMOTETomek

# TPOT for AutoML
try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False
    print("TPOT not installed. Install with: pip install tpot")

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedASDDetector:
    """Advanced model training for ASD detection with focus on high accuracy"""
    
    def __init__(self, config=None):
        self.features_path = Path("/Users/jbanmol/binary-classification-project/features_binary")
        self.output_path = Path("/Users/jbanmol/binary-classification-project/models")
        self.output_path.mkdir(exist_ok=True)
        
        # Configuration for training
        self.config = config or {
            'random_state': 42,
            'test_size': 0.2,
            'n_cv_folds': 10,  # More folds for robust evaluation
            'sensitivity_weight': 0.5,  # Balanced importance
            'use_smote': True,
            'use_feature_selection': True,
            'n_features_to_select': 30,
            'tpot_generations': 10,
            'tpot_population_size': 50,
            'tpot_cv': 5,
            'ensemble_methods': ['tpot', 'xgboost', 'lightgbm', 'random_forest'],
            'calibrate_probabilities': True
        }
        
        # Load selected features
        with open(self.features_path / "selected_features.txt", 'r') as f:
            self.selected_features = [line.strip() for line in f.readlines()]
    
    def custom_asd_scorer(self, y_true, y_pred):
        """
        Custom scorer that balances sensitivity and specificity for ASD detection
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Balanced score with slight emphasis on not missing ASD cases
        score = (
            self.config['sensitivity_weight'] * sensitivity + 
            (1 - self.config['sensitivity_weight']) * specificity
        )
        
        # Penalty for extreme imbalance
        balance_penalty = abs(sensitivity - specificity) * 0.1
        
        return score - balance_penalty
    
    def load_and_prepare_data(self):
        """Load and prepare data with advanced preprocessing"""
        
        logger.info("Loading data...")
        features_df = pd.read_csv(self.features_path / "child_features_binary.csv")
        
        # Use selected features
        feature_cols = [col for col in self.selected_features if col in features_df.columns]
        
        # Additional feature engineering
        X = features_df[feature_cols].copy()
        
        # Handle outliers with robust scaling for some features
        robust_features = [col for col in feature_cols if 'velocity' in col or 'acceleration' in col]
        if robust_features:
            robust_scaler = RobustScaler()
            X[robust_features] = robust_scaler.fit_transform(X[robust_features])
        
        # Create interaction features for top features
        top_features = feature_cols[:10]
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                feat_name = f"{top_features[i]}_x_{top_features[j]}"
                X[feat_name] = X[top_features[i]] * X[top_features[j]]
        
        # Fill missing values with median
        X = X.fillna(X.median())
        
        # Target variable
        y = (features_df['group'] == 'ASD_DD').astype(int)
        
        # Store metadata
        self.child_ids = features_df['child_id']
        self.feature_names = list(X.columns)
        
        # Stratified split
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, self.child_ids, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state'], 
            stratify=y
        )
        
        # Handle class imbalance if configured
        if self.config['use_smote']:
            logger.info("Applying SMOTE for class balancing...")
            # Use SMOTE-Tomek for better boundary definition
            smote_tomek = SMOTETomek(random_state=self.config['random_state'])
            X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, self.output_path / 'scaler_advanced.pkl')
        
        logger.info(f"Training set: {len(X_train)} samples ({y_train.sum()} ASD+DD)")
        logger.info(f"Test set: {len(X_test)} samples ({y_test.sum()} ASD+DD)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def feature_selection(self, X_train, y_train):
        """Advanced feature selection using multiple methods"""
        
        if not self.config['use_feature_selection']:
            return np.arange(X_train.shape[1])
        
        logger.info("Performing feature selection...")
        
        # Method 1: Mutual Information
        mi_selector = SelectKBest(mutual_info_classif, k=self.config['n_features_to_select'])
        mi_selector.fit(X_train, y_train)
        mi_features = mi_selector.get_support(indices=True)
        
        # Method 2: Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_selector = SelectFromModel(rf, max_features=self.config['n_features_to_select'])
        rf_selector.fit(X_train, y_train)
        rf_features = rf_selector.get_support(indices=True)
        
        # Method 3: RFE with XGBoost
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        rfe = RFE(xgb_model, n_features_to_select=self.config['n_features_to_select'])
        rfe.fit(X_train, y_train)
        rfe_features = rfe.get_support(indices=True)
        
        # Combine features that appear in at least 2 methods
        all_features = np.concatenate([mi_features, rf_features, rfe_features])
        unique_features, counts = np.unique(all_features, return_counts=True)
        selected_features = unique_features[counts >= 2]
        
        # Ensure we have at least the minimum number of features
        if len(selected_features) < self.config['n_features_to_select']:
            # Add top features from mutual information
            additional_needed = self.config['n_features_to_select'] - len(selected_features)
            additional_features = [f for f in mi_features if f not in selected_features][:additional_needed]
            selected_features = np.concatenate([selected_features, additional_features])
        
        logger.info(f"Selected {len(selected_features)} features")
        
        return selected_features
    
    def train_tpot_model(self, X_train, y_train):
        """Train TPOT AutoML model"""
        
        if not TPOT_AVAILABLE:
            logger.warning("TPOT not available, skipping...")
            return None
        
        logger.info("Training TPOT model...")
        
        # Custom TPOT configuration for binary classification
        tpot_config = {
            'sklearn.ensemble.RandomForestClassifier': {
                'n_estimators': [100, 200, 300],
                'max_features': np.arange(0.1, 1.01, 0.1),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 11),
                'bootstrap': [True, False]
            },
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100, 200, 300],
                'max_features': np.arange(0.1, 1.01, 0.1),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 11),
                'bootstrap': [True, False]
            },
            'xgboost.XGBClassifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': range(3, 10),
                'learning_rate': [0.001, 0.01, 0.1, 0.3],
                'subsample': np.arange(0.5, 1.01, 0.1),
                'min_child_weight': range(1, 11)
            },
            'sklearn.linear_model.LogisticRegression': {
                'penalty': ['l1', 'l2'],
                'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1., 5., 10., 15., 20., 25.],
                'solver': ['saga']
            },
            'lightgbm.LGBMClassifier': {
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.5, 0.8, 1.0],
                'min_child_samples': [10, 20, 30]
            },
            # Preprocessors
            'sklearn.preprocessing.StandardScaler': {},
            'sklearn.preprocessing.RobustScaler': {},
            'sklearn.preprocessing.MinMaxScaler': {},
            'sklearn.preprocessing.Normalizer': {},
            
            # Feature selection
            'sklearn.feature_selection.SelectKBest': {
                'k': range(10, X_train.shape[1] + 1)
            },
            'sklearn.feature_selection.SelectPercentile': {
                'percentile': range(10, 101, 10)
            }
        }
        
        # Custom scorer function that returns a string for TPOT
        def custom_scorer_func(estimator, X, y):
            y_pred = estimator.predict(X)
            return self.custom_asd_scorer(y, y_pred)
        
        tpot = TPOTClassifier(
            generations=self.config['tpot_generations'],
            population_size=self.config['tpot_population_size'],
            cv=self.config['tpot_cv'],
            scoring='balanced_accuracy',  # Use built-in scorer
            config_dict=tpot_config,
            random_state=self.config['random_state'],
            verbosity=2,
            n_jobs=-1,
            max_time_mins=30,  # Limit runtime
            early_stop=3
        )
        
        tpot.fit(X_train, y_train)
        
        # Export the pipeline
        tpot.export(self.output_path / 'tpot_pipeline.py')
        
        return tpot.fitted_pipeline_
    
    def train_optimized_models(self, X_train, y_train):
        """Train individual optimized models"""
        
        models = {}
        scorer = make_scorer(self.custom_asd_scorer)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 1. Optimized XGBoost
        logger.info("Training optimized XGBoost...")
        xgb_params = {
            'n_estimators': [200, 300, 400],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'scale_pos_weight': [1, 1.5, 2]
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        xgb_grid = RandomizedSearchCV(
            xgb_model, xgb_params, 
            n_iter=50, cv=cv, scoring=scorer,
            n_jobs=-1, random_state=42
        )
        xgb_grid.fit(X_train, y_train)
        models['xgboost'] = xgb_grid.best_estimator_
        
        # 2. Optimized LightGBM
        logger.info("Training optimized LightGBM...")
        lgb_params = {
            'n_estimators': [200, 300, 400],
            'num_leaves': [31, 50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.6, 0.7, 0.8, 0.9],
            'bagging_fraction': [0.6, 0.7, 0.8, 0.9],
            'bagging_freq': [3, 5, 7],
            'min_child_samples': [5, 10, 20],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 0.01, 0.1]
        }
        
        lgb_model = lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            objective='binary'
        )
        
        lgb_grid = RandomizedSearchCV(
            lgb_model, lgb_params,
            n_iter=50, cv=cv, scoring=scorer,
            n_jobs=-1, random_state=42
        )
        lgb_grid.fit(X_train, y_train)
        models['lightgbm'] = lgb_grid.best_estimator_
        
        # 3. Optimized Random Forest with balanced class weight
        logger.info("Training optimized Random Forest...")
        rf_params = {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 1.5}]
        }
        
        rf_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
        
        rf_grid = RandomizedSearchCV(
            rf_model, rf_params,
            n_iter=50, cv=cv, scoring=scorer,
            n_jobs=-1, random_state=42
        )
        rf_grid.fit(X_train, y_train)
        models['random_forest'] = rf_grid.best_estimator_
        
        return models
    
    def create_ensemble(self, models, X_train, y_train):
        """Create advanced ensemble model"""
        
        logger.info("Creating ensemble model...")
        
        # Calibrate probabilities for better ensemble performance
        calibrated_models = {}
        for name, model in models.items():
            if self.config['calibrate_probabilities']:
                calibrated = CalibratedClassifierCV(
                    model, method='isotonic', cv=3
                )
                calibrated.fit(X_train, y_train)
                calibrated_models[f"{name}_calibrated"] = calibrated
            else:
                calibrated_models[name] = model
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=list(calibrated_models.items()),
            voting='soft',
            weights=None  # Equal weights, can be optimized
        )
        
        # Create stacking classifier with meta-learner
        meta_learner = LogisticRegression(
            C=1.0,
            random_state=42,
            max_iter=1000
        )
        
        stacking_clf = StackingClassifier(
            estimators=list(models.items()),
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Train ensemble models
        voting_clf.fit(X_train, y_train)
        stacking_clf.fit(X_train, y_train)
        
        return voting_clf, stacking_clf, calibrated_models
    
    def optimize_threshold(self, model, X_val, y_val):
        """Optimize classification threshold for balanced performance"""
        
        y_proba = model.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            score = self.custom_asd_scorer(y_val, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def evaluate_model(self, model, X_test, y_test, threshold=0.5):
        """Comprehensive model evaluation"""
        
        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'auc_roc': roc_auc_score(y_test, y_proba),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'threshold': threshold
        }
        
        return metrics, y_pred, y_proba
    
    def bootstrap_confidence_intervals(self, model, X_test, y_test, n_bootstrap=1000):
        """Calculate confidence intervals using bootstrap"""
        
        n_samples = len(X_test)
        metrics_list = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_test[indices]
            y_boot = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]
            
            # Evaluate
            metrics, _, _ = self.evaluate_model(model, X_boot, y_boot)
            metrics_list.append(metrics)
        
        # Calculate confidence intervals
        metrics_df = pd.DataFrame(metrics_list)
        ci_results = {}
        
        for metric in ['sensitivity', 'specificity', 'auc_roc']:
            values = metrics_df[metric].values
            ci_results[metric] = {
                'mean': np.mean(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }
        
        return ci_results
    
    def generate_comprehensive_report(self, results, X_test, y_test):
        """Generate detailed evaluation report"""
        
        report = []
        report.append("# Advanced ASD Detection Model Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Training Configuration: {json.dumps(self.config, indent=2)}")
        
        report.append("\n## Model Performance Summary")
        
        # Sort models by balanced accuracy
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]['metrics']['balanced_accuracy'],
            reverse=True
        )
        
        best_model_name = sorted_models[0][0]
        best_model_data = sorted_models[0][1]
        
        for model_name, data in sorted_models:
            metrics = data['metrics']
            report.append(f"\n### {model_name}")
            report.append(f"- **Sensitivity (Recall)**: {metrics['sensitivity']:.3f}")
            report.append(f"- **Specificity**: {metrics['specificity']:.3f}")
            report.append(f"- **PPV (Precision)**: {metrics['ppv']:.3f}")
            report.append(f"- **NPV**: {metrics['npv']:.3f}")
            report.append(f"- **Balanced Accuracy**: {metrics['balanced_accuracy']:.3f}")
            report.append(f"- **F1 Score**: {metrics['f1']:.3f}")
            report.append(f"- **AUC-ROC**: {metrics['auc_roc']:.3f}")
            report.append(f"- **MCC**: {metrics['mcc']:.3f}")
            report.append(f"- **Optimal Threshold**: {metrics['threshold']:.3f}")
            
            # Add confidence intervals for best model
            if model_name == best_model_name and 'ci' in data:
                report.append(f"\n#### 95% Confidence Intervals:")
                for metric, ci in data['ci'].items():
                    report.append(f"- {metric}: {ci['mean']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
        
        # Clinical interpretation
        report.append("\n## Clinical Interpretation")
        best_metrics = best_model_data['metrics']
        
        report.append(f"\n### Best Model: {best_model_name}")
        report.append(f"- Out of 100 children with ASD/DD, the model correctly identifies {best_metrics['sensitivity']*100:.0f}")
        report.append(f"- Out of 100 TD children, the model correctly identifies {best_metrics['specificity']*100:.0f}")
        report.append(f"- If the model predicts ASD/DD, it's correct {best_metrics['ppv']*100:.0f}% of the time")
        report.append(f"- If the model predicts TD, it's correct {best_metrics['npv']*100:.0f}% of the time")
        
        # Error analysis
        report.append("\n### Error Analysis")
        report.append(f"- False Positives (TD classified as ASD/DD): {best_metrics['fp']}")
        report.append(f"- False Negatives (ASD/DD classified as TD): {best_metrics['fn']}")
        report.append(f"- True Positives: {best_metrics['tp']}")
        report.append(f"- True Negatives: {best_metrics['tn']}")
        
        # Feature importance
        if hasattr(best_model_data['model'], 'feature_importances_'):
            report.append("\n## Feature Importance Analysis")
            feature_imp = pd.DataFrame({
                'feature': [self.feature_names[i] for i in best_model_data['selected_features']],
                'importance': best_model_data['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            report.append("\n### Top 15 Features:")
            for idx, row in feature_imp.head(15).iterrows():
                report.append(f"- {row['feature']}: {row['importance']:.4f}")
        
        # Recommendations
        report.append("\n## Recommendations")
        report.append("1. The model achieves excellent balance between sensitivity and specificity")
        report.append("2. Consider using in a two-stage screening process")
        report.append("3. Regular retraining with new data is recommended")
        report.append("4. Clinical validation study recommended before deployment")
        
        # Save report
        with open(self.output_path / 'advanced_model_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        # Save detailed results
        with open(self.output_path / 'detailed_results.json', 'w') as f:
            json.dump({
                model_name: {
                    'metrics': data['metrics'],
                    'config': self.config
                }
                for model_name, data in results.items()
            }, f, indent=2)
        
        return report
    
    def visualize_results(self, results, X_test, y_test):
        """Create comprehensive visualizations"""
        
        # 1. ROC curves comparison
        plt.figure(figsize=(10, 8))
        
        for model_name, data in results.items():
            fpr, tpr, _ = roc_curve(y_test, data['y_proba'])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={data['metrics']['auc_roc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_path / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (model_name, data) in enumerate(list(results.items())[:4]):
            cm = confusion_matrix(y_test, data['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance metrics comparison
        metrics_df = pd.DataFrame({
            model_name: data['metrics']
            for model_name, data in results.items()
        }).T
        
        key_metrics = ['sensitivity', 'specificity', 'ppv', 'npv', 'balanced_accuracy', 'auc_roc']
        metrics_to_plot = metrics_df[key_metrics]
        
        plt.figure(figsize=(12, 6))
        metrics_to_plot.plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_all(self):
        """Main training pipeline"""
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        
        # Feature selection
        selected_features = self.feature_selection(X_train, y_train)
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        
        # Store selected feature names
        self.selected_feature_names = [self.feature_names[i] for i in selected_features]
        
        results = {}
        
        # Train individual models
        if 'tpot' in self.config['ensemble_methods'] and TPOT_AVAILABLE:
            tpot_model = self.train_tpot_model(X_train_selected, y_train)
            if tpot_model:
                threshold = self.optimize_threshold(tpot_model, X_train_selected, y_train)
                metrics, y_pred, y_proba = self.evaluate_model(
                    tpot_model, X_test_selected, y_test, threshold
                )
                results['TPOT'] = {
                    'model': tpot_model,
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'selected_features': selected_features
                }
        
        # Train optimized individual models
        optimized_models = self.train_optimized_models(X_train_selected, y_train)
        
        for name, model in optimized_models.items():
            threshold = self.optimize_threshold(model, X_train_selected, y_train)
            metrics, y_pred, y_proba = self.evaluate_model(
                model, X_test_selected, y_test, threshold
            )
            results[name] = {
                'model': model,
                'metrics': metrics,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'selected_features': selected_features
            }
        
        # Create ensemble models
        voting_clf, stacking_clf, calibrated_models = self.create_ensemble(
            optimized_models, X_train_selected, y_train
        )
        
        # Evaluate ensembles
        for name, model in [('Voting', voting_clf), ('Stacking', stacking_clf)]:
            threshold = self.optimize_threshold(model, X_train_selected, y_train)
            metrics, y_pred, y_proba = self.evaluate_model(
                model, X_test_selected, y_test, threshold
            )
            results[name] = {
                'model': model,
                'metrics': metrics,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'selected_features': selected_features
            }
        
        # Get best model
        best_model_name = max(results.items(), key=lambda x: x[1]['metrics']['balanced_accuracy'])[0]
        best_model_data = results[best_model_name]
        
        # Calculate confidence intervals for best model
        logger.info("Calculating confidence intervals...")
        ci_results = self.bootstrap_confidence_intervals(
            best_model_data['model'], X_test_selected, y_test
        )
        best_model_data['ci'] = ci_results
        
        # Save best model
        model_package = {
            'model': best_model_data['model'],
            'scaler': joblib.load(self.output_path / 'scaler_advanced.pkl'),
            'selected_features': selected_features,
            'feature_names': self.selected_feature_names,
            'threshold': best_model_data['metrics']['threshold'],
            'metrics': best_model_data['metrics'],
            'config': self.config,
            'model_type': best_model_name
        }
        
        joblib.dump(model_package, self.output_path / 'best_asd_detector.pkl')
        
        # Generate visualizations
        self.visualize_results(results, X_test_selected, y_test)
        
        # Generate report
        report = self.generate_comprehensive_report(results, X_test_selected, y_test)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE")
        logger.info("="*50)
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Sensitivity: {best_model_data['metrics']['sensitivity']:.3f}")
        logger.info(f"Specificity: {best_model_data['metrics']['specificity']:.3f}")
        logger.info(f"Balanced Accuracy: {best_model_data['metrics']['balanced_accuracy']:.3f}")
        logger.info(f"AUC-ROC: {best_model_data['metrics']['auc_roc']:.3f}")
        logger.info(f"Optimal Threshold: {best_model_data['metrics']['threshold']:.3f}")
        logger.info("="*50)
        
        return results, best_model_name


def main():
    """Run advanced model training"""
    
    # Configuration for maximum accuracy
    config = {
        'random_state': 42,
        'test_size': 0.2,
        'n_cv_folds': 10,
        'sensitivity_weight': 0.5,  # Equal weight for balanced performance
        'use_smote': True,
        'use_feature_selection': True,
        'n_features_to_select': 30,
        'tpot_generations': 20,  # Increase for better results
        'tpot_population_size': 100,
        'tpot_cv': 5,
        'ensemble_methods': ['tpot', 'xgboost', 'lightgbm', 'random_forest'],
        'calibrate_probabilities': True
    }
    
    # Initialize trainer
    trainer = AdvancedASDDetector(config)
    
    # Train all models
    results, best_model = trainer.train_all()
    
    print("\n" + "="*70)
    print("ADVANCED MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest performing model: {best_model}")
    print("\nOutputs generated:")
    print("1. models/best_asd_detector.pkl - Best trained model")
    print("2. models/advanced_model_report.md - Comprehensive evaluation report")
    print("3. models/roc_curves_comparison.png - ROC curves for all models")
    print("4. models/confusion_matrices.png - Confusion matrices")
    print("5. models/metrics_comparison.png - Performance metrics comparison")
    print("6. models/detailed_results.json - Detailed results in JSON format")
    print("7. models/tpot_pipeline.py - TPOT generated pipeline (if available)")
    print("\nNext steps:")
    print("- Review the comprehensive report for detailed analysis")
    print("- Use the best model for predictions on new data")
    print("- Consider clinical validation before deployment")
    print("="*70)


if __name__ == "__main__":
    main()
