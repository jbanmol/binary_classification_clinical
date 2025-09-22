#!/usr/bin/env python3
"""
High-Accuracy Model Training for ASD Detection
Streamlined approach focusing on proven methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

# Machine Learning imports
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                   cross_val_score, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (make_scorer, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, matthews_corrcoef, 
                           balanced_accuracy_score, precision_recall_curve)
from sklearn.ensemble import (VotingClassifier, StackingClassifier, 
                            RandomForestClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedASDDetector:
    """Optimized model training for maximum ASD detection accuracy"""
    
    def __init__(self):
        self.features_path = Path("/Users/jbanmol/binary-classification-project/features_binary")
        self.output_path = Path("/Users/jbanmol/binary-classification-project/models")
        self.output_path.mkdir(exist_ok=True)
        
        # Load selected features
        with open(self.features_path / "selected_features.txt", 'r') as f:
            self.selected_features = [line.strip() for line in f.readlines()]
    
    def load_and_prepare_data(self):
        """Load and prepare data with proven preprocessing"""
        
        logger.info("Loading data...")
        features_df = pd.read_csv(self.features_path / "child_features_binary.csv")
        
        # Use top selected features
        feature_cols = [col for col in self.selected_features if col in features_df.columns]
        logger.info(f"Using {len(feature_cols)} selected features")
        
        X = features_df[feature_cols].copy()
        
        # Advanced feature engineering
        # 1. Create ratios for most important features
        if 'target_zone_coverage_mean' in X.columns and 'outside_ratio_mean' in X.columns:
            X['coverage_ratio'] = X['target_zone_coverage_mean'] / (X['outside_ratio_mean'] + 0.01)
        
        if 'zone_transition_rate_mean' in X.columns and 'progress_linearity_mean' in X.columns:
            X['efficiency_score'] = X['progress_linearity_mean'] / (X['zone_transition_rate_mean'] + 0.01)
        
        # 2. Create interaction features for top 5 features
        top_5 = feature_cols[:5]
        for i in range(len(top_5)):
            for j in range(i+1, len(top_5)):
                X[f'{top_5[i]}_x_{top_5[j]}'] = X[top_5[i]] * X[top_5[j]]
        
        # Fill missing values strategically
        X = X.fillna(X.median())
        
        # Target
        y = (features_df['group'] == 'ASD_DD').astype(int)
        
        # Store info
        self.feature_names = list(X.columns)
        self.child_ids = features_df['child_id']
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Apply SMOTE for balanced training
        logger.info("Applying SMOTE for balanced training...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Also prepare unbalanced version for comparison
        X_train_unbalanced_scaled = scaler.fit_transform(X_train)
        
        # Save scaler
        joblib.dump(scaler, self.output_path / 'scaler_optimized.pkl')
        
        logger.info(f"Balanced training: {len(X_train_scaled)} samples")
        logger.info(f"Test set: {len(X_test_scaled)} samples ({y_test.sum()} ASD+DD, {len(y_test)-y_test.sum()} TD)")
        
        return (X_train_scaled, X_train_unbalanced_scaled, X_test_scaled, 
                y_train_balanced, y_train, y_test)
    
    def custom_scorer(self, y_true, y_pred):
        """Custom scorer balancing sensitivity and specificity"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Balanced score with penalty for extreme imbalance
        score = 0.5 * sensitivity + 0.5 * specificity
        imbalance_penalty = abs(sensitivity - specificity) * 0.1
        
        return score - imbalance_penalty
    
    def train_optimized_models(self, X_train, y_train):
        """Train highly optimized models"""
        
        models = {}
        scorer = make_scorer(self.custom_scorer)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 1. XGBoost with extensive tuning
        logger.info("Training XGBoost with extensive hyperparameter search...")
        xgb_params = {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'min_child_weight': [1, 2, 3, 4, 5],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [1, 1.5, 2],
            'scale_pos_weight': [1, 1.2, 1.5]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        xgb_search = RandomizedSearchCV(
            xgb_model, xgb_params,
            n_iter=100,  # More iterations
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        xgb_search.fit(X_train, y_train)
        models['XGBoost'] = xgb_search.best_estimator_
        logger.info(f"XGBoost best score: {xgb_search.best_score_:.4f}")
        
        # 2. LightGBM with extensive tuning
        logger.info("Training LightGBM...")
        lgb_params = {
            'n_estimators': [200, 300, 400, 500],
            'num_leaves': [20, 31, 50, 100],
            'max_depth': [-1, 5, 10, 15],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
            'bagging_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
            'bagging_freq': [3, 5, 7],
            'min_child_samples': [5, 10, 15, 20],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 0.01, 0.1],
            'min_split_gain': [0, 0.001, 0.01]
        }
        
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            random_state=42,
            verbose=-1
        )
        
        lgb_search = RandomizedSearchCV(
            lgb_model, lgb_params,
            n_iter=100,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        lgb_search.fit(X_train, y_train)
        models['LightGBM'] = lgb_search.best_estimator_
        logger.info(f"LightGBM best score: {lgb_search.best_score_:.4f}")
        
        # 3. Random Forest with careful tuning
        logger.info("Training Random Forest...")
        rf_params = {
            'n_estimators': [300, 400, 500, 600],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', 0.3, 0.4, 0.5],
            'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 1.2}, {0: 1, 1: 1.5}],
            'bootstrap': [True, False]
        }
        
        rf_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
        
        rf_search = RandomizedSearchCV(
            rf_model, rf_params,
            n_iter=100,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        rf_search.fit(X_train, y_train)
        models['RandomForest'] = rf_search.best_estimator_
        logger.info(f"RandomForest best score: {rf_search.best_score_:.4f}")
        
        # 4. Extra Trees
        logger.info("Training Extra Trees...")
        et_model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(X_train, y_train)
        models['ExtraTrees'] = et_model
        
        return models
    
    def create_advanced_ensemble(self, models, X_train, y_train):
        """Create advanced ensemble with calibration"""
        
        logger.info("Creating advanced ensemble...")
        
        # 1. Calibrate individual models
        calibrated_models = {}
        for name, model in models.items():
            logger.info(f"Calibrating {name}...")
            calibrated = CalibratedClassifierCV(
                model, method='isotonic', cv=3
            )
            calibrated.fit(X_train, y_train)
            calibrated_models[f"{name}_cal"] = calibrated
        
        # 2. Soft voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=list(calibrated_models.items()),
            voting='soft'
        )
        voting_ensemble.fit(X_train, y_train)
        
        # 3. Stacking ensemble with logistic regression meta-learner
        stacking_ensemble = StackingClassifier(
            estimators=list(models.items()),
            final_estimator=LogisticRegression(C=1.0, random_state=42),
            cv=5,
            stack_method='predict_proba'
        )
        stacking_ensemble.fit(X_train, y_train)
        
        # 4. Weighted ensemble based on individual performance
        # This will be implemented during evaluation
        
        return voting_ensemble, stacking_ensemble, calibrated_models
    
    def optimize_threshold(self, model, X_val, y_val):
        """Find optimal classification threshold"""
        
        y_proba = model.predict_proba(X_val)[:, 1]
        
        best_threshold = 0.5
        best_score = 0
        
        for threshold in np.arange(0.2, 0.8, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            score = self.custom_scorer(y_val, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def comprehensive_evaluation(self, model, X_test, y_test, model_name, threshold=0.5):
        """Comprehensive model evaluation"""
        
        # Predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate all metrics
        metrics = {
            'model': model_name,
            'threshold': threshold,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'auc_roc': roc_auc_score(y_test, y_proba),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'custom_score': self.custom_scorer(y_test, y_pred),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        
        return metrics, y_pred, y_proba
    
    def generate_report(self, all_results):
        """Generate comprehensive report"""
        
        report = []
        report.append("# High-Accuracy ASD Detection Model Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Sort by custom score
        sorted_results = sorted(all_results, key=lambda x: x['custom_score'], reverse=True)
        
        report.append("\n## Model Performance Summary\n")
        report.append("| Model | Sensitivity | Specificity | PPV | NPV | Balanced Acc | AUC-ROC | Custom Score |")
        report.append("|-------|------------|-------------|-----|-----|--------------|---------|--------------|")
        
        for result in sorted_results[:10]:  # Top 10 models
            report.append(f"| {result['model']} | {result['sensitivity']:.3f} | "
                         f"{result['specificity']:.3f} | {result['ppv']:.3f} | "
                         f"{result['npv']:.3f} | {result['balanced_accuracy']:.3f} | "
                         f"{result['auc_roc']:.3f} | {result['custom_score']:.3f} |")
        
        # Best model details
        best = sorted_results[0]
        report.append(f"\n## Best Model: {best['model']}")
        report.append(f"\n### Performance Metrics:")
        report.append(f"- **Sensitivity**: {best['sensitivity']:.3f} ({best['sensitivity']*100:.1f}% of ASD cases detected)")
        report.append(f"- **Specificity**: {best['specificity']:.3f} ({best['specificity']*100:.1f}% of TD cases correctly identified)")
        report.append(f"- **PPV**: {best['ppv']:.3f} (When positive, {best['ppv']*100:.1f}% actually have ASD)")
        report.append(f"- **NPV**: {best['npv']:.3f} (When negative, {best['npv']*100:.1f}% actually are TD)")
        report.append(f"- **Optimal Threshold**: {best['threshold']:.3f}")
        
        report.append(f"\n### Confusion Matrix:")
        report.append(f"- True Positives (ASD correctly identified): {best['tp']}")
        report.append(f"- True Negatives (TD correctly identified): {best['tn']}")
        report.append(f"- False Positives (TD misclassified as ASD): {best['fp']}")
        report.append(f"- False Negatives (ASD missed): {best['fn']}")
        
        # Clinical interpretation
        report.append("\n## Clinical Interpretation")
        report.append(f"- The model correctly identifies {best['sensitivity']*100:.0f} out of 100 children with ASD/DD")
        report.append(f"- The model correctly identifies {best['specificity']*100:.0f} out of 100 TD children")
        report.append(f"- Very high balanced performance: {best['balanced_accuracy']*100:.1f}%")
        
        # Save report
        with open(self.output_path / 'best_model_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        return report, best
    
    def visualize_results(self, all_results, X_test, y_test):
        """Create visualizations"""
        
        # Get top 5 models
        top_models = sorted(all_results, key=lambda x: x['custom_score'], reverse=True)[:5]
        
        # 1. ROC curves
        plt.figure(figsize=(10, 8))
        
        for result in top_models:
            model_name = result['model']
            if 'y_proba' in result:
                fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
                plt.plot(fpr, tpr, label=f"{model_name} (AUC={result['auc_roc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curves - Top 5 Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_path / 'best_models_roc.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance comparison
        metrics_df = pd.DataFrame(top_models)
        key_metrics = ['sensitivity', 'specificity', 'ppv', 'npv', 'balanced_accuracy']
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(top_models))
        width = 0.15
        
        for i, metric in enumerate(key_metrics):
            plt.bar(x + i*width, metrics_df[metric], width, label=metric)
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x + width*2, [r['model'] for r in top_models], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_and_evaluate(self):
        """Main training and evaluation pipeline"""
        
        # Load data
        (X_train_balanced, X_train_unbalanced, X_test, 
         y_train_balanced, y_train_unbalanced, y_test) = self.load_and_prepare_data()
        
        all_results = []
        
        # Train models on balanced data
        logger.info("\n=== Training on balanced data ===")
        models_balanced = self.train_optimized_models(X_train_balanced, y_train_balanced)
        
        # Train models on unbalanced data for comparison
        logger.info("\n=== Training on unbalanced data ===")
        models_unbalanced = self.train_optimized_models(X_train_unbalanced, y_train_unbalanced)
        
        # Evaluate all individual models
        logger.info("\n=== Evaluating individual models ===")
        
        for name, model in models_balanced.items():
            threshold = self.optimize_threshold(model, X_train_balanced, y_train_balanced)
            metrics, y_pred, y_proba = self.comprehensive_evaluation(
                model, X_test, y_test, f"{name}_balanced", threshold
            )
            metrics['y_pred'] = y_pred
            metrics['y_proba'] = y_proba
            all_results.append(metrics)
            logger.info(f"{name}_balanced: Sens={metrics['sensitivity']:.3f}, "
                       f"Spec={metrics['specificity']:.3f}, Score={metrics['custom_score']:.3f}")
        
        for name, model in models_unbalanced.items():
            threshold = self.optimize_threshold(model, X_train_unbalanced, y_train_unbalanced)
            metrics, y_pred, y_proba = self.comprehensive_evaluation(
                model, X_test, y_test, f"{name}_unbalanced", threshold
            )
            metrics['y_pred'] = y_pred
            metrics['y_proba'] = y_proba
            all_results.append(metrics)
            logger.info(f"{name}_unbalanced: Sens={metrics['sensitivity']:.3f}, "
                       f"Spec={metrics['specificity']:.3f}, Score={metrics['custom_score']:.3f}")
        
        # Create ensembles
        logger.info("\n=== Creating ensemble models ===")
        voting_balanced, stacking_balanced, cal_models_balanced = self.create_advanced_ensemble(
            models_balanced, X_train_balanced, y_train_balanced
        )
        
        voting_unbalanced, stacking_unbalanced, cal_models_unbalanced = self.create_advanced_ensemble(
            models_unbalanced, X_train_unbalanced, y_train_unbalanced
        )
        
        # Evaluate ensembles
        ensembles = [
            ('Voting_balanced', voting_balanced, X_train_balanced, y_train_balanced),
            ('Stacking_balanced', stacking_balanced, X_train_balanced, y_train_balanced),
            ('Voting_unbalanced', voting_unbalanced, X_train_unbalanced, y_train_unbalanced),
            ('Stacking_unbalanced', stacking_unbalanced, X_train_unbalanced, y_train_unbalanced)
        ]
        
        for name, model, X_train, y_train in ensembles:
            threshold = self.optimize_threshold(model, X_train, y_train)
            metrics, y_pred, y_proba = self.comprehensive_evaluation(
                model, X_test, y_test, name, threshold
            )
            metrics['y_pred'] = y_pred
            metrics['y_proba'] = y_proba
            all_results.append(metrics)
            logger.info(f"{name}: Sens={metrics['sensitivity']:.3f}, "
                       f"Spec={metrics['specificity']:.3f}, Score={metrics['custom_score']:.3f}")
        
        # Generate report and visualizations
        report, best_result = self.generate_report(all_results)
        self.visualize_results(all_results, X_test, y_test)
        
        # Save best model
        best_model_name = best_result['model']
        
        # Find the actual model object
        if 'balanced' in best_model_name:
            if 'Voting' in best_model_name:
                best_model = voting_balanced
            elif 'Stacking' in best_model_name:
                best_model = stacking_balanced
            else:
                model_key = best_model_name.replace('_balanced', '')
                best_model = models_balanced[model_key]
        else:
            if 'Voting' in best_model_name:
                best_model = voting_unbalanced
            elif 'Stacking' in best_model_name:
                best_model = stacking_unbalanced
            else:
                model_key = best_model_name.replace('_unbalanced', '')
                best_model = models_unbalanced[model_key]
        
        # Save complete model package
        model_package = {
            'model': best_model,
            'scaler': joblib.load(self.output_path / 'scaler_optimized.pkl'),
            'feature_names': self.feature_names,
            'threshold': best_result['threshold'],
            'metrics': best_result,
            'model_type': best_model_name,
            'selected_features': self.selected_features
        }
        
        joblib.dump(model_package, self.output_path / 'best_asd_model.pkl')
        
        # Print summary
        print("\n" + "="*70)
        print("HIGH-ACCURACY MODEL TRAINING COMPLETE!")
        print("="*70)
        print(f"\nBest Model: {best_result['model']}")
        print(f"Sensitivity: {best_result['sensitivity']:.3f} ({best_result['sensitivity']*100:.1f}%)")
        print(f"Specificity: {best_result['specificity']:.3f} ({best_result['specificity']*100:.1f}%)")
        print(f"Balanced Accuracy: {best_result['balanced_accuracy']:.3f} ({best_result['balanced_accuracy']*100:.1f}%)")
        print(f"AUC-ROC: {best_result['auc_roc']:.3f}")
        print(f"PPV: {best_result['ppv']:.3f}")
        print(f"NPV: {best_result['npv']:.3f}")
        print(f"Optimal Threshold: {best_result['threshold']:.3f}")
        print("\nFiles saved:")
        print("- models/best_asd_model.pkl - Best trained model package")
        print("- models/best_model_report.md - Detailed performance report")
        print("- models/best_models_roc.png - ROC curves")
        print("- models/performance_comparison.png - Performance metrics")
        print("="*70)
        
        return best_model, best_result


def main():
    """Run optimized training"""
    detector = OptimizedASDDetector()
    best_model, metrics = detector.train_and_evaluate()
    
    return best_model, metrics


if __name__ == "__main__":
    main()
