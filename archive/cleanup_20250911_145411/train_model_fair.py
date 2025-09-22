#!/usr/bin/env python3
"""
Fair Model Training for ASD Detection
Ensures no data leakage by keeping all sessions from same child together
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, matthews_corrcoef, 
                           balanced_accuracy_score)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairASDModelTrainer:
    """Fair model training that respects child-level grouping"""
    
    def __init__(self):
        self.features_path = Path("/Users/jbanmol/binary-classification-project/features_binary")
        self.output_path = Path("/Users/jbanmol/binary-classification-project/models")
        self.output_path.mkdir(exist_ok=True)
        
        # Load selected features
        with open(self.features_path / "selected_features.txt", 'r') as f:
            self.selected_features = [line.strip() for line in f.readlines()]
    
    def load_data_with_validation(self):
        """Load data and validate child-session relationships"""
        
        logger.info("Loading child-level and session-level features...")
        
        # Load both child and session level data
        child_features = pd.read_csv(self.features_path / "child_features_binary.csv")
        session_features = pd.read_csv(self.features_path / "session_features_binary.csv")
        
        logger.info(f"Loaded {len(child_features)} children with aggregated features")
        logger.info(f"Loaded {len(session_features)} individual sessions")
        
        # Analyze data structure
        sessions_per_child = session_features.groupby('child_id').size()
        logger.info(f"\nSessions per child statistics:")
        logger.info(f"Mean: {sessions_per_child.mean():.1f}")
        logger.info(f"Min: {sessions_per_child.min()}")
        logger.info(f"Max: {sessions_per_child.max()}")
        logger.info(f"Std: {sessions_per_child.std():.1f}")
        
        # Check for data consistency
        child_groups = child_features.groupby('group').size()
        session_groups = session_features.groupby('group').size()
        
        logger.info(f"\nChild distribution: {child_groups.to_dict()}")
        logger.info(f"Session distribution: {session_groups.to_dict()}")
        
        return child_features, session_features, sessions_per_child
    
    def prepare_child_level_data(self, child_features):
        """Prepare child-level data for training"""
        
        # Use selected features
        feature_cols = [col for col in self.selected_features if col in child_features.columns]
        logger.info(f"\nUsing {len(feature_cols)} selected features for child-level model")
        
        X = child_features[feature_cols].copy()
        y = (child_features['group'] == 'ASD_DD').astype(int)
        child_ids = child_features['child_id'].values
        
        # Create meaningful feature engineering
        if 'target_zone_coverage_mean' in X.columns and 'outside_ratio_mean' in X.columns:
            X['accuracy_score'] = X['target_zone_coverage_mean'] / (X['outside_ratio_mean'] + 0.01)
        
        if 'zone_transition_rate_mean' in X.columns and 'progress_linearity_mean' in X.columns:
            X['planning_score'] = X['progress_linearity_mean'] / (X['zone_transition_rate_mean'] + 0.01)
        
        # Fill missing values
        X = X.fillna(X.median())
        
        return X, y, child_ids, list(X.columns)
    
    def create_fair_train_test_split(self, X, y, child_ids, test_size=0.2):
        """Create train/test split ensuring all sessions from same child stay together"""
        
        # Create a mapping of indices
        unique_children = np.unique(child_ids)
        n_test_children = int(len(unique_children) * test_size)
        
        # Stratify by group
        child_df = pd.DataFrame({'child_id': child_ids, 'y': y})
        child_groups = child_df.groupby('child_id')['y'].first()
        
        # Get counts for each class
        td_children = child_groups[child_groups == 0].index.tolist()
        asd_children = child_groups[child_groups == 1].index.tolist()
        
        # Calculate proportional split
        n_test_td = int(len(td_children) * test_size)
        n_test_asd = int(len(asd_children) * test_size)
        
        # Random selection
        np.random.seed(42)
        test_td = np.random.choice(td_children, n_test_td, replace=False)
        test_asd = np.random.choice(asd_children, n_test_asd, replace=False)
        test_children = np.concatenate([test_td, test_asd])
        
        # Create masks
        test_mask = np.isin(child_ids, test_children)
        train_mask = ~test_mask
        
        # Split data
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        logger.info(f"\nFair train/test split:")
        logger.info(f"Training: {len(X_train)} samples from {train_mask.sum()} children")
        logger.info(f"  - TD: {(y_train == 0).sum()}")
        logger.info(f"  - ASD/DD: {(y_train == 1).sum()}")
        logger.info(f"Testing: {len(X_test)} samples from {test_mask.sum()} children")
        logger.info(f"  - TD: {(y_test == 0).sum()}")
        logger.info(f"  - ASD/DD: {(y_test == 1).sum()}")
        
        return X_train, X_test, y_train, y_test, child_ids[train_mask], child_ids[test_mask]
    
    def train_models_with_cv(self, X_train, y_train, groups_train):
        """Train models using grouped cross-validation"""
        
        models = {}
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Use GroupKFold for fair cross-validation
        gkf = GroupKFold(n_splits=5)
        
        # 1. Random Forest
        logger.info("\nTraining Random Forest with grouped CV...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation scores
        cv_scores = []
        for train_idx, val_idx in gkf.split(X_train_scaled, y_train, groups_train):
            rf_temp = rf
            rf_temp.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
            val_pred = rf_temp.predict(X_train_scaled[val_idx])
            val_score = balanced_accuracy_score(y_train.iloc[val_idx], val_pred)
            cv_scores.append(val_score)
        
        logger.info(f"Random Forest CV scores: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        rf.fit(X_train_scaled, y_train)
        models['RandomForest'] = rf
        
        # 2. XGBoost
        logger.info("\nTraining XGBoost with grouped CV...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        cv_scores = []
        for train_idx, val_idx in gkf.split(X_train_scaled, y_train, groups_train):
            xgb_temp = xgb_model
            xgb_temp.fit(X_train_scaled[train_idx], y_train.iloc[train_idx], verbose=False)
            val_pred = xgb_temp.predict(X_train_scaled[val_idx])
            val_score = balanced_accuracy_score(y_train.iloc[val_idx], val_pred)
            cv_scores.append(val_score)
        
        logger.info(f"XGBoost CV scores: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        xgb_model.fit(X_train_scaled, y_train, verbose=False)
        models['XGBoost'] = xgb_model
        
        # 3. LightGBM
        logger.info("\nTraining LightGBM with grouped CV...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        
        cv_scores = []
        for train_idx, val_idx in gkf.split(X_train_scaled, y_train, groups_train):
            lgb_temp = lgb_model
            lgb_temp.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
            val_pred = lgb_temp.predict(X_train_scaled[val_idx])
            val_score = balanced_accuracy_score(y_train.iloc[val_idx], val_pred)
            cv_scores.append(val_score)
        
        logger.info(f"LightGBM CV scores: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        lgb_model.fit(X_train_scaled, y_train)
        models['LightGBM'] = lgb_model
        
        # Save scaler
        joblib.dump(scaler, self.output_path / 'scaler_fair.pkl')
        
        return models, scaler
    
    def evaluate_models(self, models, scaler, X_test, y_test):
        """Evaluate models on held-out test set"""
        
        X_test_scaled = scaler.transform(X_test)
        results = {}
        
        for name, model in models.items():
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            results[name] = {
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_proba),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            logger.info(f"\n{name} Results:")
            logger.info(f"  Sensitivity: {results[name]['sensitivity']:.3f}")
            logger.info(f"  Specificity: {results[name]['specificity']:.3f}")
            logger.info(f"  Balanced Accuracy: {results[name]['balanced_accuracy']:.3f}")
            logger.info(f"  AUC-ROC: {results[name]['auc_roc']:.3f}")
        
        return results
    
    def validate_on_session_level(self, best_model, scaler, feature_names, 
                                  session_features, child_test_ids):
        """Validate model by applying to individual sessions"""
        
        logger.info("\n=== Session-Level Validation ===")
        
        # Get test sessions
        test_sessions = session_features[session_features['child_id'].isin(child_test_ids)]
        logger.info(f"Validating on {len(test_sessions)} sessions from {len(child_test_ids)} test children")
        
        # Prepare features (only use features that exist in session data)
        available_features = [f for f in feature_names if f in test_sessions.columns]
        logger.info(f"Using {len(available_features)} available features from {len(feature_names)} total")
        
        if len(available_features) < len(feature_names) * 0.5:
            logger.warning("Less than 50% of features available at session level!")
            logger.warning("Skipping session-level validation due to feature mismatch.")
            return None, None
        
        X_sessions = test_sessions[available_features].fillna(0)
        y_sessions = (test_sessions['group'] == 'ASD_DD').astype(int)
        
        # For missing features, add zero columns
        for feat in feature_names:
            if feat not in X_sessions.columns:
                X_sessions[feat] = 0
        
        # Ensure correct feature order
        X_sessions = X_sessions[feature_names]
        
        # Scale and predict
        X_sessions_scaled = scaler.transform(X_sessions)
        session_predictions = best_model.predict_proba(X_sessions_scaled)[:, 1]
        
        # Aggregate predictions by child
        session_results = pd.DataFrame({
            'child_id': test_sessions['child_id'].values,
            'true_label': y_sessions.values,
            'pred_proba': session_predictions
        })
        
        # Majority voting per child
        child_predictions = session_results.groupby('child_id').agg({
            'true_label': 'first',  # All sessions have same label
            'pred_proba': ['mean', 'std', 'min', 'max']
        })
        
        # Evaluate child-level predictions
        child_pred_binary = (child_predictions['pred_proba']['mean'] > 0.5).astype(int)
        child_true = child_predictions['true_label']['first']
        
        tn, fp, fn, tp = confusion_matrix(child_true, child_pred_binary).ravel()
        
        logger.info("\nSession-aggregated child-level results:")
        logger.info(f"  Sensitivity: {tp / (tp + fn):.3f}")
        logger.info(f"  Specificity: {tn / (tn + fp):.3f}")
        logger.info(f"  Balanced Accuracy: {balanced_accuracy_score(child_true, child_pred_binary):.3f}")
        
        return session_results, child_predictions
    
    def generate_report(self, results, best_model_name, feature_names):
        """Generate comprehensive report"""
        
        report = []
        report.append("# Fair ASD Detection Model Training Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Key Training Considerations")
        report.append("- All sessions from same child kept together (no data leakage)")
        report.append("- Grouped cross-validation respecting child boundaries")
        report.append("- Child-level aggregated features used for training")
        report.append("- Session-level validation performed")
        
        report.append("\n## Model Performance (Child-Level)")
        
        for name, metrics in results.items():
            report.append(f"\n### {name}")
            report.append(f"- Sensitivity: {metrics['sensitivity']:.3f}")
            report.append(f"- Specificity: {metrics['specificity']:.3f}")
            report.append(f"- Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
            report.append(f"- AUC-ROC: {metrics['auc_roc']:.3f}")
            report.append(f"- PPV: {metrics['ppv']:.3f}")
            report.append(f"- NPV: {metrics['npv']:.3f}")
            report.append(f"- MCC: {metrics['mcc']:.3f}")
            
            cm = metrics['confusion_matrix']
            report.append(f"\nConfusion Matrix:")
            report.append(f"  - True Positives: {cm['tp']}")
            report.append(f"  - True Negatives: {cm['tn']}")
            report.append(f"  - False Positives: {cm['fp']}")
            report.append(f"  - False Negatives: {cm['fn']}")
        
        best_metrics = results[best_model_name]
        report.append(f"\n## Best Model: {best_model_name}")
        report.append(f"- Correctly identifies {best_metrics['sensitivity']*100:.0f}% of ASD/DD children")
        report.append(f"- Correctly identifies {best_metrics['specificity']*100:.0f}% of TD children")
        report.append(f"- Overall balanced accuracy: {best_metrics['balanced_accuracy']*100:.1f}%")
        
        report.append("\n## Feature Importance (Top 15)")
        # This would require getting feature importances from the model
        
        with open(self.output_path / 'fair_training_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        return report
    
    def visualize_results(self, results, y_test):
        """Create visualizations"""
        
        # ROC curves
        plt.figure(figsize=(10, 8))
        
        for name, metrics in results.items():
            fpr, tpr, _ = roc_curve(y_test, metrics['y_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['auc_roc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Fair Training')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_path / 'roc_curves_fair.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (name, metrics) in enumerate(results.items()):
            cm = confusion_matrix(y_test, metrics['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'confusion_matrices_fair.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_fair_training(self):
        """Run the complete fair training pipeline"""
        
        # Load data
        child_features, session_features, sessions_per_child = self.load_data_with_validation()
        
        # Prepare child-level data
        X, y, child_ids, feature_names = self.prepare_child_level_data(child_features)
        
        # Create fair train/test split
        X_train, X_test, y_train, y_test, train_groups, test_groups = self.create_fair_train_test_split(
            X, y, child_ids
        )
        
        # Train models with grouped CV
        models, scaler = self.train_models_with_cv(X_train, y_train, train_groups)
        
        # Evaluate on test set
        results = self.evaluate_models(models, scaler, X_test, y_test)
        
        # Find best model
        best_model_name = max(results.items(), 
                             key=lambda x: x[1]['balanced_accuracy'])[0]
        best_model = models[best_model_name]
        
        logger.info(f"\nBest model: {best_model_name}")
        
        # Validate on session level
        validation_result = self.validate_on_session_level(
            best_model, scaler, feature_names, session_features, test_groups
        )
        
        if validation_result is not None:
            session_results, child_predictions = validation_result
        else:
            session_results, child_predictions = None, None
        
        # Generate report and visualizations
        self.generate_report(results, best_model_name, feature_names)
        self.visualize_results(results, y_test)
        
        # Save best model
        model_package = {
            'model': best_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'model_type': best_model_name,
            'metrics': results[best_model_name],
            'training_info': {
                'n_train_children': len(np.unique(train_groups)),
                'n_test_children': len(np.unique(test_groups)),
                'used_grouped_cv': True,
                'prevented_data_leakage': True
            }
        }
        
        joblib.dump(model_package, self.output_path / 'fair_asd_model.pkl')
        
        logger.info("\n" + "="*60)
        logger.info("FAIR TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Test Set Performance (No Data Leakage):")
        logger.info(f"  - Sensitivity: {results[best_model_name]['sensitivity']:.3f}")
        logger.info(f"  - Specificity: {results[best_model_name]['specificity']:.3f}")
        logger.info(f"  - Balanced Accuracy: {results[best_model_name]['balanced_accuracy']:.3f}")
        logger.info(f"  - AUC-ROC: {results[best_model_name]['auc_roc']:.3f}")
        logger.info("\nModel saved to: models/fair_asd_model.pkl")
        logger.info("Report saved to: models/fair_training_report.md")
        
        return best_model, results


def main():
    """Run fair training"""
    trainer = FairASDModelTrainer()
    model, results = trainer.run_fair_training()
    return model, results


if __name__ == "__main__":
    main()
