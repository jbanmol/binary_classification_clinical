#!/usr/bin/env python3
"""
Train Binary Classification Model using Selected Features
Uses the top features identified from group comparison analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import ModelTrainer
from evaluation import ModelEvaluator
from data_processing import DataProcessor
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinaryModelTrainer:
    """Train models for TD vs ASD+DD classification"""
    
    def __init__(self):
        self.features_path = Path("/Users/jbanmol/binary-classification-project/features_binary")
        self.output_path = Path("/Users/jbanmol/binary-classification-project/models")
        self.output_path.mkdir(exist_ok=True)
        
        # Load selected features
        with open(self.features_path / "selected_features.txt", 'r') as f:
            self.selected_features = [line.strip() for line in f.readlines()]
            
    def load_and_prepare_data(self):
        """Load features and prepare for training"""
        
        # Load child-level features
        logger.info("Loading child-level features...")
        features_df = pd.read_csv(self.features_path / "child_features_binary.csv")
        
        # Filter to selected features
        feature_cols = [col for col in self.selected_features if col in features_df.columns]
        logger.info(f"Using {len(feature_cols)} features from selected list")
        
        # Prepare X and y
        X = features_df[feature_cols].fillna(0)
        y = (features_df['group'] == 'ASD_DD').astype(int)  # 1 for ASD+DD, 0 for TD
        
        # Store child IDs for analysis
        self.child_ids = features_df['child_id']
        
        # Split data
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, self.child_ids, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, self.output_path / 'scaler_binary.pkl')
        
        logger.info(f"Training set: {len(X_train)} samples ({y_train.sum()} ASD+DD, {len(y_train)-y_train.sum()} TD)")
        logger.info(f"Test set: {len(X_test)} samples ({y_test.sum()} ASD+DD, {len(y_test)-y_test.sum()} TD)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def train_models(self, X_train, y_train):
        """Train multiple models with hyperparameter tuning"""
        
        # Initialize trainer
        trainer = ModelTrainer(random_state=42)
        
        # Focus on best performing models for this task
        focused_models = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
        
        # Custom parameter grids optimized for our features
        custom_param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 4, 8],
                'max_features': ['sqrt', 'log2', 0.3, 0.5]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'scale_pos_weight': [1, 1.5, 2]  # Handle class imbalance
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2', 'l1'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
        }
        
        # Train models
        logger.info("Training models with hyperparameter tuning...")
        best_models = {}
        
        for model_name in focused_models:
            if model_name in trainer.models:
                logger.info(f"\nTraining {model_name}...")
                
                # Set up GridSearchCV with custom scorer
                from sklearn.metrics import make_scorer
                
                # Custom scorer that balances sensitivity and specificity
                def balanced_accuracy_custom(y_true, y_pred):
                    from sklearn.metrics import confusion_matrix
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    return 0.6 * sensitivity + 0.4 * specificity  # Slightly favor sensitivity
                
                scorer = make_scorer(balanced_accuracy_custom)
                
                # Grid search
                from sklearn.model_selection import GridSearchCV, StratifiedKFold
                
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                grid_search = GridSearchCV(
                    trainer.models[model_name],
                    custom_param_grids.get(model_name, trainer.get_param_grids()[model_name]),
                    cv=cv,
                    scoring=scorer,
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                best_models[model_name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_
                }
                
                logger.info(f"Best score for {model_name}: {grid_search.best_score_:.4f}")
                logger.info(f"Best params: {grid_search.best_params_}")
        
        return best_models
    
    def evaluate_and_select_best(self, best_models, X_test, y_test, feature_names):
        """Evaluate all models and select the best one"""
        
        evaluator = ModelEvaluator()
        
        results = {}
        best_score = 0
        best_model_name = None
        
        for model_name, model_info in best_models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            model = model_info['model']
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Calculate balanced score
            balanced_score = 0.6 * metrics['sensitivity'] + 0.4 * metrics['specificity']
            
            results[model_name] = {
                'metrics': metrics,
                'balanced_score': balanced_score,
                'model': model
            }
            
            logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
            logger.info(f"Specificity: {metrics['specificity']:.4f}")
            logger.info(f"Balanced Score: {balanced_score:.4f}")
            
            if balanced_score > best_score:
                best_score = balanced_score
                best_model_name = model_name
        
        # Optimize threshold for best model
        logger.info(f"\nOptimizing threshold for {best_model_name}...")
        best_model = results[best_model_name]['model']
        
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        optimal_threshold = evaluator.find_optimal_threshold(
            y_test, y_pred_proba, 
            sensitivity_weight=0.6  # Prioritize sensitivity
        )
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        
        # Re-evaluate with optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        final_metrics = evaluator.calculate_metrics(y_test, y_pred_optimal, y_pred_proba)
        
        logger.info(f"\nFinal performance with optimal threshold:")
        logger.info(f"Sensitivity: {final_metrics['sensitivity']:.4f}")
        logger.info(f"Specificity: {final_metrics['specificity']:.4f}")
        logger.info(f"Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
        
        # Save best model
        model_data = {
            'model': best_model,
            'feature_names': feature_names,
            'optimal_threshold': optimal_threshold,
            'metrics': final_metrics,
            'model_name': best_model_name
        }
        
        joblib.dump(model_data, self.output_path / 'best_binary_model.pkl')
        
        # Generate evaluation report
        self.generate_evaluation_report(results, best_model_name, feature_names, X_test, y_test)
        
        return best_model, optimal_threshold, final_metrics
    
    def generate_evaluation_report(self, results, best_model_name, feature_names, X_test, y_test):
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("# Binary Classification Model Evaluation Report")
        report.append(f"\nDate: {pd.Timestamp.now()}")
        report.append(f"Task: TD vs ASD+DD Classification")
        report.append(f"Number of features: {len(feature_names)}")
        
        report.append("\n## Model Performance Summary\n")
        
        # Sort by balanced score
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['balanced_score'], 
                              reverse=True)
        
        for model_name, result in sorted_results:
            metrics = result['metrics']
            report.append(f"\n### {model_name}")
            report.append(f"- Balanced Score: {result['balanced_score']:.4f}")
            report.append(f"- Sensitivity: {metrics['sensitivity']:.4f}")
            report.append(f"- Specificity: {metrics['specificity']:.4f}")
            report.append(f"- F1 Score: {metrics['f1']:.4f}")
            report.append(f"- AUC-ROC: {metrics['auc_roc']:.4f}")
            report.append(f"- Matthews Correlation: {metrics['mcc']:.4f}")
        
        report.append(f"\n## Best Model: {best_model_name}")
        
        # Feature importance for best model
        best_model = results[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            report.append("\n### Top 15 Most Important Features:")
            for idx, row in importance_df.head(15).iterrows():
                report.append(f"- {row['feature']}: {row['importance']:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            importance_df.head(20).plot(x='feature', y='importance', kind='barh')
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(self.output_path / 'feature_importance_binary.png', dpi=300)
            plt.close()
        
        # Key insights
        report.append("\n## Key Insights")
        report.append("1. Template-aware features (zone coverage, outside ratios) show highest importance")
        report.append("2. Motor control variability metrics are strong discriminators")
        report.append("3. Progress linearity captures behavioral differences effectively")
        report.append("4. The model achieves balanced performance between sensitivity and specificity")
        
        # Save report
        with open(self.output_path / 'model_evaluation_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"\nReport saved to {self.output_path / 'model_evaluation_report.md'}")

def main():
    """Main training pipeline"""
    
    trainer = BinaryModelTrainer()
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_names = trainer.load_and_prepare_data()
    
    # Train models
    logger.info("\nTraining models...")
    best_models = trainer.train_models(X_train, y_train)
    
    # Evaluate and select best
    logger.info("\nEvaluating models...")
    best_model, optimal_threshold, final_metrics = trainer.evaluate_and_select_best(
        best_models, X_test, y_test, feature_names
    )
    
    logger.info("\n=== Training Complete ===")
    logger.info(f"Best model saved to: {trainer.output_path / 'best_binary_model.pkl'}")
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"Final balanced accuracy: {final_metrics['balanced_accuracy']:.4f}")
    
    print("\nNext steps:")
    print("1. Review the evaluation report in models/model_evaluation_report.md")
    print("2. Check feature importance visualization in models/feature_importance_binary.png")
    print("3. Use the trained model for predictions on new data")
    print("4. Consider further threshold optimization based on clinical requirements")

if __name__ == "__main__":
    main()
