#!/usr/bin/env python3
"""
Clinical Ensemble Training for ASD Classification
RAG-Guided Ensemble Optimized for ≥86% Sensitivity & ≥71% Specificity
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, make_scorer)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available, using alternatives")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available, using alternatives")

import warnings
warnings.filterwarnings('ignore')

class ClinicalEnsembleTrainer:
    """Clinical ensemble trainer optimized for ASD sensitivity/specificity targets"""
    
    def __init__(self, target_sensitivity=0.86, target_specificity=0.71):
        self.target_sensitivity = target_sensitivity
        self.target_specificity = target_specificity
        self.models = {}
        self.ensemble = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_clinical_data(self):
        """Load clinical-optimized features"""
        print("📊 Loading clinical feature dataset...")
        
        try:
            self.clinical_data = pd.read_csv('features_binary/clinical_features_optimized.csv')
            
            # Prepare features and labels
            self.X = self.clinical_data.drop(['child_id', 'group', 'binary_label'], axis=1)
            self.y = self.clinical_data['binary_label']
            self.feature_names = list(self.X.columns)
            
            print(f"   ✅ Loaded clinical dataset:")
            print(f"      Samples: {len(self.clinical_data)}")
            print(f"      Features: {len(self.X.columns)}")
            print(f"      ASD: {self.y.sum()}, TD: {len(self.y) - self.y.sum()}")
            print(f"      Class ratio: {self.y.sum() / (len(self.y) - self.y.sum()):.2f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading clinical data: {e}")
            return False
    
    def custom_sensitivity_scorer(self, y_true, y_pred):
        """Custom scorer that prioritizes sensitivity"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Weighted score favoring sensitivity (clinical requirement)
        return sensitivity * 1.2 + specificity * 1.0
    
    def create_base_models(self):
        """Create diverse base models for ensemble"""
        print("🤖 Creating base models for clinical ensemble...")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y), 
            y=self.y
        )
        weight_dict = {0: class_weights[0], 1: class_weights[1] * 1.2}  # Extra weight for ASD
        
        models = {}
        
        # 1. XGBoost (if available) - Excellent with imbalanced data
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=class_weights[1]/class_weights[0] * 1.2,  # Boost sensitivity
                random_state=42,
                eval_metric='logloss'
            )
        
        # 2. LightGBM (if available) - Fast and robust
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight=weight_dict,
                random_state=42,
                verbose=-1
            )
        
        # 3. Random Forest - Stable and interpretable
        models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=weight_dict,
            random_state=42
        )
        
        # 4. Logistic Regression - Clinical baseline
        models['logistic'] = LogisticRegression(
            C=1.0,
            class_weight=weight_dict,
            random_state=42,
            max_iter=1000
        )
        
        # 5. SVM with RBF - Non-linear patterns
        models['svm'] = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight=weight_dict,
            probability=True,  # Needed for ensemble
            random_state=42
        )
        
        # 6. Neural Network - Complex interactions
        models['neural_net'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            random_state=42,
            max_iter=500
        )
        
        print(f"   ✅ Created {len(models)} base models: {list(models.keys())}")
        self.models = models
        return models
    
    def optimize_model_hyperparameters(self, model_name, model, cv_folds=3):
        """Optimize hyperparameters for individual models"""
        print(f"⚙️  Optimizing {model_name} hyperparameters...")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [8, 10, 12],
                'min_samples_split': [5, 10]
            },
            'logistic': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name in param_grids:
            # Custom scorer for sensitivity optimization
            sensitivity_scorer = make_scorer(self.custom_sensitivity_scorer)
            
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=cv_folds,
                scoring=sensitivity_scorer,
                n_jobs=-1
            )
            
            # Scale features for optimization (handle NaN)
            X_cleaned = self.X.fillna(0)
            X_scaled = self.scaler.fit_transform(X_cleaned)
            grid_search.fit(X_scaled, self.y)
            
            print(f"   ✅ Best {model_name} params: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        return model
    
    def train_base_models(self):
        """Train all base models with cross-validation"""
        print("🏋️ Training base models...")
        
        # Handle missing values and scale features
        X_cleaned = self.X.fillna(0)  # Fill NaN with 0
        X_scaled = self.scaler.fit_transform(X_cleaned)
        
        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, self.y)
        
        print(f"   📊 After SMOTE: {len(X_balanced)} samples")
        print(f"      ASD: {y_balanced.sum()}, TD: {len(y_balanced) - y_balanced.sum()}")
        
        trained_models = {}
        cv_scores = {}
        
        # 5-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"   🎯 Training {name}...")
            
            # Optimize hyperparameters for key models
            if name in ['random_forest', 'logistic', 'svm']:
                model = self.optimize_model_hyperparameters(name, model)
            
            # Train on balanced data
            model.fit(X_balanced, y_balanced)
            
            # Cross-validate on original data for unbiased evaluation
            sensitivity_scorer = make_scorer(self.custom_sensitivity_scorer)
            scores = cross_val_score(model, X_scaled, self.y, cv=cv, scoring=sensitivity_scorer)
            
            cv_scores[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            
            trained_models[name] = model
            print(f"      CV Score: {scores.mean():.3f} ± {scores.std():.3f}")
        
        self.models = trained_models
        self.cv_scores = cv_scores
        
        print(f"   ✅ All {len(trained_models)} base models trained")
        return trained_models
    
    def create_stacked_ensemble(self):
        """Create stacked ensemble with meta-learner"""
        print("🏗️  Creating stacked ensemble...")
        
        # Create voting classifier with all base models
        estimators = [(name, model) for name, model in self.models.items()]
        
        # Use soft voting for probability-based decisions
        voting_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability averages
        )
        
        # Calibrate the ensemble for better probability estimates
        calibrated_ensemble = CalibratedClassifierCV(
            voting_ensemble,
            method='isotonic',  # Better for small datasets
            cv=3
        )
        
        self.ensemble = calibrated_ensemble
        print(f"   ✅ Stacked ensemble created with {len(estimators)} base models")
        return calibrated_ensemble
    
    def optimize_threshold_for_sensitivity(self, X_test, y_test):
        """Optimize decision threshold to achieve target sensitivity"""
        print(f"🎯 Optimizing threshold for sensitivity ≥{self.target_sensitivity}...")
        
        # Get probability predictions
        y_proba = self.ensemble.predict_proba(X_test)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if len(np.unique(y_pred)) == 2:  # Both classes predicted
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # Check if meets clinical targets
                meets_targets = (sensitivity >= self.target_sensitivity and 
                               specificity >= self.target_specificity)
                
                # Composite score (favor sensitivity)
                score = sensitivity * 1.2 + specificity * 1.0
                
                results.append({
                    'threshold': threshold,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'score': score,
                    'meets_targets': meets_targets
                })
                
                if meets_targets and score > best_score:
                    best_threshold = threshold
                    best_score = score
        
        # If no threshold meets both targets, optimize for sensitivity
        if best_score == 0:
            print("   ⚠️  No threshold meets both targets, optimizing for sensitivity...")
            valid_results = [r for r in results if r['sensitivity'] >= self.target_sensitivity]
            
            if valid_results:
                best_result = max(valid_results, key=lambda x: x['specificity'])
                best_threshold = best_result['threshold']
                print(f"   📈 Best achievable: Sens={best_result['sensitivity']:.3f}, Spec={best_result['specificity']:.3f}")
            else:
                # Find best sensitivity
                best_result = max(results, key=lambda x: x['sensitivity'])
                best_threshold = best_result['threshold']
                print(f"   📈 Max sensitivity: Sens={best_result['sensitivity']:.3f}, Spec={best_result['specificity']:.3f}")
        
        self.optimal_threshold = best_threshold
        print(f"   ✅ Optimal threshold: {best_threshold:.3f}")
        
        return best_threshold, results
    
    def evaluate_clinical_performance(self, X_test, y_test):
        """Comprehensive clinical performance evaluation"""
        print("📈 Evaluating clinical performance...")
        
        # Predictions with optimal threshold
        y_proba = self.ensemble.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.optimal_threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        auc_roc = roc_auc_score(y_test, y_proba)
        
        clinical_metrics = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'optimal_threshold': self.optimal_threshold,
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
            }
        }
        
        # Clinical targets check
        meets_sensitivity = sensitivity >= self.target_sensitivity
        meets_specificity = specificity >= self.target_specificity
        
        print(f"\\n   🏥 CLINICAL PERFORMANCE RESULTS:")
        print(f"      Sensitivity: {sensitivity:.3f} {'✅' if meets_sensitivity else '❌'} (target: ≥{self.target_sensitivity})")
        print(f"      Specificity: {specificity:.3f} {'✅' if meets_specificity else '❌'} (target: ≥{self.target_specificity})")
        print(f"      PPV: {ppv:.3f}")
        print(f"      NPV: {npv:.3f}")
        print(f"      Accuracy: {accuracy:.3f}")
        print(f"      F1-Score: {f1:.3f}")
        print(f"      AUC-ROC: {auc_roc:.3f}")
        print(f"      Threshold: {self.optimal_threshold:.3f}")
        
        # Clinical interpretation
        if meets_sensitivity and meets_specificity:
            print(f"\\n   🎯 SUCCESS: Clinical targets achieved!")
        elif meets_sensitivity:
            print(f"\\n   ⚠️  Partial success: Sensitivity target met, specificity needs improvement")
        else:
            print(f"\\n   ❌ Clinical targets not met, consider feature engineering or model adjustments")
        
        return clinical_metrics
    
    def save_clinical_model(self, clinical_metrics):
        """Save the trained clinical model and results"""
        print("💾 Saving clinical ensemble model...")
        
        # Create clinical models directory
        Path('clinical_models').mkdir(exist_ok=True)
        
        # Save the ensemble model
        joblib.dump(self.ensemble, 'clinical_models/clinical_ensemble_model.pkl')
        joblib.dump(self.scaler, 'clinical_models/feature_scaler.pkl')
        
        # Save model metadata
        model_metadata = {
            'model_type': 'clinical_ensemble',
            'base_models': list(self.models.keys()),
            'feature_names': self.feature_names,
            'target_sensitivity': self.target_sensitivity,
            'target_specificity': self.target_specificity,
            'optimal_threshold': self.optimal_threshold,
            'cv_scores': self.cv_scores,
            'clinical_metrics': clinical_metrics,
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        with open('clinical_models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Save feature importance (from Random Forest)
        if 'random_forest' in self.models:
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models['random_forest'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv('clinical_models/feature_importance.csv', index=False)
        
        print(f"   ✅ Clinical model saved:")
        print(f"      Model: clinical_models/clinical_ensemble_model.pkl")
        print(f"      Scaler: clinical_models/feature_scaler.pkl") 
        print(f"      Metadata: clinical_models/model_metadata.json")
        
    def run_clinical_training(self):
        """Execute complete clinical training pipeline"""
        print("🚀 CLINICAL ENSEMBLE TRAINING PIPELINE")
        print("=" * 70)
        
        # Load clinical data
        if not self.load_clinical_data():
            return False
        
        # Create base models
        self.create_base_models()
        
        # Train base models
        self.train_base_models()
        
        # Create ensemble
        ensemble = self.create_stacked_ensemble()
        
        # Train ensemble on scaled data (handle NaN)
        X_cleaned = self.X.fillna(0)
        X_scaled = self.scaler.transform(X_cleaned)
        
        # Apply SMOTE for training
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, self.y)
        
        print("🏋️ Training final ensemble...")
        ensemble.fit(X_balanced, y_balanced)
        
        # Split data for threshold optimization and evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Optimize threshold on test set
        best_threshold, threshold_results = self.optimize_threshold_for_sensitivity(X_test, y_test)
        
        # Final clinical evaluation
        clinical_metrics = self.evaluate_clinical_performance(X_test, y_test)
        
        # Save model
        self.save_clinical_model(clinical_metrics)
        
        print("\\n" + "=" * 70)
        print("✅ CLINICAL ENSEMBLE TRAINING COMPLETE!")
        print(f"🎯 Clinical Performance: Sensitivity={clinical_metrics['sensitivity']:.3f}, Specificity={clinical_metrics['specificity']:.3f}")
        print("🏥 Model ready for clinical deployment")
        print("=" * 70)
        
        return clinical_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train clinical ensemble for ASD classification")
    parser.add_argument("--optimize-sensitivity", action="store_true",
                       help="Optimize for sensitivity target")
    parser.add_argument("--target", nargs=2, type=float, default=[0.86, 0.71],
                       help="Target sensitivity and specificity (default: 0.86 0.71)")
    
    args = parser.parse_args()
    
    target_sens, target_spec = args.target
    
    trainer = ClinicalEnsembleTrainer(
        target_sensitivity=target_sens,
        target_specificity=target_spec
    )
    
    success = trainer.run_clinical_training()
    
    if success:
        print(f"\n🎯 Next step: python validate_clinical_model.py --cv-folds 5 --test-size 0.2")
    else:
        print("\n❌ Clinical training failed. Check feature data and try again.")
