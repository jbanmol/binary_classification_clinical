#!/usr/bin/env python3
"""
Clinical Model Validation with Child-Level Splits
Prevents data leakage by ensuring no child appears in both train and test sets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, accuracy_score)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class ChildLevelValidator:
    """Validates models with child-level splits to prevent data leakage"""
    
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds
        self.results = {}
        self.per_fold_results = []
        
    def load_clinical_data(self):
        """Load clinical data with child-level grouping"""
        print("📊 Loading clinical data for child-level validation...")
        
        try:
            self.clinical_data = pd.read_csv('features_binary/clinical_features_optimized.csv')
            
            # Prepare features and labels with child grouping
            self.X = self.clinical_data.drop(['child_id', 'group', 'binary_label'], axis=1).fillna(0)
            self.y = self.clinical_data['binary_label']
            self.groups = self.clinical_data['child_id']  # Child IDs for grouping
            
            # Get unique children and their labels
            child_labels = self.clinical_data.groupby('child_id')['binary_label'].first()
            
            print(f"   ✅ Clinical dataset loaded:")
            print(f"      Total samples: {len(self.clinical_data)}")
            print(f"      Unique children: {len(child_labels)}")
            print(f"      ASD children: {child_labels.sum()}")
            print(f"      TD children: {len(child_labels) - child_labels.sum()}")
            print(f"      Features: {len(self.X.columns)}")
            
            # Check for data leakage potential
            sessions_per_child = self.clinical_data['child_id'].value_counts()
            print(f"      Avg sessions per child: {sessions_per_child.mean():.2f}")
            print(f"      Max sessions per child: {sessions_per_child.max()}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def create_base_models(self):
        """Create diverse base models optimized for clinical targets"""
        print("🤖 Creating base models for child-level validation...")
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y), y=self.y)
        weight_dict = {0: class_weights[0], 1: class_weights[1] * 1.2}
        
        models = {}
        
        # XGBoost - Excellent with imbalanced data
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,  # Reduced to prevent overfitting
                max_depth=4,       # Reduced depth
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=class_weights[1]/class_weights[0] * 1.2,
                random_state=42,
                eval_metric='logloss'
            )
        
        # LightGBM - Fast and robust
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight=weight_dict,
                random_state=42,
                verbose=-1
            )
        
        # Random Forest - Stable
        models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,  # Higher to prevent overfitting
            min_samples_leaf=5,    # Higher to prevent overfitting
            class_weight=weight_dict,
            random_state=42
        )
        
        # Logistic Regression - Clinical baseline
        models['logistic'] = LogisticRegression(
            C=0.1,  # Higher regularization
            class_weight=weight_dict,
            random_state=42,
            max_iter=1000
        )
        
        # SVM - Non-linear patterns
        models['svm'] = SVC(
            C=0.1,  # Higher regularization
            kernel='rbf',
            gamma='scale',
            class_weight=weight_dict,
            probability=True,
            random_state=42
        )
        
        self.models = models
        print(f"   ✅ Created {len(models)} base models: {list(models.keys())}")
        return models
    
    def calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate comprehensive metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = accuracy_score(y_true, y_pred)
        
        if len(np.unique(y_true)) == 2:
            auc_roc = roc_auc_score(y_true, y_proba)
        else:
            auc_roc = 0.5
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity, 
            'ppv': ppv,
            'npv': npv,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
    
    def child_level_cross_validation(self):
        """Perform child-level cross-validation to prevent data leakage"""
        print(f"🔬 Performing {self.cv_folds}-fold child-level cross-validation...")
        print("   This prevents data leakage by ensuring no child appears in both train/test")
        
        # Use GroupKFold to ensure children are not split across folds
        group_kfold = GroupKFold(n_splits=self.cv_folds)
        
        # Get unique child labels for stratification guidance
        child_labels = self.clinical_data.groupby('child_id')['binary_label'].first()
        
        fold_results = {model_name: [] for model_name in self.models.keys()}
        fold_results['ensemble'] = []
        
        fold_num = 1
        
        for train_idx, test_idx in group_kfold.split(self.X, self.y, groups=self.groups):
            print(f"\\n   📁 FOLD {fold_num}/{self.cv_folds}")
            
            # Split data by child groups
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            groups_train = self.groups.iloc[train_idx]
            groups_test = self.groups.iloc[test_idx]
            
            # Verify no child leakage
            train_children = set(groups_train.unique())
            test_children = set(groups_test.unique())
            leakage_children = train_children.intersection(test_children)
            
            if leakage_children:
                print(f"      ❌ DATA LEAKAGE DETECTED: {len(leakage_children)} children in both sets")
                continue
            else:
                print(f"      ✅ No data leakage: {len(train_children)} train, {len(test_children)} test children")
            
            # Display fold statistics
            print(f"      Train: {len(X_train)} samples, {y_train.sum()} ASD, {len(y_train)-y_train.sum()} TD")
            print(f"      Test:  {len(X_test)} samples, {y_test.sum()} ASD, {len(y_test)-y_test.sum()} TD")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply SMOTE only to training data
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            # Train and evaluate each model
            fold_predictions = {}
            fold_probabilities = {}
            
            for model_name, model in self.models.items():
                print(f"         🎯 Training {model_name}...")
                
                # Train model
                model.fit(X_train_balanced, y_train_balanced)
                
                # Predict on test set
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred, y_proba)
                fold_results[model_name].append(metrics)
                
                # Store for ensemble
                fold_predictions[model_name] = y_pred
                fold_probabilities[model_name] = y_proba
                
                print(f"            Sens: {metrics['sensitivity']:.3f}, Spec: {metrics['specificity']:.3f}, AUC: {metrics['auc_roc']:.3f}")
            
            # Create ensemble predictions (simple averaging)
            if len(fold_probabilities) > 0:
                ensemble_proba = np.mean(list(fold_probabilities.values()), axis=0)
                ensemble_pred = (ensemble_proba >= 0.5).astype(int)
                
                ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred, ensemble_proba)
                fold_results['ensemble'].append(ensemble_metrics)
                
                print(f"         🏆 Ensemble: Sens: {ensemble_metrics['sensitivity']:.3f}, Spec: {ensemble_metrics['specificity']:.3f}, AUC: {ensemble_metrics['auc_roc']:.3f}")
            
            fold_num += 1
        
        self.fold_results = fold_results
        return fold_results
    
    def calculate_cv_statistics(self):
        """Calculate cross-validation statistics"""
        print("\\n📊 CROSS-VALIDATION RESULTS SUMMARY")
        print("=" * 60)
        
        cv_summary = {}
        
        for model_name, fold_metrics in self.fold_results.items():
            if not fold_metrics:
                continue
                
            # Extract metrics across folds
            metrics_by_fold = {
                metric: [fold[metric] for fold in fold_metrics]
                for metric in fold_metrics[0].keys()
                if isinstance(fold_metrics[0][metric], (int, float))
            }
            
            # Calculate statistics
            model_stats = {}
            for metric, values in metrics_by_fold.items():
                model_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            
            cv_summary[model_name] = model_stats
            
            # Print summary for this model
            print(f"\\n🤖 {model_name.upper()}:")
            print(f"   Sensitivity: {model_stats['sensitivity']['mean']:.3f} ± {model_stats['sensitivity']['std']:.3f} [{model_stats['sensitivity']['min']:.3f}, {model_stats['sensitivity']['max']:.3f}]")
            print(f"   Specificity: {model_stats['specificity']['mean']:.3f} ± {model_stats['specificity']['std']:.3f} [{model_stats['specificity']['min']:.3f}, {model_stats['specificity']['max']:.3f}]")
            print(f"   Accuracy:    {model_stats['accuracy']['mean']:.3f} ± {model_stats['accuracy']['std']:.3f} [{model_stats['accuracy']['min']:.3f}, {model_stats['accuracy']['max']:.3f}]")
            print(f"   AUC-ROC:     {model_stats['auc_roc']['mean']:.3f} ± {model_stats['auc_roc']['std']:.3f} [{model_stats['auc_roc']['min']:.3f}, {model_stats['auc_roc']['max']:.3f}]")
            
            # Check clinical targets
            sens_meets_target = model_stats['sensitivity']['mean'] >= 0.86
            spec_meets_target = model_stats['specificity']['mean'] >= 0.71
            
            if sens_meets_target and spec_meets_target:
                print(f"   🎯 CLINICAL TARGETS: ✅ ACHIEVED")
            elif sens_meets_target:
                print(f"   🎯 CLINICAL TARGETS: ⚠️  Sensitivity OK, Specificity below target")
            else:
                print(f"   🎯 CLINICAL TARGETS: ❌ Below targets")
        
        self.cv_summary = cv_summary
        return cv_summary
    
    def save_validation_results(self):
        """Save comprehensive validation results"""
        print("\\n💾 Saving validation results...")
        
        # Create validation directory
        Path('validation_results').mkdir(exist_ok=True)
        
        # Save per-fold results
        pd.DataFrame([
            {
                'fold': i+1,
                'model': model_name,
                **metrics
            }
            for model_name, fold_metrics in self.fold_results.items()
            for i, metrics in enumerate(fold_metrics)
        ]).to_csv('validation_results/per_fold_results.csv', index=False)
        
        # Save CV summary
        with open('validation_results/cv_summary.json', 'w') as f:
            json.dump(self.cv_summary, f, indent=2)
        
        # Create detailed report
        report = f"""# Child-Level Cross-Validation Report
        
## Validation Setup
- Cross-validation folds: {self.cv_folds}
- Split method: GroupKFold (child-level)
- Data leakage prevention: ✅ Ensured
- Total children: {len(self.groups.unique())}
- Total samples: {len(self.clinical_data)}

## Clinical Targets
- Target ASD Sensitivity: ≥86%
- Target ASD Specificity: ≥71%

## Results Summary
"""
        
        for model_name, stats in self.cv_summary.items():
            report += f"""
### {model_name.upper()}
- **Sensitivity**: {stats['sensitivity']['mean']:.3f} ± {stats['sensitivity']['std']:.3f}
- **Specificity**: {stats['specificity']['mean']:.3f} ± {stats['specificity']['std']:.3f}
- **Accuracy**: {stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}
- **AUC-ROC**: {stats['auc_roc']['mean']:.3f} ± {stats['auc_roc']['std']:.3f}
"""
        
        with open('validation_results/validation_report.md', 'w') as f:
            f.write(report)
        
        print("   ✅ Results saved to validation_results/")
    
    def run_validation(self):
        """Execute complete child-level validation"""
        print("🚀 CHILD-LEVEL VALIDATION PIPELINE")
        print("=" * 60)
        print("🛡️  PREVENTING DATA LEAKAGE WITH CHILD-LEVEL SPLITS")
        print("=" * 60)
        
        # Load data
        if not self.load_clinical_data():
            return False
        
        # Create models
        self.create_base_models()
        
        # Perform child-level CV
        self.child_level_cross_validation()
        
        # Calculate statistics
        self.calculate_cv_statistics()
        
        # Save results
        self.save_validation_results()
        
        print("\\n" + "=" * 60)
        print("✅ CHILD-LEVEL VALIDATION COMPLETE")
        print("🛡️  Results are trustworthy - no data leakage")
        print("📊 Check validation_results/ for detailed analysis")
        print("=" * 60)
        
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate clinical model with child-level splits")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    
    args = parser.parse_args()
    
    validator = ChildLevelValidator(cv_folds=args.cv_folds)
    success = validator.run_validation()
    
    if success:
        print("\\n🎯 Validation complete. Check results for real performance without data leakage.")
    else:
        print("\\n❌ Validation failed. Check data and try again.")
