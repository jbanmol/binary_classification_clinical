#!/usr/bin/env python3
"""
MINIMAL TPOT CLINICAL OPTIMIZATION
==================================
Minimal TPOT approach using only basic parameters.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

from tpot import TPOTClassifier

def load_clinical_data():
    """Load clinical data"""
    print("📊 Loading clinical dataset...")
    
    data_path = Path("features_binary/advanced_clinical_features.csv")
    df = pd.read_csv(data_path)
    
    feature_cols = [col for col in df.columns if col not in ['child_id', 'label', 'binary_label']]
    X = df[feature_cols].values
    
    if 'binary_label' in df.columns:
        y = df['binary_label'].values.astype(int)
    else:
        y = (df['label'] == 'ASD').astype(int)
    
    child_ids = df['child_id'].values
    
    # Simple imputation
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    print(f"   ✅ {len(df)} samples, {len(np.unique(child_ids))} children, {X.shape[1]} features")
    print(f"      ASD: {np.sum(y)}, TD: {len(y) - np.sum(y)}")
    
    return X, y, child_ids

def main():
    """Run minimal TPOT"""
    print("🧬 MINIMAL TPOT FOR CLINICAL OPTIMIZATION")
    print("🎯 Sensitivity ≥86%, Specificity ≥71%\n")
    
    # Load data
    X, y, child_ids = load_clinical_data()
    
    # Simple train/test split for TPOT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🔬 Training: {len(X_train)} samples ({np.sum(y_train)} ASD)")
    print(f"🔬 Testing:  {len(X_test)} samples ({np.sum(y_test)} ASD)")
    
    # Minimal TPOT configuration
    print("\n🧬 Running TPOT with minimal configuration...")
    
    try:
        tpot = TPOTClassifier(
            generations=8,
            population_size=15,
            verbosity=2,
            random_state=42,
            n_jobs=2,
            max_time_mins=15
        )
        
        # Fit TPOT
        tpot.fit(X_train, y_train)
        
        print("✅ TPOT optimization completed!")
        
        # Test initial performance
        y_pred = tpot.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        print(f"\n📊 TPOT Test Performance:")
        print(f"   Sensitivity: {sensitivity:.3f} {'✅' if sensitivity >= 0.86 else '❌'}")
        print(f"   Specificity: {specificity:.3f} {'✅' if specificity >= 0.71 else '❌'}")
        print(f"   Accuracy:    {accuracy:.3f}")
        
        # Display best pipeline
        print(f"\n🏆 BEST PIPELINE:")
        print("=" * 50)
        print(tpot.fitted_pipeline_)
        print("=" * 50)
        
        # Child-level validation
        print("\n🔬 Child-level cross-validation:")
        cv = GroupKFold(n_splits=5)
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=child_ids)):
            X_fold_train, X_fold_test = X[train_idx], X[test_idx]
            y_fold_train, y_fold_test = y[train_idx], y[test_idx]
            
            try:
                tpot.fitted_pipeline_.fit(X_fold_train, y_fold_train)
                y_fold_pred = tpot.fitted_pipeline_.predict(X_fold_test)
                
                tn, fp, fn, tp = confusion_matrix(y_fold_test, y_fold_pred).ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                fold_results.append({'sensitivity': sens, 'specificity': spec})
                
                sens_check = "✅" if sens >= 0.86 else "❌"
                spec_check = "✅" if spec >= 0.71 else "❌"
                print(f"   Fold {fold_idx + 1}: Sens {sens:.3f}{sens_check}, Spec {spec:.3f}{spec_check}")
                
            except Exception as e:
                print(f"   Fold {fold_idx + 1}: ❌ Failed")
        
        if fold_results:
            # Aggregate results
            sensitivities = [fr['sensitivity'] for fr in fold_results]
            specificities = [fr['specificity'] for fr in fold_results]
            
            sens_mean = np.mean(sensitivities)
            spec_mean = np.mean(specificities)
            
            sens_target_met = sens_mean >= 0.86
            spec_target_met = spec_mean >= 0.71
            both_targets_met = sens_target_met and spec_target_met
            
            print(f"\n📊 TPOT FINAL RESULTS:")
            print("-" * 40)
            print(f"Sensitivity: {sens_mean:.3f} ± {np.std(sensitivities):.3f} {'✅' if sens_target_met else '❌'}")
            print(f"Specificity: {spec_mean:.3f} ± {np.std(specificities):.3f} {'✅' if spec_target_met else '❌'}")
            
            if both_targets_met:
                print("🎉 SUCCESS! TPOT achieved both clinical targets!")
                recommendation = "✅ TPOT pipeline recommended"
            else:
                print("⚠️ TPOT didn't achieve both clinical targets")
                if sens_target_met:
                    recommendation = "⚠️ Good sensitivity, but specificity needs improvement"
                elif spec_target_met:
                    recommendation = "⚠️ Good specificity, but sensitivity needs improvement"
                else:
                    recommendation = "❌ Both targets missed - consider alternative approaches"
            
            print(f"💡 {recommendation}")
            
            # Compare with your V4.2
            print(f"\n🔄 Comparison with your V4.2:")
            print(f"   Your V4.2:     Sens 71.6%, Spec 72.5%")
            print(f"   TPOT Result:   Sens {sens_mean:.1%}, Spec {spec_mean:.1%}")
            
            if sens_mean > 0.716 and spec_mean > 0.725:
                print("🏆 TPOT outperformed your V4.2!")
            elif sens_mean > 0.716 or spec_mean > 0.725:
                print("⚡ TPOT improved one metric vs V4.2")
            else:
                print("🔄 Your V4.2 remains superior to TPOT")
        
        # Save pipeline
        results_dir = Path("tpot_minimal_results")
        results_dir.mkdir(exist_ok=True)
        
        try:
            tpot.export(results_dir / "tpot_minimal_pipeline.py")
            print(f"\n✅ Pipeline exported to {results_dir}/tpot_minimal_pipeline.py")
        except Exception as e:
            print(f"\n⚠️ Could not export pipeline: {e}")
        
    except Exception as e:
        print(f"❌ TPOT failed: {e}")
        print("\nTPOT optimization unsuccessful - your V4.2 approach remains optimal")
    
    print("\n🧬 TPOT evaluation complete!")

if __name__ == "__main__":
    main()
