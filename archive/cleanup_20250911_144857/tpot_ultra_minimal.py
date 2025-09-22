#!/usr/bin/env python3
"""
ULTRA-MINIMAL TPOT
==================
Absolute minimal TPOT with just basic parameters.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

from tpot import TPOTClassifier

def main():
    print("🧬 ULTRA-MINIMAL TPOT TEST")
    
    # Load data
    data_path = Path("features_binary/advanced_clinical_features.csv")
    df = pd.read_csv(data_path)
    
    feature_cols = [col for col in df.columns if col not in ['child_id', 'label', 'binary_label']]
    X = df[feature_cols].values
    y = df['binary_label'].values.astype(int) if 'binary_label' in df.columns else (df['label'] == 'ASD').astype(int)
    
    # Impute
    X = SimpleImputer(strategy='median').fit_transform(X)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"ASD: {np.sum(y)}, TD: {len(y) - np.sum(y)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Running TPOT...")
    
    try:
        # Absolutely minimal TPOT
        tpot = TPOTClassifier(
            generations=5,
            population_size=10,
            random_state=42
        )
        
        tpot.fit(X_train, y_train)
        
        y_pred = tpot.predict(X_test)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"✅ TPOT Results:")
        print(f"   Sensitivity: {sensitivity:.3f} {'✅' if sensitivity >= 0.86 else '❌'}")
        print(f"   Specificity: {specificity:.3f} {'✅' if specificity >= 0.71 else '❌'}")
        print(f"   Your V4.2: Sens 71.6%, Spec 72.5%")
        
        if sensitivity >= 0.86 and specificity >= 0.71:
            print("🎉 TPOT SUCCESS! Both targets achieved!")
        elif sensitivity > 0.716 or specificity > 0.725:
            print("⚡ TPOT improved one metric vs V4.2")
        else:
            print("🔄 Your V4.2 remains better")
        
        # Show pipeline
        print(f"\nBest pipeline:")
        print(tpot.fitted_pipeline_)
        
        # Save if successful
        try:
            results_dir = Path("tpot_ultra_minimal_results")
            results_dir.mkdir(exist_ok=True)
            tpot.export(results_dir / "pipeline.py")
            print(f"Pipeline saved to {results_dir}/pipeline.py")
        except:
            pass
            
    except Exception as e:
        print(f"❌ TPOT failed: {e}")
        print("Your V4.2 approach remains optimal")

if __name__ == "__main__":
    main()
