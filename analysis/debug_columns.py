#!/usr/bin/env python3
"""
Debug Columns - Quick tool to examine dataset structure and identify target columns

Usage: python analysis/debug_columns.py
"""

import pandas as pd
from pathlib import Path
import sys

def debug_columns():
    """Debug column issues in features and sample data"""
    
    print("🔍 Debugging Column Structure\n")
    
    # Check for processed features in results/
    feature_files = list(Path("results").glob("*features_aligned.csv"))
    if feature_files:
        feature_file = sorted(feature_files)[-1]
        print(f"📄 Processing features file: {feature_file.name}")
        
        try:
            df_features = pd.read_csv(feature_file)
            print(f"   Shape: {df_features.shape}")
            print(f"   Columns ({len(df_features.columns)}):")
            for i, col in enumerate(df_features.columns):
                print(f"     {i+1:2d}. {col}")
            
            # Check for target-like columns
            target_candidates = [col for col in df_features.columns if any(keyword in col.lower() 
                                for keyword in ['label', 'target', 'class', 'outcome', 'diagnosis', 'asd', 'td'])]
            print(f"\n   🎯 Potential target columns: {target_candidates or 'None found'}")
            
            # Check first few rows
            print(f"\n   📊 First 3 rows:")
            print(df_features.head(3).to_string())
            
        except Exception as e:
            print(f"   ❌ Error reading features file: {e}")
    else:
        print("❌ No processed features files found in results/")
    
    print("\n" + "="*60)
    
    # Check sample_clinical_data.csv
    sample_file = Path("sample_clinical_data.csv")
    if sample_file.exists():
        print(f"📄 Examining sample data file: {sample_file}")
        
        try:
            df_sample = pd.read_csv(sample_file)
            print(f"   Shape: {df_sample.shape}")
            print(f"   Columns ({len(df_sample.columns)}):")
            for i, col in enumerate(df_sample.columns):
                print(f"     {i+1:2d}. {col}")
            
            # Check for target-like columns
            target_candidates = [col for col in df_sample.columns if any(keyword in col.lower() 
                                for keyword in ['label', 'target', 'class', 'outcome', 'diagnosis', 'asd', 'td'])]
            print(f"\n   🎯 Target columns found: {target_candidates}")
            
            # Show target distribution if found
            for col in target_candidates:
                if col == 'target':  # Main target column
                    print(f"\n   📊 Target distribution for '{col}':")
                    print(df_sample[col].value_counts().to_string())
                    print(f"   📊 Target statistics:")
                    print(f"     - Total samples: {len(df_sample)}")
                    print(f"     - ASD (target=1): {sum(df_sample[col] == 1.0)} ({100*sum(df_sample[col] == 1.0)/len(df_sample):.1f}%)")
                    print(f"     - TD (target=0): {sum(df_sample[col] == 0.0)} ({100*sum(df_sample[col] == 0.0)/len(df_sample):.1f}%)")
                    break
                    
        except Exception as e:
            print(f"   ❌ Error reading sample file: {e}")
    else:
        print("⚠️ sample_clinical_data.csv not found")
    
    print("\n" + "="*60)
    print("\n💡 DIAGNOSIS:")
    print("The processed features files in results/ are missing target/label columns.")
    print("The labels exist in sample_clinical_data.csv but are not being carried forward")
    print("during the feature processing pipeline.\n")
    
    print("🔧 SOLUTION:")
    print("The analysis scripts need to be modified to either:")
    print("1. Load labels from sample_clinical_data.csv and merge with features")
    print("2. Use the RAG system behavioral database which contains labels")
    print("3. Fix the feature processing pipeline to preserve labels\n")
    
if __name__ == "__main__":
    debug_columns()
