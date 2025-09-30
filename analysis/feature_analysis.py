#!/usr/bin/env python3
"""
Feature Quality Analysis for Clinical Binary Classification

Analyzes feature importance, correlation, and quality metrics to identify
optimal features for achieving 86% sensitivity target.

Usage: python analysis/feature_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats

warnings.filterwarnings('ignore')

def load_labels_from_processed() -> Optional[pd.DataFrame]:
    """Load labels from data/processed/labels.csv"""
    try:
        labels_file = Path("data/processed/labels.csv")
        if not labels_file.exists():
            return None
            
        df_labels = pd.read_csv(labels_file)
        
        # Check for required columns
        if 'Unity_id' not in df_labels.columns or 'Group' not in df_labels.columns:
            print(f"   ⚠️ Missing required columns in {labels_file}")
            return None
        
        # Convert Group to target (ASD=1, TD=0)
        df_labels['target'] = (df_labels['Group'] == 'ASD').astype(int)
        
        # Rename Unity_id to child_id for consistency
        df_labels = df_labels.rename(columns={'Unity_id': 'child_id'})
        
        child_labels = df_labels[['child_id', 'target']].copy()
        
        print(f"   ✅ Loaded {len(child_labels)} labels from processed/labels.csv")
        print(f"   📊 Label distribution: ASD={child_labels['target'].sum()}, TD={len(child_labels)-child_labels['target'].sum()}")
        
        return child_labels
        
    except Exception as e:
        print(f"   ⚠️ Error loading processed labels: {e}")
        return None

def create_synthetic_labels(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic labels for analysis when real labels aren't available"""
    print("   ⚠️ Creating synthetic labels for analysis purposes...")
    
    child_labels = features_df[['child_id']].copy()
    
    # Use a feature-based heuristic to create realistic synthetic labels
    # This is ONLY for analysis - not for training!
    velocity_feature = features_df.get('velocity_cv', features_df.get('vel_std_over_mean', 0))
    tremor_feature = features_df.get('tremor_indicator', 0)
    
    # Simple heuristic: higher velocity variation + tremor indicates ASD
    # This is SYNTHETIC and only for testing the analysis pipeline!
    if hasattr(velocity_feature, 'quantile'):
        high_var_threshold = velocity_feature.quantile(0.6)
        synthetic_labels = ((velocity_feature > high_var_threshold) | (tremor_feature > tremor_feature.quantile(0.7))).astype(int)
    else:
        # Fallback: alternate labels
        synthetic_labels = (features_df.index % 2).astype(int)
    
    child_labels['target'] = synthetic_labels
    
    print(f"   🧪 Created synthetic labels: {synthetic_labels.sum()} ASD, {len(synthetic_labels) - synthetic_labels.sum()} TD")
    print(f"   ⚠️ WARNING: These are SYNTHETIC labels for analysis only!")
    
    return child_labels

def load_features_with_labels() -> Tuple[Optional[pd.DataFrame], bool]:
    """Load features and merge with real labels"""
    
    # Find latest features file
    feature_files = list(Path("results").glob("*features_aligned.csv"))
    if not feature_files:
        print("❌ No processed feature files found in results/")
        return None, False
    
    feature_file = sorted(feature_files)[-1]
    print(f"📊 Loading features from: {feature_file.name}")
    
    try:
        df_features = pd.read_csv(feature_file)
        print(f"   Shape: {df_features.shape[0]} samples, {df_features.shape[1]} features")
        
        # Check if labels already exist
        if 'target' in df_features.columns or 'label' in df_features.columns:
            target_col = 'target' if 'target' in df_features.columns else 'label'
            print(f"   ✅ Found existing target column: {target_col}")
            if target_col != 'target':
                df_features = df_features.rename(columns={target_col: 'target'})
            return df_features, True
        
        # Need to merge with labels
        print("   ⚠️ No target column found in features, attempting to merge with labels...")
        
        if 'child_id' not in df_features.columns:
            print("   ❌ No child_id column found for merging")
            return None, False
        
        # Try to load real labels first
        child_labels = load_labels_from_processed()
        
        if child_labels is None:
            print("   ⚠️ No real labels available, creating synthetic labels for analysis...")
            child_labels = create_synthetic_labels(df_features)
        
        # Merge features with labels
        print(f"   🔄 Merging {len(df_features)} feature rows with {len(child_labels)} labeled children...")
        
        # Check for child ID format compatibility
        features_ids = set(df_features['child_id'].astype(str))
        label_ids = set(child_labels['child_id'].astype(str))
        overlap = features_ids.intersection(label_ids)
        
        print(f"   🔍 ID format check: {len(overlap)} overlapping IDs")
        
        if len(overlap) == 0:
            print(f"   ⚠️ No matching child IDs - using synthetic labels for analysis")
            print(f"   Features ID example: {list(features_ids)[0] if features_ids else 'None'}")
            print(f"   Labels ID example: {list(label_ids)[0] if label_ids else 'None'}")
            
            # Use synthetic labels based on features
            child_labels = create_synthetic_labels(df_features)
        
        df_merged = df_features.merge(child_labels, on='child_id', how='inner')
        
        if df_merged.empty:
            print("   ❌ Merge failed - using features with synthetic labels")
            df_merged = df_features.copy()
            synthetic_labels = create_synthetic_labels(df_features)
            df_merged = df_merged.merge(synthetic_labels, on='child_id', how='left')
        
        print(f"   ✅ Analysis dataset ready: {len(df_merged)} samples with labels")
        return df_merged, True
        
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return None, False

def calculate_feature_importance(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
    """Calculate multiple feature importance metrics"""
    
    feature_names = X.columns.tolist()
    importance_scores = {}
    
    print("🔄 Calculating feature importance metrics...")
    
    # 1. Mutual Information
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        importance_scores['mutual_info'] = dict(zip(feature_names, mi_scores))
        print("   ✅ Mutual information calculated")
    except Exception as e:
        print(f"   ⚠️ Mutual information failed: {e}")
        importance_scores['mutual_info'] = {f: 0.0 for f in feature_names}
    
    # 2. F-test (ANOVA)
    try:
        f_scores, _ = f_classif(X, y)
        # Normalize F-scores
        f_scores_norm = f_scores / np.max(f_scores) if np.max(f_scores) > 0 else f_scores
        importance_scores['f_test'] = dict(zip(feature_names, f_scores_norm))
        print("   ✅ F-test scores calculated")
    except Exception as e:
        print(f"   ⚠️ F-test failed: {e}")
        importance_scores['f_test'] = {f: 0.0 for f in feature_names}
    
    # 3. Random Forest importance
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance_scores['random_forest'] = dict(zip(feature_names, rf.feature_importances_))
        print("   ✅ Random Forest importance calculated")
    except Exception as e:
        print(f"   ⚠️ Random Forest failed: {e}")
        importance_scores['random_forest'] = {f: 0.0 for f in feature_names}
    
    # 4. Logistic Regression coefficients (L1 regularization)
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
        lr.fit(X_scaled, y)
        
        # Use absolute coefficients as importance
        coef_importance = np.abs(lr.coef_[0])
        importance_scores['logistic_l1'] = dict(zip(feature_names, coef_importance))
        print("   ✅ Logistic regression importance calculated")
    except Exception as e:
        print(f"   ⚠️ Logistic regression failed: {e}")
        importance_scores['logistic_l1'] = {f: 0.0 for f in feature_names}
    
    return importance_scores

def calculate_correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Calculate feature correlation matrix"""
    
    print("🔄 Calculating feature correlations...")
    
    try:
        corr_matrix = X.corr().abs()  # Absolute correlation
        print(f"   ✅ Correlation matrix calculated ({corr_matrix.shape})")
        return corr_matrix
    except Exception as e:
        print(f"   ⚠️ Correlation calculation failed: {e}")
        return pd.DataFrame()

def analyze_feature_quality(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Analyze individual feature quality metrics"""
    
    print("🔄 Analyzing feature quality...")
    
    feature_quality = {}
    
    for feature in X.columns:
        values = X[feature].dropna()
        
        quality_metrics = {
            'missing_rate': X[feature].isnull().mean(),
            'unique_count': X[feature].nunique(),
            'unique_rate': X[feature].nunique() / len(X),
            'zero_rate': (X[feature] == 0).mean() if X[feature].dtype in [int, float] else 0,
        }
        
        # Numeric analysis
        if X[feature].dtype in [int, float] and len(values) > 0:
            quality_metrics.update({
                'mean': values.mean(),
                'std': values.std(),
                'cv': values.std() / values.mean() if values.mean() != 0 else np.inf,
                'skewness': stats.skew(values),
                'outlier_rate': calculate_outlier_rate(values)
            })
        
        feature_quality[feature] = quality_metrics
    
    print(f"   ✅ Quality metrics calculated for {len(feature_quality)} features")
    return feature_quality

def calculate_outlier_rate(values: pd.Series) -> float:
    """Calculate outlier rate using IQR"""
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:
        return 0.0
        
    outliers = (values < Q1 - 1.5*IQR) | (values > Q3 + 1.5*IQR)
    return outliers.mean()

def identify_problematic_features(feature_quality: Dict, importance_scores: Dict) -> List[str]:
    """Identify features that may hurt model performance"""
    
    print("🔄 Identifying problematic features...")
    
    problematic = []
    
    for feature, quality in feature_quality.items():
        issues = []
        
        # High missing rate
        if quality['missing_rate'] > 0.3:
            issues.append('high_missing')
        
        # Nearly constant
        if quality['unique_rate'] < 0.01:
            issues.append('low_variance')
        
        # High outlier rate
        if 'outlier_rate' in quality and quality['outlier_rate'] > 0.2:
            issues.append('high_outliers')
        
        # Low importance across all methods
        avg_importance = 0
        importance_count = 0
        for method_scores in importance_scores.values():
            if feature in method_scores:
                avg_importance += method_scores[feature]
                importance_count += 1
        
        if importance_count > 0:
            avg_importance /= importance_count
            if avg_importance < 0.01:
                issues.append('low_importance')
        
        if issues:
            problematic.append(feature)
    
    print(f"   🚨 Found {len(problematic)} problematic features")
    return problematic

def generate_recommendations(importance_scores: Dict, feature_quality: Dict, 
                           problematic_features: List[str]) -> Dict:
    """Generate feature selection recommendations"""
    
    print("💡 Generating feature recommendations...")
    
    # Combined importance scoring
    all_features = set()
    for method_scores in importance_scores.values():
        all_features.update(method_scores.keys())
    
    combined_scores = {}
    for feature in all_features:
        scores = []
        for method_scores in importance_scores.values():
            if feature in method_scores:
                scores.append(method_scores[feature])
        
        if scores:
            combined_scores[feature] = np.mean(scores)
        else:
            combined_scores[feature] = 0.0
    
    # Sort by importance
    sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Generate recommendations
    top_features = [f[0] for f in sorted_features[:25]]
    bottom_features = [f[0] for f in sorted_features[-10:]]
    
    # Filter out problematic features from top features
    clean_top_features = [f for f in top_features if f not in problematic_features]
    
    # Ensure minimum feature count
    if len(clean_top_features) < 15:
        backup_features = [f for f in top_features if f not in clean_top_features][:15-len(clean_top_features)]
        clean_top_features.extend(backup_features)
    
    recommendations = {
        'top_features': clean_top_features,
        'problematic_features': problematic_features,
        'bottom_features': bottom_features,
        'recommended_count': min(20, len(clean_top_features)),
        'feature_scores': dict(sorted_features),
        'feature_quality': feature_quality
    }
    
    print(f"   ✅ Generated {len(clean_top_features)} recommended features")
    
    return recommendations

def save_results(recommendations: Dict, importance_scores: Dict, feature_quality: Dict) -> None:
    """Save analysis results to files"""
    
    print("💾 Saving analysis results...")
    
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Save recommendations
    with open(analysis_dir / "feature_recommendations.json", 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)
    
    # Save importance scores
    with open(analysis_dir / "feature_importance_scores.json", 'w') as f:
        json.dump(importance_scores, f, indent=2, default=str)
    
    # Save quality metrics
    quality_df = pd.DataFrame(feature_quality).T
    quality_df.to_csv(analysis_dir / "feature_quality_metrics.csv")
    
    # Save recommended features list
    recommended_features = recommendations['top_features'][:recommendations['recommended_count']]
    features_df = pd.DataFrame({'feature': recommended_features})
    features_df.to_csv(analysis_dir / "recommended_features.csv", index=False)
    
    print(f"   ✅ Results saved to analysis/ directory")
    print(f"   📄 Key files: feature_recommendations.json, recommended_features.csv")

def print_summary(recommendations: Dict) -> None:
    """Print analysis summary"""
    
    print(f"\n{'='*60}")
    print("📊 FEATURE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n🎯 Feature Selection Results:")
    print(f"   - Recommended features: {recommendations['recommended_count']}")
    print(f"   - Problematic features: {len(recommendations['problematic_features'])}")
    
    print(f"\n⭐ Top 10 Most Important Features:")
    for i, feature in enumerate(recommendations['top_features'][:10], 1):
        score = recommendations['feature_scores'][feature]
        print(f"   {i:2d}. {feature:<30} (score: {score:.4f})")
    
    if recommendations['problematic_features']:
        print(f"\n🚨 Problematic Features to Review:")
        for feature in recommendations['problematic_features'][:5]:
            print(f"   - {feature}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review recommended_features.csv")
    print(f"   2. Retrain with selected features:")
    print(f"      ./train_final.sh --features analysis/recommended_features.csv")
    print(f"   3. Compare performance vs current 82.1% sensitivity")
    print(f"   4. Target: Achieve 86%+ sensitivity with reduced variance")
    
    print(f"\n✅ Analysis complete!")

def main():
    """Main feature analysis workflow"""
    
    print("🔬 Feature Quality Analysis")
    print("=" * 50)
    print("🎯 Goal: Identify optimal features for 86% sensitivity target")
    print("🚀 Current bagged performance: 82.1% sensitivity (3.9% gap)")
    print()
    
    # Load data
    df, has_target = load_features_with_labels()
    
    if df is None or not has_target:
        print("❌ No target column found")
        print("💡 Run the main pipeline first or check data files")
        return 1
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col not in ['child_id', 'target', 'label']]
    X = df[feature_columns]
    y = df['target']
    
    print(f"📊 Dataset loaded:")
    print(f"   - Samples: {len(df)}")
    print(f"   - Features: {len(feature_columns)}")
    print(f"   - Target distribution: ASD={y.sum()}, TD={len(y)-y.sum()}")
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("🔄 Handling missing values...")
        X = X.fillna(X.median())
    
    # Run analyses
    importance_scores = calculate_feature_importance(X, y)
    feature_quality = analyze_feature_quality(X, y)
    problematic_features = identify_problematic_features(feature_quality, importance_scores)
    recommendations = generate_recommendations(importance_scores, feature_quality, problematic_features)
    
    # Save and summarize
    save_results(recommendations, importance_scores, feature_quality)
    print_summary(recommendations)
    
    return 0

if __name__ == "__main__":
    exit(main())