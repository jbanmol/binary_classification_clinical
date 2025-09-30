#!/usr/bin/env python3
"""
Feature Quality Analysis for Clinical ML
Analyzes feature distributions and identifies problematic features

Usage: python analysis/feature_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.feature_selection import mutual_info_classif, f_classif
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    """Analyze feature quality for clinical ML"""
    
    def __init__(self):
        self.feature_stats = {}
        self.problematic_features = []
        
    def analyze_features(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze feature distributions and quality"""
        
        print("📊 Analyzing feature quality...")
        
        feature_analysis = {}
        
        for feature in X.columns:
            values = X[feature].dropna()
            
            stats_dict = {
                'name': feature,
                'missing_rate': X[feature].isnull().mean(),
                'unique_count': X[feature].nunique(),
                'unique_rate': X[feature].nunique() / len(X),
                'zero_rate': (X[feature] == 0).mean() if X[feature].dtype in [int, float] else 0,
            }
            
            # Numeric analysis
            if X[feature].dtype in [int, float] and len(values) > 0:
                stats_dict.update({
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': values.median(),
                    'min': values.min(),
                    'max': values.max(),
                    'skewness': stats.skew(values),
                    'cv': values.std() / values.mean() if values.mean() != 0 else np.inf,
                    'outlier_rate': self._outlier_rate(values)
                })
            
            feature_analysis[feature] = stats_dict
        
        return feature_analysis
    
    def _outlier_rate(self, values: pd.Series) -> float:
        """Calculate outlier rate using IQR"""
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return 0.0
            
        outliers = (values < Q1 - 1.5*IQR) | (values > Q3 + 1.5*IQR)
        return outliers.mean()
    
    def calculate_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Calculate feature importance scores"""
        
        print("🎯 Calculating feature importance...")
        
        X_clean = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        importance_results = {}
        
        # Mutual Information
        try:
            mi_scores = mutual_info_classif(X_clean, y, random_state=42)
            importance_results['mutual_info'] = dict(zip(X.columns, mi_scores))
        except Exception as e:
            print(f"⚠️ MI calculation failed: {e}")
            importance_results['mutual_info'] = {}
        
        # F-test
        try:
            f_scores, p_values = f_classif(X_clean, y)
            importance_results['f_score'] = dict(zip(X.columns, f_scores))
            importance_results['f_p_value'] = dict(zip(X.columns, p_values))
        except Exception as e:
            print(f"⚠️ F-test failed: {e}")
            importance_results['f_score'] = {}
        
        return importance_results
    
    def find_problematic_features(self, feature_analysis: Dict, importance_analysis: Dict) -> List[str]:
        """Identify features that may hurt performance"""
        
        print("🚨 Finding problematic features...")
        
        problematic = []
        reasons = {}
        
        for feature, stats in feature_analysis.items():
            issues = []
            
            # High missing rate
            if stats['missing_rate'] > 0.3:
                issues.append(f"High missing: {stats['missing_rate']:.2f}")
            
            # Nearly constant
            if stats['unique_rate'] < 0.01:
                issues.append(f"Low variance: {stats['unique_rate']:.3f}")
            
            # High outliers
            if 'outlier_rate' in stats and stats['outlier_rate'] > 0.2:
                issues.append(f"High outliers: {stats['outlier_rate']:.2f}")
            
            # Low importance
            mi_score = importance_analysis.get('mutual_info', {}).get(feature, 0)
            if mi_score < 0.01:
                issues.append(f"Low importance: {mi_score:.4f}")
            
            if issues:
                problematic.append(feature)
                reasons[feature] = issues
        
        self.problematic_features = problematic
        
        print(f"🚨 Found {len(problematic)} problematic features")
        if problematic:
            for feat in problematic[:5]:  # Show first 5
                print(f"  {feat}: {', '.join(reasons[feat])}")
        
        return problematic
    
    def generate_recommendations(self, feature_analysis: Dict, problematic_features: List[str]) -> List[str]:
        """Generate improvement recommendations"""
        
        recs = []
        
        total_features = len(feature_analysis)
        prob_count = len(problematic_features)
        
        if prob_count > 0:
            recs.append(f"🚨 Remove {prob_count} problematic features")
        
        # Missing values
        high_missing = sum(1 for stats in feature_analysis.values() 
                          if stats['missing_rate'] > 0.2)
        if high_missing > 0:
            recs.append(f"📝 Handle {high_missing} features with high missing rates")
        
        # Skewed features
        skewed_features = sum(1 for stats in feature_analysis.values() 
                            if 'skewness' in stats and abs(stats['skewness']) > 2)
        if skewed_features > 0:
            recs.append(f"📈 Consider transforming {skewed_features} highly skewed features")
        
        # Feature selection
        if prob_count > total_features * 0.3:
            recs.append("🎯 Consider aggressive feature selection")
        
        if not recs:
            recs.append("✅ Feature quality looks good")
        
        return recs


def main():
    """Run feature quality analysis"""
    
    print("🔬 Feature Quality Analysis")
    print("=" * 35)
    
    # Load processed features
    feature_files = list(Path("results").glob("*features_aligned.csv"))
    if not feature_files:
        print("❌ No processed feature files found in results/")
        print("💡 Run the main pipeline first: ./train_final.sh")
        return
    
    # Use most recent file
    feature_file = sorted(feature_files)[-1]
    print(f"📂 Loading: {feature_file}")
    
    try:
        df = pd.read_csv(feature_file)
        
        # Extract features and target
        if 'label' in df.columns:
            X = df.drop(['label'], axis=1)
            y = df['label']
        elif 'target' in df.columns:
            X = df.drop(['target'], axis=1) 
            y = df['target']
        else:
            print("❌ No label/target column found")
            return
        
        # Remove metadata columns
        meta_cols = ['child_id', 'Unnamed: 0', 'index']
        X = X.drop([col for col in meta_cols if col in X.columns], axis=1)
        
        print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 Target balance: {y.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run analysis
    analyzer = FeatureAnalyzer()
    
    feature_analysis = analyzer.analyze_features(X, y)
    importance_analysis = analyzer.calculate_importance(X, y)
    problematic_features = analyzer.find_problematic_features(feature_analysis, importance_analysis)
    recommendations = analyzer.generate_recommendations(feature_analysis, problematic_features)
    
    # Save results
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Feature stats
    stats_df = pd.DataFrame(feature_analysis).T
    stats_df.to_csv(output_dir / "feature_stats.csv")
    
    # Problematic features
    if problematic_features:
        prob_df = pd.DataFrame({'feature': problematic_features})
        prob_df.to_csv(output_dir / "problematic_features.csv", index=False)
    
    # Results summary
    print("\n📊 Analysis Summary:")
    print(f"  Total features: {len(feature_analysis)}")
    print(f"  Problematic features: {len(problematic_features)}")
    
    missing_stats = [s['missing_rate'] for s in feature_analysis.values()]
    print(f"  Average missing rate: {np.mean(missing_stats):.3f}")
    print(f"  Max missing rate: {np.max(missing_stats):.3f}")
    
    if importance_analysis.get('mutual_info'):
        mi_scores = list(importance_analysis['mutual_info'].values())
        print(f"  Average mutual info: {np.mean(mi_scores):.4f}")
    
    print("\n💡 Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    
    print(f"\n📄 Results saved:")
    print(f"  analysis/feature_stats.csv - Detailed feature statistics")
    if problematic_features:
        print(f"  analysis/problematic_features.csv - Features to review")
    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()
