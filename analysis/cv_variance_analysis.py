#!/usr/bin/env python3
"""
Cross-Validation Variance Analysis
Analyzes and reduces the high CV variance (60-90% sensitivity range)

Usage: python analysis/cv_variance_analysis.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import (
    GroupKFold, StratifiedKFold, RepeatedStratifiedKFold
)
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class CVVarianceAnalyzer:
    """Analyze and reduce cross-validation variance"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_existing_results(self) -> Dict:
        """Analyze variance from existing training results"""
        
        print("🔍 Analyzing existing CV results...")
        
        result_files = list(Path("results").glob("final_s*_ho777_*.json"))
        if not result_files:
            print("❌ No training result files found")
            return {}
        
        print(f"Found {len(result_files)} result files")
        
        all_results = []
        
        for file_path in sorted(result_files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                seed = data.get('seed', 'unknown')
                cv_results = data.get('cv', {}).get('folds', [])
                
                if not cv_results:
                    continue
                
                # Extract fold metrics
                fold_metrics = []
                for fold_data in cv_results:
                    if isinstance(fold_data, dict):
                        metrics = fold_data.get('metrics', fold_data)
                        sens = metrics.get('sensitivity', metrics.get('sens', metrics.get('recall_pos')))
                        spec = metrics.get('specificity', metrics.get('spec', metrics.get('tnr')))
                        auc = metrics.get('auc', metrics.get('auc_roc'))
                        
                        if sens is not None:
                            fold_metrics.append({
                                'sensitivity': sens,
                                'specificity': spec or 0,
                                'auc': auc or 0
                            })
                
                if fold_metrics:
                    # Calculate CV statistics for this seed
                    sensitivities = [m['sensitivity'] for m in fold_metrics]
                    specificities = [m['specificity'] for m in fold_metrics]
                    aucs = [m['auc'] for m in fold_metrics]
                    
                    seed_analysis = {
                        'seed': seed,
                        'file': file_path.name,
                        'n_folds': len(fold_metrics),
                        'sens_mean': np.mean(sensitivities),
                        'sens_std': np.std(sensitivities),
                        'sens_min': np.min(sensitivities),
                        'sens_max': np.max(sensitivities),
                        'sens_range': np.max(sensitivities) - np.min(sensitivities),
                        'sens_cv': np.std(sensitivities) / np.mean(sensitivities) if np.mean(sensitivities) > 0 else 0,
                        'spec_mean': np.mean(specificities),
                        'spec_std': np.std(specificities),
                        'spec_cv': np.std(specificities) / np.mean(specificities) if np.mean(specificities) > 0 else 0,
                        'auc_mean': np.mean(aucs),
                        'auc_std': np.std(aucs),
                        'auc_cv': np.std(aucs) / np.mean(aucs) if np.mean(aucs) > 0 else 0
                    }
                    
                    all_results.append(seed_analysis)
                    
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                continue
        
        if not all_results:
            print("❌ No valid CV results found")
            return {}
        
        # Overall variance analysis
        df = pd.DataFrame(all_results)
        
        variance_analysis = {
            'n_seeds': len(df),
            'overall_sens_cv_mean': df['sens_cv'].mean(),
            'overall_sens_cv_std': df['sens_cv'].std(),
            'worst_sens_variance': df['sens_cv'].max(),
            'best_sens_variance': df['sens_cv'].min(),
            'sens_range_mean': df['sens_range'].mean(),
            'sens_range_max': df['sens_range'].max(),
            'spec_cv_mean': df['spec_cv'].mean(),
            'auc_cv_mean': df['auc_cv'].mean(),
            'detailed_results': all_results
        }
        
        # Print analysis
        print("\n📊 CV Variance Analysis:")
        print(f"  Seeds analyzed: {variance_analysis['n_seeds']}")
        print(f"  Mean sensitivity CV: {variance_analysis['overall_sens_cv_mean']:.4f}")
        print(f"  Sensitivity range (mean): {variance_analysis['sens_range_mean']:.3f}")
        print(f"  Sensitivity range (max): {variance_analysis['sens_range_max']:.3f}")
        
        if variance_analysis['sens_range_max'] > 0.3:  # 30% range
            print(f"  🚨 HIGH VARIANCE DETECTED: {variance_analysis['sens_range_max']:.1%} range")
        elif variance_analysis['sens_range_max'] > 0.2:  # 20% range
            print(f"  ⚠️ MODERATE VARIANCE: {variance_analysis['sens_range_max']:.1%} range")
        else:
            print(f"  ✅ ACCEPTABLE VARIANCE: {variance_analysis['sens_range_max']:.1%} range")
        
        return variance_analysis
    
    def test_cv_strategies(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Dict:
        """Test different CV strategies for variance reduction"""
        
        print("\n🧪 Testing CV strategies...")
        
        strategies = {
            'GroupKFold': GroupKFold(n_splits=5),
            'StratifiedKFold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            'RepeatedStratified': RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        }
        
        # Simple model for testing
        model = LogisticRegression(random_state=42, max_iter=1000)
        scaler = StandardScaler()
        
        strategy_results = {}
        
        for strategy_name, cv_strategy in strategies.items():
            print(f"  Testing {strategy_name}...")
            
            fold_scores = []
            
            try:
                if strategy_name == 'GroupKFold':
                    splits = cv_strategy.split(X, y, groups=groups)
                else:
                    splits = cv_strategy.split(X, y)
                
                for train_idx, val_idx in splits:
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Preprocess
                    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
                    X_val_scaled = scaler.transform(X_val.fillna(0))
                    
                    # Train and predict
                    model.fit(X_train_scaled, y_train)
                    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # Metrics
                    auc = roc_auc_score(y_val, y_pred_proba)
                    sensitivity = recall_score(y_val, y_pred)
                    specificity = recall_score(y_val, y_pred, pos_label=0)
                    
                    fold_scores.append({
                        'auc': auc,
                        'sensitivity': sensitivity,
                        'specificity': specificity
                    })
                
                # Calculate variance statistics
                sens_scores = [s['sensitivity'] for s in fold_scores]
                spec_scores = [s['specificity'] for s in fold_scores]
                auc_scores = [s['auc'] for s in fold_scores]
                
                strategy_results[strategy_name] = {
                    'n_folds': len(fold_scores),
                    'sens_mean': np.mean(sens_scores),
                    'sens_std': np.std(sens_scores),
                    'sens_cv': np.std(sens_scores) / np.mean(sens_scores) if np.mean(sens_scores) > 0 else 0,
                    'sens_range': np.max(sens_scores) - np.min(sens_scores),
                    'spec_mean': np.mean(spec_scores),
                    'spec_std': np.std(spec_scores),
                    'auc_mean': np.mean(auc_scores),
                    'auc_std': np.std(auc_scores)
                }
                
                result = strategy_results[strategy_name]
                print(f"    Sensitivity: {result['sens_mean']:.3f} ± {result['sens_std']:.3f} (range: {result['sens_range']:.3f})")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                continue
        
        return strategy_results
    
    def generate_recommendations(self, existing_analysis: Dict, strategy_results: Dict) -> List[str]:
        """Generate variance reduction recommendations"""
        
        recommendations = []
        
        # Analyze current variance
        if existing_analysis:
            max_range = existing_analysis.get('sens_range_max', 0)
            mean_cv = existing_analysis.get('overall_sens_cv_mean', 0)
            
            if max_range > 0.3:
                recommendations.append("🚨 CRITICAL: Reduce high sensitivity variance (>30% range)")
                recommendations.append("   - Current fold variance is affecting clinical reliability")
                recommendations.append("   - Consider data quality issues or model overfitting")
            elif max_range > 0.2:
                recommendations.append("⚠️ MODERATE: Improve sensitivity consistency (>20% range)")
            
            if mean_cv > 0.15:
                recommendations.append("📉 High coefficient of variation - model is unstable")
        
        # Strategy recommendations
        if strategy_results:
            best_strategy = min(strategy_results.keys(), 
                              key=lambda k: strategy_results[k].get('sens_cv', 1))
            best_cv = strategy_results[best_strategy].get('sens_cv', 0)
            
            recommendations.append(f"🎯 Best CV strategy tested: {best_strategy} (CV: {best_cv:.4f})")
            
            # Specific strategy advice
            if 'RepeatedStratified' in strategy_results:
                repeated_cv = strategy_results['RepeatedStratified'].get('sens_cv', 0)
                if repeated_cv < existing_analysis.get('overall_sens_cv_mean', 1):
                    recommendations.append("🔄 Repeated CV shows better stability - consider using")
        
        # General recommendations
        recommendations.extend([
            "🔧 Variance Reduction Strategies:",
            "   - Feature selection to reduce overfitting",
            "   - Stronger regularization parameters", 
            "   - Ensemble methods for stability",
            "   - Data quality improvements",
            "   - Larger training dataset if possible"
        ])
        
        if not recommendations:
            recommendations.append("✅ CV variance appears acceptable")
        
        return recommendations


def main():
    """Run CV variance analysis"""
    
    print("🔬 CV Variance Analysis")
    print("=" * 30)
    print("Analyzing the 60-90% sensitivity range issue...")
    
    analyzer = CVVarianceAnalyzer()
    
    # Analyze existing results
    existing_analysis = analyzer.analyze_existing_results()
    
    # Test CV strategies if we have features
    strategy_results = {}
    feature_files = list(Path("results").glob("*features_aligned.csv"))
    
    if feature_files:
        print("\n🔬 Testing CV strategies on current features...")
        
        try:
            # Load most recent features
            feature_file = sorted(feature_files)[-1]
            df = pd.read_csv(feature_file)
            
            # Extract features and target
            if 'label' in df.columns:
                X = df.drop(['label'], axis=1)
                y = df['label']
            elif 'target' in df.columns:
                X = df.drop(['target'], axis=1)
                y = df['target']
            else:
                raise ValueError("No target column found")
            
            # Get groups
            if 'child_id' in df.columns:
                groups = df['child_id']
            else:
                groups = pd.Series(range(len(df)))
            
            # Clean features
            meta_cols = ['child_id', 'Unnamed: 0', 'index']
            X = X.drop([col for col in meta_cols if col in X.columns], axis=1)
            
            print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {groups.nunique()} children")
            
            # Test strategies
            strategy_results = analyzer.test_cv_strategies(X, y, groups)
            
        except Exception as e:
            print(f"⚠️ Could not test CV strategies: {e}")
    
    # Generate recommendations
    recommendations = analyzer.generate_recommendations(existing_analysis, strategy_results)
    
    # Save results
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    results_summary = {
        'existing_analysis': existing_analysis,
        'strategy_comparison': strategy_results,
        'recommendations': recommendations
    }
    
    with open(output_dir / "cv_variance_analysis.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Print summary
    print("\n📊 Analysis Summary:")
    
    if existing_analysis:
        print(f"  Current sensitivity variance: {existing_analysis.get('sens_range_max', 0):.1%}")
        
        if existing_analysis.get('sens_range_max', 0) > 0.3:
            print("  🚨 HIGH VARIANCE - needs immediate attention")
        elif existing_analysis.get('sens_range_max', 0) > 0.2:
            print("  ⚠️ MODERATE VARIANCE - room for improvement")
        else:
            print("  ✅ ACCEPTABLE VARIANCE")
    
    print("\n💡 Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    
    print(f"\n📄 Detailed results: analysis/cv_variance_analysis.json")
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()
