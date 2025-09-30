#!/usr/bin/env python3
"""
Robust Feature Selection for Model Stability
Implements feature selection to reduce variance and improve performance

Usage: python analysis/feature_selection.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, f_classif
)
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    """Robust feature selection for improved model stability"""
    
    def __init__(self):
        self.selected_features = {}
        self.stability_scores = {}
        
    def stability_selection(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                          n_iterations: int = 50, threshold: float = 0.6) -> Dict:
        """Implement stability selection with bootstrap sampling"""
        
        print(f"🔄 Running stability selection ({n_iterations} iterations)...")
        
        feature_counts = Counter()
        
        # Bootstrap with group preservation
        unique_groups = groups.unique()
        
        for i in range(n_iterations):
            # Sample groups with replacement
            sampled_groups = np.random.choice(unique_groups, 
                                             size=len(unique_groups), 
                                             replace=True)
            
            # Get all samples from sampled groups
            bootstrap_indices = []
            for group in sampled_groups:
                group_indices = groups[groups == group].index
                bootstrap_indices.extend(group_indices)
            
            X_boot = X.loc[bootstrap_indices]
            y_boot = y.loc[bootstrap_indices]
            
            # Feature selection on bootstrap sample
            selected = self._select_top_features(X_boot, y_boot, k=20)
            
            # Count selections
            for feature in selected:
                feature_counts[feature] += 1
        
        # Calculate stability scores
        stability_scores = {}
        for feature in X.columns:
            stability_scores[feature] = feature_counts[feature] / n_iterations
        
        # Select stable features
        stable_features = [f for f, score in stability_scores.items() 
                          if score >= threshold]
        
        print(f"  📊 Selected {len(stable_features)} stable features (threshold: {threshold})")
        
        return {
            'stable_features': stable_features,
            'stability_scores': stability_scores,
            'threshold': threshold,
            'n_iterations': n_iterations
        }
    
    def _select_top_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> List[str]:
        """Select top features using mutual information"""
        
        X_clean = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
            selector.fit(X_clean, y)
            return X.columns[selector.get_support()].tolist()
        except:
            # Fallback to high variance features
            variances = X_clean.var()
            return variances.nlargest(k).index.tolist()
    
    def ensemble_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Combine multiple feature selection methods"""
        
        print("🎯 Running ensemble feature selection...")
        
        X_clean = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        ensemble_results = {}
        
        # Method 1: Mutual Information
        try:
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=30)
            mi_selector.fit(X_clean, y)
            mi_features = X.columns[mi_selector.get_support()].tolist()
            ensemble_results['mutual_info'] = mi_features
            print(f"  📊 Mutual Info: {len(mi_features)} features")
        except Exception as e:
            print(f"  ⚠️ Mutual Info failed: {e}")
            ensemble_results['mutual_info'] = []
        
        # Method 2: F-test
        try:
            f_selector = SelectKBest(score_func=f_classif, k=30)
            f_selector.fit(X_clean, y)
            f_features = X.columns[f_selector.get_support()].tolist()
            ensemble_results['f_test'] = f_features
            print(f"  📈 F-test: {len(f_features)} features")
        except Exception as e:
            print(f"  ⚠️ F-test failed: {e}")
            ensemble_results['f_test'] = []
        
        # Method 3: Random Forest importance
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_clean, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            rf_features = importances.nlargest(30).index.tolist()
            ensemble_results['random_forest'] = rf_features
            print(f"  🌳 Random Forest: {len(rf_features)} features")
        except Exception as e:
            print(f"  ⚠️ Random Forest failed: {e}")
            ensemble_results['random_forest'] = []
        
        # Method 4: Lasso
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
            lasso.fit(X_scaled, y)
            lasso_features = X.columns[np.abs(lasso.coef_) > 1e-6].tolist()
            ensemble_results['lasso'] = lasso_features
            print(f"  📐 Lasso: {len(lasso_features)} features")
        except Exception as e:
            print(f"  ⚠️ Lasso failed: {e}")
            ensemble_results['lasso'] = []
        
        # Voting combination
        all_features = set()
        for features in ensemble_results.values():
            all_features.update(features)
        
        feature_votes = {}
        for feature in all_features:
            votes = sum(1 for features in ensemble_results.values() if feature in features)
            feature_votes[feature] = votes
        
        # Different consensus levels
        majority_features = [f for f, votes in feature_votes.items() if votes >= 2]
        consensus_features = [f for f, votes in feature_votes.items() if votes >= 3]
        unanimous_features = [f for f, votes in feature_votes.items() if votes == 4]
        
        print(f"  🗺 Majority vote (2+): {len(majority_features)} features")
        print(f"  🤝 Consensus (3+): {len(consensus_features)} features")
        print(f"  🔥 Unanimous (4): {len(unanimous_features)} features")
        
        return {
            'method_results': ensemble_results,
            'feature_votes': feature_votes,
            'majority_features': majority_features,
            'consensus_features': consensus_features,
            'unanimous_features': unanimous_features
        }
    
    def evaluate_feature_sets(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                            feature_sets: Dict[str, List[str]]) -> Dict:
        """Evaluate different feature sets using cross-validation"""
        
        print("📊 Evaluating feature sets...")
        
        cv_strategy = GroupKFold(n_splits=5)
        model = LogisticRegression(random_state=42, max_iter=1000)
        scaler = StandardScaler()
        
        evaluation_results = {}
        
        for set_name, features in feature_sets.items():
            if not features:
                print(f"  ⚠️ Skipping empty set: {set_name}")
                continue
            
            print(f"  Testing {set_name} ({len(features)} features)...")
            
            try:
                X_subset = X[features].fillna(X[features].median())
                
                # Cross-validation scores
                scores = []
                for train_idx, val_idx in cv_strategy.split(X_subset, y, groups=groups):
                    X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Scale and train
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    model.fit(X_train_scaled, y_train)
                    score = model.score(X_val_scaled, y_val)
                    scores.append(score)
                
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                cv_coeff = std_score / mean_score if mean_score > 0 else np.inf
                
                evaluation_results[set_name] = {
                    'features': features,
                    'n_features': len(features),
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'cv_coefficient': cv_coeff,
                    'scores': scores
                }
                
                print(f"    Score: {mean_score:.4f} ± {std_score:.4f} (CV: {cv_coeff:.4f})")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        return evaluation_results
    
    def recommend_features(self, stability_results: Dict, ensemble_results: Dict, 
                         evaluation_results: Dict) -> Dict:
        """Generate final feature recommendations"""
        
        print("🎯 Generating feature recommendations...")
        
        # Start with stable features
        stable_features = set(stability_results.get('stable_features', []))
        consensus_features = set(ensemble_results.get('consensus_features', []))
        
        # Combine strategies
        recommended_features = list(stable_features.union(consensus_features))
        
        # If too many, prioritize by stability and consensus
        if len(recommended_features) > 50:  # Reasonable limit
            # Sort by stability score and keep top features
            stability_scores = stability_results.get('stability_scores', {})
            sorted_features = sorted(recommended_features, 
                                   key=lambda f: stability_scores.get(f, 0), 
                                   reverse=True)
            recommended_features = sorted_features[:50]
        
        # Find best performing set from evaluation
        best_set = None
        best_score = 0
        if evaluation_results:
            for set_name, results in evaluation_results.items():
                if results['mean_score'] > best_score:
                    best_score = results['mean_score']
                    best_set = set_name
        
        recommendation = {
            'recommended_features': recommended_features,
            'n_recommended': len(recommended_features),
            'strategy': 'stable + consensus',
            'best_evaluated_set': best_set,
            'best_score': best_score,
            'composition': {
                'stable_count': len(stable_features & set(recommended_features)),
                'consensus_count': len(consensus_features & set(recommended_features)),
                'overlap_count': len(stable_features & consensus_features)
            }
        }
        
        print(f"  🎯 Final recommendation: {len(recommended_features)} features")
        print(f"  📈 Composition: {recommendation['composition']['stable_count']} stable + {recommendation['composition']['consensus_count']} consensus")
        
        return recommendation


def main():
    """Run feature selection analysis"""
    
    print("🔬 Robust Feature Selection")
    print("=" * 30)
    print("Selecting features to reduce model variance...")
    
    # Load processed features
    feature_files = list(Path("results").glob("*features_aligned.csv"))
    if not feature_files:
        print("❌ No processed feature files found in results/")
        print("💡 Run the main pipeline first: ./train_final.sh")
        return
    
    # Use most recent file
    feature_file = sorted(feature_files)[-1]
    print(f"📂 Using: {feature_file}")
    
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
            print("❌ No target column found")
            return
        
        # Get groups
        if 'child_id' in df.columns:
            groups = df['child_id']
        else:
            groups = pd.Series(range(len(df)))
        
        # Clean features
        meta_cols = ['child_id', 'Unnamed: 0', 'index']
        X = X.drop([col for col in meta_cols if col in X.columns], axis=1)
        
        print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features, {groups.nunique()} children")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize selector
    selector = FeatureSelector()
    
    # Run stability selection
    stability_results = selector.stability_selection(X, y, groups)
    
    # Run ensemble selection
    ensemble_results = selector.ensemble_selection(X, y)
    
    # Prepare feature sets for evaluation
    feature_sets = {
        'all_features': X.columns.tolist(),
        'stable_features': stability_results['stable_features'],
        'consensus_features': ensemble_results['consensus_features'],
        'majority_features': ensemble_results['majority_features'],
        'unanimous_features': ensemble_results['unanimous_features']
    }
    
    # Remove empty sets
    feature_sets = {k: v for k, v in feature_sets.items() if v}
    
    # Evaluate feature sets
    evaluation_results = selector.evaluate_feature_sets(X, y, groups, feature_sets)
    
    # Generate recommendations
    recommendation = selector.recommend_features(stability_results, ensemble_results, evaluation_results)
    
    # Save results
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Save recommended features
    recommended_df = pd.DataFrame({
        'feature': recommendation['recommended_features']
    })
    recommended_df.to_csv(output_dir / "recommended_features.csv", index=False)
    
    # Save detailed results
    detailed_results = {
        'stability_selection': {
            'stable_features': stability_results['stable_features'],
            'n_stable': len(stability_results['stable_features']),
            'threshold': stability_results['threshold']
        },
        'ensemble_selection': {
            'consensus_features': ensemble_results['consensus_features'],
            'majority_features': ensemble_results['majority_features'],
            'n_consensus': len(ensemble_results['consensus_features'])
        },
        'evaluation': {
            name: {
                'n_features': results['n_features'],
                'mean_score': results['mean_score'],
                'cv_coefficient': results['cv_coefficient']
            }
            for name, results in evaluation_results.items()
        },
        'recommendation': recommendation
    }
    
    import json
    with open(output_dir / "feature_selection_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Print summary
    print("\n📊 Selection Summary:")
    print(f"  Original features: {X.shape[1]}")
    print(f"  Stable features: {len(stability_results['stable_features'])}")
    print(f"  Consensus features: {len(ensemble_results['consensus_features'])}")
    print(f"  Final recommendation: {len(recommendation['recommended_features'])}")
    
    if evaluation_results:
        best_set = recommendation['best_evaluated_set']
        best_score = recommendation['best_score']
        print(f"  Best performing set: {best_set} (score: {best_score:.4f})")
    
    print("\n💡 Impact on Variance:")
    print("  - Feature selection should reduce overfitting")
    print("  - Stable features improve cross-validation consistency")
    print("  - Consensus features are robust across methods")
    print("  - Recommended set balances performance and stability")
    
    print(f"\n📄 Results saved:")
    print(f"  analysis/recommended_features.csv - Features to use")
    print(f"  analysis/feature_selection_results.json - Detailed analysis")
    
    print("✅ Feature selection complete!")
    print("\n🚀 Next steps:")
    print("  1. Retrain models using recommended features")
    print("  2. Compare CV variance before/after feature selection")
    print("  3. Monitor sensitivity stability improvement")


if __name__ == "__main__":
    main()
