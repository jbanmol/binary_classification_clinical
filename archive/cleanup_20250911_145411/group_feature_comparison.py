#!/usr/bin/env python3
"""
Group Feature Comparison Analysis
Identifies the most discriminative features between TD and ASD+DD groups
for binary classification model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class FeatureComparison:
    """Compare features between TD and ASD+DD groups"""
    
    def __init__(self, features_path):
        self.features_path = Path(features_path)
        self.features_df = None
        self.feature_importance = {}
        
    def load_features(self, filename='child_features_binary.csv'):
        """Load extracted features"""
        self.features_df = pd.read_csv(self.features_path / filename)
        print(f"Loaded {len(self.features_df)} children's features")
        print(f"Groups: {self.features_df['group'].value_counts().to_dict()}")
        
        # Get numeric features only
        self.numeric_features = [col for col in self.features_df.columns 
                               if col not in ['child_id', 'group', 'num_sessions']]
        
    def statistical_tests(self):
        """Perform statistical tests to compare groups"""
        results = []
        
        td_data = self.features_df[self.features_df['group'] == 'TD']
        asd_dd_data = self.features_df[self.features_df['group'] == 'ASD_DD']
        
        for feature in self.numeric_features:
            td_values = td_data[feature].dropna()
            asd_dd_values = asd_dd_data[feature].dropna()
            
            if len(td_values) > 0 and len(asd_dd_values) > 0:
                # T-test
                t_stat, t_pval = stats.ttest_ind(td_values, asd_dd_values)
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_pval = stats.mannwhitneyu(td_values, asd_dd_values, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((td_values.std()**2 + asd_dd_values.std()**2) / 2)
                cohen_d = (td_values.mean() - asd_dd_values.mean()) / pooled_std if pooled_std > 0 else 0
                
                # Store results
                results.append({
                    'feature': feature,
                    'td_mean': td_values.mean(),
                    'td_std': td_values.std(),
                    'asd_dd_mean': asd_dd_values.mean(),
                    'asd_dd_std': asd_dd_values.std(),
                    'mean_diff': td_values.mean() - asd_dd_values.mean(),
                    'mean_diff_pct': ((td_values.mean() - asd_dd_values.mean()) / td_values.mean() * 100) if td_values.mean() != 0 else 0,
                    't_statistic': t_stat,
                    't_pvalue': t_pval,
                    'u_statistic': u_stat,
                    'u_pvalue': u_pval,
                    'cohen_d': cohen_d,
                    'abs_cohen_d': abs(cohen_d)
                })
        
        self.stats_results = pd.DataFrame(results)
        
        # Add significance flags
        self.stats_results['significant_t'] = self.stats_results['t_pvalue'] < 0.05
        self.stats_results['significant_u'] = self.stats_results['u_pvalue'] < 0.05
        
        # Sort by effect size
        self.stats_results = self.stats_results.sort_values('abs_cohen_d', ascending=False)
        
        return self.stats_results
    
    def feature_importance_analysis(self):
        """Calculate feature importance using multiple methods"""
        
        # Prepare data
        X = self.features_df[self.numeric_features].fillna(0)
        y = (self.features_df['group'] == 'ASD_DD').astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. Mutual Information
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
        self.feature_importance['mutual_info'] = dict(zip(self.numeric_features, mi_scores))
        
        # 2. ANOVA F-scores
        f_scores, _ = f_classif(X_scaled, y)
        self.feature_importance['f_score'] = dict(zip(self.numeric_features, f_scores))
        
        # 3. Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        self.feature_importance['rf_importance'] = dict(zip(self.numeric_features, rf.feature_importances_))
        
        # 4. Cross-validated Random Forest performance for individual features
        cv_scores = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for feature in self.numeric_features:
            X_single = X[[feature]].values
            scores = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), 
                                   X_single, y, cv=cv, scoring='roc_auc')
            cv_scores[feature] = scores.mean()
        
        self.feature_importance['cv_auc'] = cv_scores
        
        # Combine all importance scores
        importance_df = pd.DataFrame(self.feature_importance)
        
        # Normalize scores to 0-1 range
        for col in importance_df.columns:
            importance_df[f'{col}_norm'] = (importance_df[col] - importance_df[col].min()) / (importance_df[col].max() - importance_df[col].min())
        
        # Calculate composite score
        norm_cols = [col for col in importance_df.columns if col.endswith('_norm')]
        importance_df['composite_score'] = importance_df[norm_cols].mean(axis=1)
        
        # Rank features
        importance_df = importance_df.sort_values('composite_score', ascending=False)
        
        return importance_df
    
    def visualize_top_features(self, n_top=20):
        """Visualize top discriminative features"""
        
        # Get top features by Cohen's d
        top_features = self.stats_results.nlargest(n_top, 'abs_cohen_d')['feature'].tolist()
        
        # Create comparison plots
        fig, axes = plt.subplots(5, 4, figsize=(16, 20))
        axes = axes.ravel()
        
        for idx, feature in enumerate(top_features[:20]):
            ax = axes[idx]
            
            td_data = self.features_df[self.features_df['group'] == 'TD'][feature].dropna()
            asd_dd_data = self.features_df[self.features_df['group'] == 'ASD_DD'][feature].dropna()
            
            # Box plot
            ax.boxplot([td_data, asd_dd_data], labels=['TD', 'ASD+DD'])
            
            # Add statistics
            stats_row = self.stats_results[self.stats_results['feature'] == feature].iloc[0]
            ax.set_title(f"{feature}\nCohen's d: {stats_row['cohen_d']:.3f}, p: {stats_row['u_pvalue']:.4f}", fontsize=8)
            ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        plt.savefig(self.features_path / 'top_features_comparison.png', dpi=150)
        plt.close()
        
        # Create heatmap of top features
        top_features_data = self.features_df[['group'] + top_features[:30]]
        
        # Group by group and calculate means
        grouped_means = top_features_data.groupby('group').mean()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(grouped_means.T, cmap='RdBu_r', center=0, 
                    cbar_kws={'label': 'Normalized Feature Value'})
        plt.title('Feature Comparison Heatmap: TD vs ASD+DD')
        plt.xlabel('Group')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(self.features_path / 'feature_heatmap.png', dpi=150)
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive feature analysis report"""
        
        report = []
        report.append("# Feature Comparison Report: TD vs ASD+DD\n")
        report.append(f"Total children: {len(self.features_df)}")
        report.append(f"Groups: {self.features_df['group'].value_counts().to_dict()}\n")
        
        # Top features by effect size
        report.append("## Top 30 Features by Effect Size (Cohen's d)\n")
        top_effect = self.stats_results.nlargest(30, 'abs_cohen_d')
        
        for idx, row in top_effect.iterrows():
            report.append(f"{idx+1}. **{row['feature']}**")
            report.append(f"   - TD: {row['td_mean']:.3f} ± {row['td_std']:.3f}")
            report.append(f"   - ASD+DD: {row['asd_dd_mean']:.3f} ± {row['asd_dd_std']:.3f}")
            report.append(f"   - Cohen's d: {row['cohen_d']:.3f}")
            report.append(f"   - p-value: {row['u_pvalue']:.4f}")
            report.append(f"   - Mean difference: {row['mean_diff_pct']:.1f}%\n")
        
        # Feature categories analysis
        report.append("\n## Feature Categories Performance\n")
        
        categories = {
            'velocity': [f for f in self.numeric_features if 'velocity' in f],
            'acceleration': [f for f in self.numeric_features if 'acc' in f],
            'area': [f for f in self.numeric_features if 'area' in f],
            'time': [f for f in self.numeric_features if 'time' in f or 'duration' in f],
            'tremor': [f for f in self.numeric_features if 'tremor' in f],
            'pressure': [f for f in self.numeric_features if 'pressure' in f],
            'palm': [f for f in self.numeric_features if 'palm' in f],
            'efficiency': [f for f in self.numeric_features if 'efficiency' in f],
            'progression': [f for f in self.numeric_features if 'progression' in f or 'consistency' in f],
            'completion': [f for f in self.numeric_features if 'completion' in f],
            'multi_touch': [f for f in self.numeric_features if 'multi' in f]
        }
        
        category_scores = {}
        for cat, features in categories.items():
            if features:
                cat_stats = self.stats_results[self.stats_results['feature'].isin(features)]
                if not cat_stats.empty:
                    avg_cohen_d = cat_stats['abs_cohen_d'].mean()
                    sig_features = cat_stats[cat_stats['significant_u']].shape[0]
                    category_scores[cat] = {
                        'avg_effect_size': avg_cohen_d,
                        'n_significant': sig_features,
                        'n_features': len(features)
                    }
        
        # Sort by average effect size
        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1]['avg_effect_size'], reverse=True)
        
        for cat, scores in sorted_cats:
            report.append(f"**{cat.capitalize()}**: avg Cohen's d = {scores['avg_effect_size']:.3f}, "
                         f"{scores['n_significant']}/{scores['n_features']} significant")
        
        # Recommendations
        report.append("\n## Recommendations for Model Training\n")
        
        # Get features with both high effect size and significance
        strong_features = self.stats_results[
            (self.stats_results['abs_cohen_d'] > 0.5) & 
            (self.stats_results['significant_u'])
        ]
        
        report.append(f"### Strong discriminative features ({len(strong_features)} total):")
        report.append("These features show both large effect sizes (Cohen's d > 0.5) and statistical significance:\n")
        
        for feature in strong_features['feature'].head(20):
            report.append(f"- {feature}")
        
        # Feature selection recommendation
        report.append("\n### Feature Selection Strategy:")
        report.append("1. Start with top 30-40 features by effect size")
        report.append("2. Include features from different categories for comprehensive representation")
        report.append("3. Consider removing highly correlated features (correlation > 0.9)")
        report.append("4. Use recursive feature elimination with cross-validation for final selection")
        
        # Save report
        with open(self.features_path / 'feature_comparison_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)
    
    def save_feature_rankings(self):
        """Save feature rankings to CSV"""
        
        # Combine statistical results with importance scores
        importance_df = self.feature_importance_analysis()
        
        # Merge with statistical results
        final_rankings = self.stats_results.merge(
            importance_df[['composite_score']], 
            left_on='feature', 
            right_index=True,
            how='left'
        )
        
        # Calculate final score
        final_rankings['final_score'] = (
            final_rankings['abs_cohen_d'] * 0.4 +  # Effect size weight
            final_rankings['composite_score'] * 0.3 +  # ML importance weight
            (1 - final_rankings['u_pvalue']) * 0.3  # Significance weight
        )
        
        # Sort by final score
        final_rankings = final_rankings.sort_values('final_score', ascending=False)
        
        # Save to CSV
        final_rankings.to_csv(self.features_path / 'feature_rankings.csv', index=False)
        
        # Save top features list
        top_features = final_rankings.head(40)['feature'].tolist()
        with open(self.features_path / 'selected_features.txt', 'w') as f:
            f.write('\n'.join(top_features))
        
        print(f"Saved feature rankings to {self.features_path / 'feature_rankings.csv'}")
        print(f"Top 40 features saved to {self.features_path / 'selected_features.txt'}")
        
        return final_rankings

def main():
    """Run group comparison analysis"""
    
    # Initialize analyzer
    analyzer = FeatureComparison('/Users/jbanmol/binary-classification-project/features_binary')
    
    # Load features
    analyzer.load_features()
    
    print("Running statistical tests...")
    stats_results = analyzer.statistical_tests()
    
    print("\nCalculating feature importance...")
    importance_results = analyzer.feature_importance_analysis()
    
    print("\nGenerating visualizations...")
    analyzer.visualize_top_features()
    
    print("\nGenerating report...")
    report = analyzer.generate_report()
    print(report[:1000] + "...\n")  # Print first part of report
    
    print("Saving feature rankings...")
    rankings = analyzer.save_feature_rankings()
    
    print("\nAnalysis complete! Check the features_binary folder for:")
    print("- feature_comparison_report.md")
    print("- feature_rankings.csv")
    print("- selected_features.txt")
    print("- top_features_comparison.png")
    print("- feature_heatmap.png")

if __name__ == "__main__":
    main()
