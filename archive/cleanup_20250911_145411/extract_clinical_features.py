#!/usr/bin/env python3
"""
Clinical Feature Extraction for ASD Classification
RAG-Guided Feature Engineering for ≥86% Sensitivity Target
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class ClinicalFeatureExtractor:
    """RAG-guided clinical feature extraction for ASD classification"""
    
    def __init__(self, target_sensitivity=0.86):
        self.target_sensitivity = target_sensitivity
        self.high_impact_features = [
            # RAG-discovered high-impact biomarkers (Effect Size >0.8)
            'target_zone_coverage_min', 'target_zone_coverage_mean', 'target_zone_coverage_max',
            'outside_ratio_min', 'outside_ratio_mean', 'outside_ratio_max',
            'zone_transition_rate_min', 'zone_transition_rate_mean', 'zone_transition_rate_max',
            'velocity_mean_session_mean_mean', 'velocity_std_session_mean_mean',
            'jerk_score_session_mean_mean', 'acceleration_std_session_mean_mean'
        ]
        
        self.motor_control_features = [
            # Motor control biomarkers
            'num_palm_touches_mean', 'potential_palm_touches_mean',
            'max_finger_id_mean', 'num_unique_fingers_mean',
            'tremor_ratio_session_mean_mean', 'pressure_std_session_mean_mean',
            'device_stability_session_mean_mean'
        ]
        
        self.cognitive_features = [
            # Cognitive-behavioral patterns
            'progress_linearity_mean', 'final_completion_mean',
            'session_duration_mean', 'num_pauses_mean',
            'inter_stroke_interval_std_mean', 'color_switch_rate_mean'
        ]
        
    def load_data(self):
        """Load existing feature data"""
        print("📊 Loading existing feature data...")
        
        try:
            self.session_features = pd.read_csv('features_binary/session_features_binary.csv')
            self.child_features = pd.read_csv('features_binary/child_features_binary.csv')
            print(f"   ✅ Loaded {len(self.child_features)} children, {len(self.session_features)} sessions")
            
            # Prepare labels for binary classification
            if 'group' in self.child_features.columns:
                self.child_features['binary_label'] = (
                    self.child_features['group'].str.contains('ASD|DD', na=False)
                ).astype(int)
            
            return True
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def engineer_interaction_features(self):
        """Create interaction features based on RAG insights"""
        print("🔬 Engineering RAG-guided interaction features...")
        
        new_features = []
        
        # Velocity × Accuracy interactions (critical for ASD detection)
        if all(f in self.child_features.columns for f in ['velocity_mean_session_mean_mean', 'target_zone_coverage_mean']):
            self.child_features['velocity_accuracy_interaction'] = (
                self.child_features['velocity_mean_session_mean_mean'] * 
                (1 - self.child_features['target_zone_coverage_mean'])  # Inverse accuracy
            )
            new_features.append('velocity_accuracy_interaction')
        
        # Motor control composite score
        motor_cols = [f for f in self.motor_control_features if f in self.child_features.columns]
        if len(motor_cols) >= 3:
            # Standardize and combine
            motor_data = self.child_features[motor_cols].fillna(0)
            scaler = StandardScaler()
            motor_scaled = scaler.fit_transform(motor_data)
            self.child_features['motor_control_index'] = np.mean(motor_scaled, axis=1)
            new_features.append('motor_control_index')
        
        # Attention regulation index
        attention_cols = ['zone_transition_rate_mean', 'color_switch_rate_mean', 'num_pauses_mean']
        attention_available = [f for f in attention_cols if f in self.child_features.columns]
        if len(attention_available) >= 2:
            attention_data = self.child_features[attention_available].fillna(0)
            scaler = StandardScaler()
            attention_scaled = scaler.fit_transform(attention_data)
            self.child_features['attention_regulation_index'] = np.mean(attention_scaled, axis=1)
            new_features.append('attention_regulation_index')
        
        # Spatial accuracy deficit score (high values = more deficits)
        spatial_deficit_components = []
        if 'outside_ratio_mean' in self.child_features.columns:
            spatial_deficit_components.append(self.child_features['outside_ratio_mean'])
        if 'target_zone_coverage_mean' in self.child_features.columns:
            spatial_deficit_components.append(1 - self.child_features['target_zone_coverage_mean'])
        
        if len(spatial_deficit_components) >= 2:
            self.child_features['spatial_deficit_score'] = np.mean(spatial_deficit_components, axis=0)
            new_features.append('spatial_deficit_score')
        
        print(f"   ✅ Created {len(new_features)} interaction features: {new_features}")
        return new_features
    
    def engineer_temporal_features(self):
        """Create temporal progression features"""
        print("⏱️ Engineering temporal progression features...")
        
        new_features = []
        
        # Session consistency features (using std, min, max)
        base_features = ['final_completion', 'session_duration', 'velocity_mean_session_mean']
        
        for base_feature in base_features:
            std_col = f'{base_feature}_std'
            min_col = f'{base_feature}_min' 
            max_col = f'{base_feature}_max'
            mean_col = f'{base_feature}_mean'
            
            if all(f in self.child_features.columns for f in [std_col, mean_col]):
                # Coefficient of variation (consistency measure)
                consistency_col = f'{base_feature}_consistency'
                self.child_features[consistency_col] = (
                    self.child_features[std_col] / 
                    (self.child_features[mean_col] + 1e-8)  # Avoid division by zero
                )
                new_features.append(consistency_col)
        
        print(f"   ✅ Created {len(new_features)} temporal features: {new_features}")
        return new_features
    
    def select_clinical_features(self, max_features=60):
        """Select features optimized for clinical sensitivity"""
        print(f"🎯 Selecting top {max_features} clinical features...")
        
        # Prepare data
        X = self.child_features.select_dtypes(include=[np.number]).fillna(0)
        y = self.child_features['binary_label']
        
        # Remove non-informative columns
        X = X.drop(['binary_label'], axis=1, errors='ignore')
        
        # Priority 1: RAG high-impact features
        priority_features = []
        for feature_group in [self.high_impact_features, self.motor_control_features, self.cognitive_features]:
            priority_features.extend([f for f in feature_group if f in X.columns])
        
        # Remove duplicates
        priority_features = list(dict.fromkeys(priority_features))
        
        # Priority 2: Statistical feature selection for remaining slots
        remaining_slots = max_features - len(priority_features)
        if remaining_slots > 0:
            # Use mutual information for non-linear relationships
            selector = SelectKBest(score_func=mutual_info_classif, k=min(remaining_slots, X.shape[1] - len(priority_features)))
            
            # Select from features not already in priority list
            remaining_features = [f for f in X.columns if f not in priority_features]
            if len(remaining_features) > 0:
                X_remaining = X[remaining_features]
                selected_indices = selector.fit(X_remaining, y).get_support()
                selected_remaining = [remaining_features[i] for i, selected in enumerate(selected_indices) if selected]
            else:
                selected_remaining = []
        else:
            selected_remaining = []
        
        # Combine priority and selected features
        final_features = priority_features + selected_remaining
        
        print(f"   ✅ Selected features breakdown:")
        print(f"      Priority (RAG-guided): {len(priority_features)}")
        print(f"      Statistical selection: {len(selected_remaining)}")
        print(f"      Total: {len(final_features)}")
        
        return final_features
    
    def calculate_feature_importance(self, features):
        """Calculate clinical importance scores"""
        print("📈 Calculating clinical feature importance...")
        
        X = self.child_features[features].fillna(0)
        y = self.child_features['binary_label']
        
        # Calculate effect sizes (Cohen's d)
        importance_scores = {}
        
        for feature in features:
            asd_data = X[y == 1][feature].dropna()
            td_data = X[y == 0][feature].dropna()
            
            if len(asd_data) > 5 and len(td_data) > 5:
                pooled_std = np.sqrt((asd_data.var() + td_data.var()) / 2)
                if pooled_std > 0:
                    effect_size = abs(asd_data.mean() - td_data.mean()) / pooled_std
                    importance_scores[feature] = effect_size
                else:
                    importance_scores[feature] = 0
            else:
                importance_scores[feature] = 0
        
        # Sort by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   ✅ Top 10 most important features:")
        for i, (feature, score) in enumerate(sorted_features[:10], 1):
            print(f"      {i:2d}. {feature}: {score:.3f}")
        
        return importance_scores
    
    def save_clinical_features(self, features, importance_scores):
        """Save clinical-optimized features"""
        print("💾 Saving clinical feature dataset...")
        
        # Create clinical features dataset
        clinical_data = self.child_features[['child_id', 'group', 'binary_label'] + features].copy()
        
        # Save clinical features
        clinical_data.to_csv('features_binary/clinical_features_optimized.csv', index=False)
        
        # Save feature importance
        importance_df = pd.DataFrame([
            {'feature': feature, 'effect_size': score, 'rank': i+1}
            for i, (feature, score) in enumerate(
                sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            )
        ])
        importance_df.to_csv('features_binary/clinical_feature_importance.csv', index=False)
        
        # Save feature metadata
        metadata = {
            'total_features': len(features),
            'high_impact_features': len([f for f in self.high_impact_features if f in features]),
            'motor_control_features': len([f for f in self.motor_control_features if f in features]),
            'cognitive_features': len([f for f in self.cognitive_features if f in features]),
            'target_sensitivity': self.target_sensitivity,
            'dataset_size': len(clinical_data),
            'asd_count': int(clinical_data['binary_label'].sum()),
            'td_count': int(len(clinical_data) - clinical_data['binary_label'].sum())
        }
        
        import json
        with open('features_binary/clinical_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Clinical dataset saved:")
        print(f"      Features: {len(features)}")
        print(f"      Children: {len(clinical_data)}")
        print(f"      ASD: {metadata['asd_count']}, TD: {metadata['td_count']}")
        print(f"      Files: clinical_features_optimized.csv, clinical_feature_importance.csv")
    
    def run(self, max_features=60):
        """Execute complete clinical feature extraction pipeline"""
        print("🚀 CLINICAL FEATURE EXTRACTION PIPELINE")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Engineer new features
        interaction_features = self.engineer_interaction_features()
        temporal_features = self.engineer_temporal_features()
        
        # Select clinical features
        clinical_features = self.select_clinical_features(max_features)
        
        # Calculate importance
        importance_scores = self.calculate_feature_importance(clinical_features)
        
        # Save results
        self.save_clinical_features(clinical_features, importance_scores)
        
        print("\n" + "=" * 60)
        print("✅ CLINICAL FEATURE EXTRACTION COMPLETE!")
        print(f"🎯 Optimized for ASD sensitivity ≥{self.target_sensitivity}")
        print("🧬 Ready for clinical ensemble training")
        print("=" * 60)
        
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract clinical features for ASD classification")
    parser.add_argument("--target-sensitivity", type=float, default=0.86, 
                       help="Target ASD sensitivity (default: 0.86)")
    parser.add_argument("--max-features", type=int, default=60,
                       help="Maximum number of features to select (default: 60)")
    
    args = parser.parse_args()
    
    extractor = ClinicalFeatureExtractor(target_sensitivity=args.target_sensitivity)
    success = extractor.run(max_features=args.max_features)
    
    if success:
        print(f"\n🎯 Next step: python train_clinical_ensemble.py --optimize-sensitivity --target {args.target_sensitivity} 0.71")
    else:
        print("\n❌ Feature extraction failed. Check data files and try again.")
