#!/usr/bin/env python3
"""
Advanced RAG-Guided Feature Engineering for Clinical Targets
Extracts sophisticated behavioral biomarkers to reach ≥86% sensitivity & ≥71% specificity
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy import stats
from scipy.signal import find_peaks, welch
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

from rag_system.config import config as RAG_CFG

class AdvancedRAGFeatureEngineer:
    """Advanced RAG-guided feature engineering for clinical success"""
    
    def __init__(self):
        # Use project-relative, configurable paths
        self.raw_data_path = RAG_CFG.RAW_DATA_PATH
        self.labels_path = RAG_CFG.LABELS_PATH
        self.labels_dict = {}
        self.advanced_features = []
        
    def load_labels(self):
        """Load clinical labels"""
        print("📊 Loading clinical labels...")
        
        try:
            labels_df = pd.read_csv(self.labels_path)
            label_mapping = {'ASD_DD': 'ASD', 'ASD': 'ASD', 'DD': 'ASD', 'TD': 'TD'}
            labels_df['Binary_Label'] = labels_df['Group'].map(label_mapping)
            self.labels_dict = dict(zip(labels_df['Unity_id'], labels_df['Binary_Label']))
            
            print(f"   ✅ Loaded labels for {len(self.labels_dict)} children")
            print(f"   Distribution: ASD={sum(1 for v in self.labels_dict.values() if v == 'ASD')}, TD={sum(1 for v in self.labels_dict.values() if v == 'TD')}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading labels: {e}")
            return False
    
    def extract_child_id(self, filepath):
        """Extract child ID from filepath"""
        return filepath.parent.name
    
    def parse_session_file(self, filepath):
        """Parse raw session file with comprehensive extraction"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if 'json' not in data or 'touchData' not in data['json']:
                return None
                
            touch_data = data['json']['touchData']
            child_id = self.extract_child_id(filepath)
            label = self.labels_dict.get(child_id, 'UNKNOWN')
            
            if label == 'UNKNOWN':
                return None
            
            # Flatten all touch points with temporal ordering
            all_points = []
            for stroke_id, stroke_points in touch_data.items():
                for point in stroke_points:
                    point['stroke_id'] = stroke_id
                    all_points.append(point)
            
            # Sort by time for sequential analysis
            all_points.sort(key=lambda x: x.get('time', 0))
            
            return {
                'child_id': child_id,
                'label': label,
                'touch_data': touch_data,
                'all_points': all_points,
                'total_points': len(all_points),
                'total_strokes': len(touch_data)
            }
            
        except Exception as e:
            return None
    
    def extract_advanced_motor_features(self, session_data):
        """Extract sophisticated motor control biomarkers"""
        if not session_data or not session_data['all_points']:
            return {}
        
        points = session_data['all_points']
        features = {}
        
        # 1. MICRO-MOVEMENT ANALYSIS
        velocities, accelerations, jerks = [], [], []
        path_curvatures = []
        
        for i in range(2, len(points)):
            p_prev2, p_prev, p_curr = points[i-2], points[i-1], points[i]
            
            # Ensure we have coordinate and time data
            if all(k in p for k in ['x', 'y', 'time'] for p in [p_prev2, p_prev, p_curr]):
                dt1 = p_prev['time'] - p_prev2['time']
                dt2 = p_curr['time'] - p_prev['time']
                
                if dt1 > 0 and dt2 > 0:
                    # Velocity calculation
                    dx1, dy1 = p_prev['x'] - p_prev2['x'], p_prev['y'] - p_prev2['y']
                    dx2, dy2 = p_curr['x'] - p_prev['x'], p_curr['y'] - p_prev['y']
                    
                    v1 = np.sqrt(dx1**2 + dy1**2) / dt1
                    v2 = np.sqrt(dx2**2 + dy2**2) / dt2
                    
                    velocities.extend([v1, v2])
                    
                    # Acceleration
                    if dt1 + dt2 > 0:
                        acc = (v2 - v1) / ((dt1 + dt2) / 2)
                        accelerations.append(acc)
                    
                    # Jerk (change in acceleration)
                    if i > 2 and len(accelerations) > 1:
                        jerk = (accelerations[-1] - accelerations[-2]) / ((dt1 + dt2) / 2)
                        jerks.append(jerk)
                    
                    # Path curvature
                    if dx1 != 0 or dy1 != 0 and dx2 != 0 or dy2 != 0:
                        angle1 = np.arctan2(dy1, dx1)
                        angle2 = np.arctan2(dy2, dx2)
                        curvature = abs(angle2 - angle1)
                        if curvature > np.pi:
                            curvature = 2*np.pi - curvature
                        path_curvatures.append(curvature)
        
        # Motor control features
        if velocities:
            features.update({
                'velocity_mean': np.mean(velocities),
                'velocity_std': np.std(velocities),
                'velocity_cv': np.std(velocities) / np.mean(velocities) if np.mean(velocities) > 0 else 0,
                'velocity_skewness': stats.skew(velocities),
                'velocity_kurtosis': stats.kurtosis(velocities)
            })
        
        if accelerations:
            features.update({
                'acceleration_mean': np.mean(accelerations),
                'acceleration_std': np.std(accelerations),
                'acceleration_cv': np.std(accelerations) / np.mean(accelerations) if np.mean(accelerations) > 0 else 0
            })
        
        if jerks:
            features.update({
                'jerk_mean': np.mean(jerks),
                'jerk_std': np.std(jerks),
                'movement_smoothness': 1 / (1 + np.std(jerks))  # Higher = smoother
            })
        
        if path_curvatures:
            features.update({
                'path_curvature_mean': np.mean(path_curvatures),
                'path_straightness': 1 / (1 + np.mean(path_curvatures))  # Higher = straighter
            })
        
        # 2. TREMOR ANALYSIS (Frequency Domain)
        acc_x = [p.get('accx', 0) for p in points if 'accx' in p]
        acc_y = [p.get('accy', 0) for p in points if 'accy' in p]
        acc_z = [p.get('accz', 0) for p in points if 'accz' in p]
        
        if len(acc_x) > 50:  # Need sufficient data for frequency analysis
            # Power spectral density analysis
            fs = len(acc_x) / (points[-1]['time'] - points[0]['time']) if points[-1]['time'] != points[0]['time'] else 1
            
            try:
                freqs_x, psd_x = welch(acc_x, fs=fs, nperseg=min(len(acc_x)//4, 256))
                freqs_y, psd_y = welch(acc_y, fs=fs, nperseg=min(len(acc_y)//4, 256))
                freqs_z, psd_z = welch(acc_z, fs=fs, nperseg=min(len(acc_z)//4, 256))
                
                # Tremor frequency range (4-12 Hz typical)
                tremor_mask_x = (freqs_x >= 4) & (freqs_x <= 12)
                tremor_mask_y = (freqs_y >= 4) & (freqs_y <= 12)
                tremor_mask_z = (freqs_z >= 4) & (freqs_z <= 12)
                
                features.update({
                    'tremor_power_x': np.sum(psd_x[tremor_mask_x]) / np.sum(psd_x) if np.sum(psd_x) > 0 else 0,
                    'tremor_power_y': np.sum(psd_y[tremor_mask_y]) / np.sum(psd_y) if np.sum(psd_y) > 0 else 0,
                    'tremor_power_z': np.sum(psd_z[tremor_mask_z]) / np.sum(psd_z) if np.sum(psd_z) > 0 else 0,
                    'tremor_dominance': np.max([
                        np.sum(psd_x[tremor_mask_x]),
                        np.sum(psd_y[tremor_mask_y]), 
                        np.sum(psd_z[tremor_mask_z])
                    ])
                })
            except:
                # Fallback to time-domain tremor indicators
                features.update({
                    'acc_variability_x': np.std(acc_x),
                    'acc_variability_y': np.std(acc_y),
                    'acc_variability_z': np.std(acc_z),
                    'acc_magnitude_std': np.std([np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(acc_x, acc_y, acc_z)])
                })
        
        return features
    
    def extract_attention_planning_features(self, session_data):
        """Extract attention and executive function biomarkers"""
        if not session_data or not session_data['all_points']:
            return {}
        
        points = session_data['all_points']
        features = {}
        
        # 1. ATTENTION REGULATION ANALYSIS
        zones = [p.get('zone', 'Unknown') for p in points]
        colors = [p.get('color', 'Unknown') for p in points]
        
        # Zone transition patterns
        zone_transitions = []
        color_switches = []
        attention_lapses = []
        
        for i in range(1, len(zones)):
            if zones[i] != zones[i-1]:
                zone_transitions.append(i)
            if colors[i] != colors[i-1]:
                color_switches.append(i)
        
        # Attention metrics
        features.update({
            'zone_transition_rate': len(zone_transitions) / len(points) if points else 0,
            'color_switch_rate': len(color_switches) / len(points) if points else 0,
            'zone_persistence': np.mean([len(list(g)) for k, g in 
                                       __import__('itertools').groupby(zones)]) if zones else 0,
            'unique_zones_touched': len(set(zones)),
            'zone_entropy': stats.entropy([zones.count(z) for z in set(zones)]) if len(set(zones)) > 1 else 0
        })
        
        # 2. PAUSE AND RHYTHM ANALYSIS
        times = [p['time'] for p in points if 'time' in p]
        if len(times) > 1:
            inter_point_intervals = np.diff(times)
            
            # Identify pauses (intervals > 95th percentile)
            pause_threshold = np.percentile(inter_point_intervals, 95)
            pauses = inter_point_intervals[inter_point_intervals > pause_threshold]
            
            features.update({
                'num_pauses': len(pauses),
                'pause_duration_mean': np.mean(pauses) if len(pauses) > 0 else 0,
                'pause_frequency': len(pauses) / (times[-1] - times[0]) if times[-1] != times[0] else 0,
                'rhythm_regularity': 1 / (1 + np.std(inter_point_intervals)),  # Higher = more regular
                'session_tempo': len(times) / (times[-1] - times[0]) if times[-1] != times[0] else 0
            })
        
        # 3. TASK PROGRESSION ANALYSIS
        completion_percs = [p.get('completionPerc', 0) for p in points if 'completionPerc' in p]
        if len(completion_percs) > 5:
            # Linear fit to completion over time
            time_normalized = np.linspace(0, 1, len(completion_percs))
            slope, intercept, r_value, _, _ = stats.linregress(time_normalized, completion_percs)
            
            features.update({
                'completion_slope': slope,
                'completion_linearity': r_value**2,
                'completion_efficiency': (max(completion_percs) - min(completion_percs)) / len(completion_percs) if completion_percs else 0,
                'final_completion': max(completion_percs) if completion_percs else 0
            })
        
        return features
    
    def extract_spatial_accuracy_features(self, session_data):
        """Extract spatial processing and accuracy biomarkers"""
        if not session_data or not session_data['all_points']:
            return {}
        
        points = session_data['all_points']
        features = {}
        
        # 1. SPATIAL ACCURACY ANALYSIS
        zones = [p.get('zone', 'Unknown') for p in points]
        target_zones = ['Wall', 'Roof', 'Door', 'TopCake', 'BottomCake', 'TopIcing', 'BottomIcing', 'Cherry']
        outside_zones = ['Outside', 'Super outside', 'Bound']
        
        target_touches = sum(1 for z in zones if z in target_zones)
        outside_touches = sum(1 for z in zones if z in outside_zones)
        
        features.update({
            'spatial_accuracy': target_touches / len(zones) if zones else 0,
            'outside_boundary_ratio': outside_touches / len(zones) if zones else 0,
            'boundary_violations': outside_touches,
            'precision_score': target_touches / (target_touches + outside_touches) if (target_touches + outside_touches) > 0 else 0
        })
        
        # 2. COORDINATE ANALYSIS
        x_coords = [p['x'] for p in points if 'x' in p]
        y_coords = [p['y'] for p in points if 'y' in p]
        
        if x_coords and y_coords:
            # Spatial distribution
            features.update({
                'spatial_spread_x': np.max(x_coords) - np.min(x_coords),
                'spatial_spread_y': np.max(y_coords) - np.min(y_coords),
                'spatial_coverage_area': (np.max(x_coords) - np.min(x_coords)) * (np.max(y_coords) - np.min(y_coords)),
                'coordinate_variability_x': np.std(x_coords),
                'coordinate_variability_y': np.std(y_coords)
            })
            
            # Center of mass analysis
            center_x, center_y = np.mean(x_coords), np.mean(y_coords)
            distances_from_center = [euclidean((x, y), (center_x, center_y)) for x, y in zip(x_coords, y_coords)]
            
            features.update({
                'center_focus': 1 / (1 + np.mean(distances_from_center)),  # Higher = more focused
                'spatial_scatter': np.std(distances_from_center)
            })
        
        return features
    
    def extract_multi_touch_features(self, session_data):
        """Extract multi-touch and palm detection biomarkers"""
        if not session_data or not session_data['all_points']:
            return {}
        
        points = session_data['all_points']
        features = {}
        
        # Multi-touch analysis
        finger_ids = [p.get('fingerId', 0) for p in points]
        touch_phases = [p.get('touchPhase', 'Unknown') for p in points]
        
        # Finger usage patterns
        unique_fingers = len(set(finger_ids))
        max_finger_id = max(finger_ids) if finger_ids else 0
        
        # Palm touch detection (high finger IDs, canceled touches)
        potential_palm_touches = sum(1 for f in finger_ids if f > 2)  # Finger ID > 2 often indicates palm
        canceled_touches = sum(1 for p in touch_phases if p == 'Canceled')
        
        # Multi-finger coordination
        finger_switches = sum(1 for i in range(1, len(finger_ids)) if finger_ids[i] != finger_ids[i-1])
        
        features.update({
            'unique_fingers_used': unique_fingers,
            'max_finger_id': max_finger_id,
            'potential_palm_touches': potential_palm_touches,
            'palm_touch_ratio': potential_palm_touches / len(points) if points else 0,
            'canceled_touch_ratio': canceled_touches / len(points) if points else 0,
            'finger_switch_rate': finger_switches / len(points) if points else 0,
            'multi_touch_complexity': unique_fingers * (1 + finger_switches/len(points)) if points else 0
        })
        
        return features
    
    def create_composite_biomarkers(self, features):
        """Create composite biomarkers from individual features"""
        composite_features = {}
        
        # Motor Control Composite
        motor_features = ['velocity_cv', 'movement_smoothness', 'path_straightness', 'acc_magnitude_std']
        motor_values = [features.get(f, 0) for f in motor_features if f in features]
        if motor_values:
            composite_features['motor_control_deficit_score'] = np.mean([
                features.get('velocity_cv', 0),
                1 - features.get('movement_smoothness', 1),
                1 - features.get('path_straightness', 1),
                features.get('acc_magnitude_std', 0)
            ])
        
        # Attention Regulation Composite  
        attention_features = ['zone_transition_rate', 'color_switch_rate', 'num_pauses', 'rhythm_regularity']
        if all(f in features for f in attention_features[:2]):
            composite_features['attention_dysregulation_score'] = (
                features['zone_transition_rate'] + 
                features['color_switch_rate'] + 
                features.get('num_pauses', 0) * 0.1 +
                (1 - features.get('rhythm_regularity', 1))
            ) / 4
        
        # Spatial Processing Composite
        if 'outside_boundary_ratio' in features and 'spatial_accuracy' in features:
            composite_features['spatial_processing_deficit'] = (
                features['outside_boundary_ratio'] + 
                (1 - features['spatial_accuracy']) +
                features.get('spatial_scatter', 0) * 0.001
            ) / 3
        
        # Executive Function Composite
        if 'completion_linearity' in features and 'completion_efficiency' in features:
            composite_features['executive_function_score'] = (
                features['completion_linearity'] + 
                features['completion_efficiency'] +
                features.get('final_completion', 0) * 0.01
            ) / 3
        
        return composite_features
    
    def process_all_sessions(self, limit=None):
        """Process all sessions with advanced feature extraction"""
        print("🔬 Processing sessions for advanced RAG feature extraction...")
        
        all_features = []
        processed = 0
        
        for child_folder in self.raw_data_path.iterdir():
            if not child_folder.is_dir():
                continue
            
            coloring_files = list(child_folder.glob("Coloring_*.json"))
            
            for filepath in coloring_files:
                if limit and processed >= limit:
                    break
                
                session_data = self.parse_session_file(filepath)
                
                if session_data:
                    # Extract all feature categories
                    motor_features = self.extract_advanced_motor_features(session_data)
                    attention_features = self.extract_attention_planning_features(session_data)
                    spatial_features = self.extract_spatial_accuracy_features(session_data)
                    multitouch_features = self.extract_multi_touch_features(session_data)
                    
                    # Combine all features
                    combined_features = {
                        'child_id': session_data['child_id'],
                        'label': session_data['label'],
                        **motor_features,
                        **attention_features,
                        **spatial_features,
                        **multitouch_features
                    }
                    
                    # Add composite biomarkers
                    composite_features = self.create_composite_biomarkers(combined_features)
                    combined_features.update(composite_features)
                    
                    all_features.append(combined_features)
                    processed += 1
                    
                    if processed % 50 == 0:
                        print(f"   Processed {processed} sessions...")
        
        print(f"✅ Extracted advanced features from {len(all_features)} sessions")
        return pd.DataFrame(all_features)
    
    def run_advanced_extraction(self, output_path=None):
        """Execute advanced RAG feature extraction pipeline"""
        print("🚀 ADVANCED RAG FEATURE EXTRACTION")
        print("=" * 60)
        
        # Load labels
        if not self.load_labels():
            return False
        
        # Process all sessions
        features_df = self.process_all_sessions()
        
        if features_df.empty:
            print("❌ No features extracted")
            return False
        
        # Remove rows with unknown labels
        features_df = features_df[features_df['label'] != 'UNKNOWN']
        
        # Create binary labels
        features_df['binary_label'] = (features_df['label'] == 'ASD').astype(int)
        
        print(f"📊 Final dataset:")
        print(f"   Children: {features_df['child_id'].nunique()}")
        print(f"   Sessions: {len(features_df)}")
        print(f"   Features: {len([c for c in features_df.columns if c not in ['child_id', 'label', 'binary_label']])}")
        print(f"   ASD: {features_df['binary_label'].sum()}, TD: {len(features_df) - features_df['binary_label'].sum()}")
        
        # Resolve output path (project-relative by default)
        if output_path is None:
            out_dir = RAG_CFG.PROJECT_PATH / 'features_binary'
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / 'advanced_clinical_features.csv'
        else:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_df.to_csv(output_file, index=False)
        print(f"💾 Advanced features saved to: {output_file}")
        
        return True

if __name__ == "__main__":
    extractor = AdvancedRAGFeatureEngineer()
    success = extractor.run_advanced_extraction()
    
    if success:
        print("\n🎯 Next step: python clinical_ensemble_optimized.py")
    else:
        print("\n❌ Advanced feature extraction failed.")
