#!/usr/bin/env python3
"""
Advanced Feature Extraction for Kidaura Coloring Game Data
Extracts clinically relevant features for ASD/TD/DD classification
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats, signal
from scipy.spatial import distance
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Paths
from rag_system.config import config as RAG_CFG
DATA_PATH = RAG_CFG.RAW_DATA_PATH
LABELS_PATH = RAG_CFG.LABELS_PATH
OUTPUT_PATH = RAG_CFG.PROJECT_PATH / "extracted_features"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

class ColoringFeatureExtractor:
    """Extract clinically relevant features from coloring game data"""
    
    def __init__(self):
        self.labels_df = pd.read_csv(LABELS_PATH)
        self.labels_dict = dict(zip(self.labels_df['Unity_id'], self.labels_df['Group']))
        
    def parse_coloring_file(self, filepath):
        """Parse a single coloring file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            if 'json' in data and 'touchData' in data['json']:
                return data['json']['touchData']
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
    
    def extract_session_metadata(self, filepath):
        """Extract metadata from filename"""
        filename = filepath.name
        # Format: Coloring_YYYY-MM-DD HH:MM:SS.microseconds_childid.json
        parts = filename.replace('Coloring_', '').replace('.json', '').split('_')
        timestamp_str = parts[0] + ' ' + parts[1]
        
        try:
            timestamp = datetime.strptime(timestamp_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
        except:
            timestamp = None
            
        return {
            'timestamp': timestamp,
            'filename': filename
        }
    
    def calculate_stroke_features(self, stroke_points):
        """Calculate features for a single stroke"""
        if len(stroke_points) < 2:
            return None
            
        features = {}
        
        # Extract basic data
        times = [p['time'] for p in stroke_points]
        xs = [p['x'] for p in stroke_points]
        ys = [p['y'] for p in stroke_points]
        
        # Stroke duration
        features['duration'] = times[-1] - times[0]
        
        # Path length
        path_length = 0
        velocities = []
        accelerations = []
        angles = []
        
        for i in range(1, len(stroke_points)):
            # Distance between consecutive points
            dx = xs[i] - xs[i-1]
            dy = ys[i] - ys[i-1]
            dt = times[i] - times[i-1]
            
            dist = np.sqrt(dx**2 + dy**2)
            path_length += dist
            
            # Velocity
            if dt > 0:
                velocity = dist / dt
                velocities.append(velocity)
                
                # Angle of movement
                if dx != 0 or dy != 0:
                    angle = np.arctan2(dy, dx)
                    angles.append(angle)
        
        features['path_length'] = path_length
        
        # Velocity statistics
        if velocities:
            features['velocity_mean'] = np.mean(velocities)
            features['velocity_std'] = np.std(velocities)
            features['velocity_max'] = np.max(velocities)
            features['velocity_min'] = np.min(velocities)
            
            # Acceleration (change in velocity)
            for i in range(1, len(velocities)):
                dt = times[i] - times[i-1]
                if dt > 0:
                    acc = (velocities[i] - velocities[i-1]) / dt
                    accelerations.append(acc)
        
        # Acceleration statistics
        if accelerations:
            features['acceleration_mean'] = np.mean(accelerations)
            features['acceleration_std'] = np.std(accelerations)
            features['acceleration_max'] = np.max(np.abs(accelerations))
        
        # Jerkiness (number of direction changes)
        if angles:
            angle_changes = np.diff(angles)
            # Normalize to [-pi, pi]
            angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
            features['direction_changes'] = np.sum(np.abs(angle_changes) > np.pi/4)
            features['curvature_mean'] = np.mean(np.abs(angle_changes))
            features['curvature_std'] = np.std(angle_changes)
        
        # Smoothness (using spectral arc length)
        if len(velocities) > 10:
            try:
                freqs, psd = signal.periodogram(velocities)
                # Spectral arc length
                sal = -np.sum(np.sqrt(1 + np.diff(psd)**2))
                features['smoothness_sal'] = sal
            except:
                features['smoothness_sal'] = 0
        
        # Bounding box
        features['bbox_width'] = max(xs) - min(xs)
        features['bbox_height'] = max(ys) - min(ys)
        features['bbox_area'] = features['bbox_width'] * features['bbox_height']
        
        # Straightness (ratio of euclidean distance to path length)
        euclidean_dist = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        features['straightness'] = euclidean_dist / path_length if path_length > 0 else 0
        
        # Accelerometer features
        acc_features = self.calculate_accelerometer_features(stroke_points)
        features.update(acc_features)
        
        return features
    
    def calculate_accelerometer_features(self, stroke_points):
        """Calculate features from accelerometer data"""
        features = {}
        
        accx = [p.get('accx', 0) for p in stroke_points]
        accy = [p.get('accy', 0) for p in stroke_points]
        accz = [p.get('accz', 0) for p in stroke_points]
        
        # Magnitude of acceleration
        acc_magnitude = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(accx, accy, accz)]
        
        # Basic statistics
        for axis, values in [('x', accx), ('y', accy), ('z', accz), ('mag', acc_magnitude)]:
            if values:
                features[f'acc_{axis}_mean'] = np.mean(values)
                features[f'acc_{axis}_std'] = np.std(values)
                features[f'acc_{axis}_range'] = max(values) - min(values)
                
                # Tremor-like features (high frequency components)
                if len(values) > 10:
                    # Calculate power in different frequency bands
                    try:
                        freqs, psd = signal.periodogram(values, fs=30)  # Assume ~30Hz sampling
                        
                        # Low frequency (0-4 Hz) - normal movement
                        low_freq_idx = freqs < 4
                        features[f'acc_{axis}_power_low'] = np.sum(psd[low_freq_idx])
                        
                        # High frequency (4-12 Hz) - tremor
                        high_freq_idx = (freqs >= 4) & (freqs < 12)
                        features[f'acc_{axis}_power_high'] = np.sum(psd[high_freq_idx])
                        
                        # Tremor ratio
                        total_power = np.sum(psd)
                        if total_power > 0:
                            features[f'acc_{axis}_tremor_ratio'] = features[f'acc_{axis}_power_high'] / total_power
                    except:
                        pass
        
        return features
    
    def calculate_zone_features(self, touch_data):
        """Calculate features related to coloring zones"""
        features = {}
        
        # Zone sequence
        zone_sequence = []
        zone_times = defaultdict(float)
        zone_points = defaultdict(int)
        
        for stroke_id, stroke_points in touch_data.items():
            for point in stroke_points:
                zone = point.get('zone', 'Unknown')
                zone_sequence.append(zone)
                zone_points[zone] += 1
                
                # Calculate time spent in each zone
                if len(stroke_points) > 1:
                    idx = stroke_points.index(point)
                    if idx > 0:
                        dt = point['time'] - stroke_points[idx-1]['time']
                        zone_times[zone] += dt
        
        # Zone distribution
        total_points = sum(zone_points.values())
        for zone, count in zone_points.items():
            features[f'zone_{zone}_ratio'] = count / total_points if total_points > 0 else 0
        
        # Time distribution
        total_time = sum(zone_times.values())
        for zone, time in zone_times.items():
            features[f'zone_{zone}_time_ratio'] = time / total_time if total_time > 0 else 0
        
        # Zone transitions
        if len(zone_sequence) > 1:
            transitions = sum(1 for i in range(1, len(zone_sequence)) if zone_sequence[i] != zone_sequence[i-1])
            features['zone_transitions'] = transitions
            features['zone_transition_rate'] = transitions / len(zone_sequence)
        
        # Out of bounds ratio
        out_zones = ['Outside', 'Super outside', 'Bound']
        out_ratio = sum(zone_points.get(z, 0) for z in out_zones) / total_points if total_points > 0 else 0
        features['out_of_bounds_ratio'] = out_ratio
        
        # Zone entropy (measure of randomness in zone selection)
        if zone_sequence:
            zone_counts = Counter(zone_sequence)
            zone_probs = [count/len(zone_sequence) for count in zone_counts.values()]
            features['zone_entropy'] = stats.entropy(zone_probs)
        
        return features
    
    def calculate_color_features(self, touch_data):
        """Calculate features related to color usage"""
        features = {}
        
        color_sequence = []
        color_strokes = defaultdict(int)
        
        for stroke_id, stroke_points in touch_data.items():
            if stroke_points:
                # Get the dominant color for this stroke
                colors = [p.get('color', 'Unknown') for p in stroke_points]
                dominant_color = max(set(colors), key=colors.count)
                color_sequence.append(dominant_color)
                color_strokes[dominant_color] += 1
        
        # Number of different colors used
        features['num_colors_used'] = len(color_strokes)
        
        # Color switches
        if len(color_sequence) > 1:
            color_switches = sum(1 for i in range(1, len(color_sequence)) 
                               if color_sequence[i] != color_sequence[i-1])
            features['color_switches'] = color_switches
            features['color_switch_rate'] = color_switches / len(color_sequence)
        
        return features
    
    def calculate_completion_features(self, touch_data):
        """Calculate features related to completion progress"""
        features = {}
        
        # Get all completion percentages with timestamps
        completion_timeline = []
        
        for stroke_id, stroke_points in touch_data.items():
            for point in stroke_points:
                completion_timeline.append({
                    'time': point['time'],
                    'completion': point.get('completionPerc', 0)
                })
        
        # Sort by time
        completion_timeline.sort(key=lambda x: x['time'])
        
        if completion_timeline:
            # Final completion
            features['final_completion'] = completion_timeline[-1]['completion']
            
            # Completion rate
            total_time = completion_timeline[-1]['time'] - completion_timeline[0]['time']
            if total_time > 0:
                features['completion_rate'] = features['final_completion'] / total_time
            
            # Progress consistency (how steadily they progress)
            completions = [p['completion'] for p in completion_timeline]
            if len(completions) > 10:
                # Calculate autocorrelation of progress
                progress_diff = np.diff(completions)
                if len(progress_diff) > 1 and np.std(progress_diff) > 0:
                    features['progress_consistency'] = np.corrcoef(progress_diff[:-1], progress_diff[1:])[0, 1]
                
                # Regression line fit (R-squared)
                times = [p['time'] for p in completion_timeline]
                if len(times) > 2:
                    slope, intercept, r_value, _, _ = stats.linregress(times, completions)
                    features['progress_linearity'] = r_value**2
                    features['progress_slope'] = slope
        
        return features
    
    def calculate_session_features(self, touch_data):
        """Calculate session-level features"""
        features = {}
        
        # Basic counts
        features['num_strokes'] = len(touch_data)
        features['num_points'] = sum(len(points) for points in touch_data.values())
        
        # Time features
        all_times = []
        stroke_durations = []
        inter_stroke_intervals = []
        
        stroke_start_times = []
        for stroke_id, stroke_points in touch_data.items():
            if stroke_points:
                times = [p['time'] for p in stroke_points]
                all_times.extend(times)
                stroke_start_times.append(min(times))
                stroke_durations.append(max(times) - min(times))
        
        if all_times:
            features['session_duration'] = max(all_times) - min(all_times)
            features['points_per_second'] = features['num_points'] / features['session_duration'] if features['session_duration'] > 0 else 0
        
        # Stroke timing
        if stroke_durations:
            features['stroke_duration_mean'] = np.mean(stroke_durations)
            features['stroke_duration_std'] = np.std(stroke_durations)
            features['stroke_duration_max'] = np.max(stroke_durations)
        
        # Inter-stroke intervals
        if len(stroke_start_times) > 1:
            stroke_start_times.sort()
            intervals = np.diff(stroke_start_times)
            features['inter_stroke_interval_mean'] = np.mean(intervals)
            features['inter_stroke_interval_std'] = np.std(intervals)
            features['inter_stroke_interval_max'] = np.max(intervals)
        
        # Touch phase distribution
        phase_counts = defaultdict(int)
        for stroke_id, stroke_points in touch_data.items():
            for point in stroke_points:
                phase = point.get('touchPhase', 'Unknown')
                phase_counts[phase] += 1
        
        total_phases = sum(phase_counts.values())
        for phase, count in phase_counts.items():
            features[f'phase_{phase}_ratio'] = count / total_phases if total_phases > 0 else 0
        
        # Multi-touch usage
        finger_ids = set()
        for stroke_id, stroke_points in touch_data.items():
            for point in stroke_points:
                finger_ids.add(point.get('fingerId', 0))
        
        features['multi_touch_used'] = 1 if len(finger_ids) > 1 else 0
        features['num_fingers_used'] = len(finger_ids)
        
        return features
    
    def extract_all_features(self, child_id, session_files):
        """Extract features from all sessions for a child"""
        child_features = []
        
        for i, filepath in enumerate(sorted(session_files)):
            # Parse file
            touch_data = self.parse_coloring_file(filepath)
            if not touch_data:
                continue
            
            # Get metadata
            metadata = self.extract_session_metadata(filepath)
            
            # Initialize feature dict
            features = {
                'child_id': child_id,
                'session_number': i + 1,
                'filename': metadata['filename'],
                'timestamp': metadata['timestamp'],
                'group': self.labels_dict.get(child_id, 'Unknown')
            }
            
            # Extract session-level features
            session_features = self.calculate_session_features(touch_data)
            features.update(session_features)
            
            # Extract zone features
            zone_features = self.calculate_zone_features(touch_data)
            features.update(zone_features)
            
            # Extract color features
            color_features = self.calculate_color_features(touch_data)
            features.update(color_features)
            
            # Extract completion features
            completion_features = self.calculate_completion_features(touch_data)
            features.update(completion_features)
            
            # Extract stroke-level features and aggregate
            stroke_features_list = []
            for stroke_id, stroke_points in touch_data.items():
                if stroke_points and len(stroke_points) > 1:
                    stroke_feat = self.calculate_stroke_features(stroke_points)
                    if stroke_feat:
                        stroke_features_list.append(stroke_feat)
            
            # Aggregate stroke features
            if stroke_features_list:
                stroke_df = pd.DataFrame(stroke_features_list)
                for col in stroke_df.columns:
                    if stroke_df[col].dtype in ['float64', 'int64']:
                        features[f'stroke_{col}_mean'] = stroke_df[col].mean()
                        features[f'stroke_{col}_std'] = stroke_df[col].std()
                        features[f'stroke_{col}_max'] = stroke_df[col].max()
                        features[f'stroke_{col}_min'] = stroke_df[col].min()
            
            child_features.append(features)
        
        return child_features
    
    def process_all_children(self):
        """Process all children and extract features"""
        all_features = []
        
        # Get all child folders
        child_folders = [f for f in DATA_PATH.iterdir() if f.is_dir()]
        
        print(f"Processing {len(child_folders)} children...")
        
        for i, child_folder in enumerate(child_folders):
            child_id = child_folder.name
            
            # Skip if not in labels
            if child_id not in self.labels_dict:
                print(f"  Skipping {child_id} - no label found")
                continue
            
            # Get coloring files
            coloring_files = list(child_folder.glob("Coloring_*.json"))
            
            if not coloring_files:
                print(f"  No coloring files for {child_id}")
                continue
            
            print(f"  Processing {child_id} ({self.labels_dict[child_id]}) - {len(coloring_files)} files")
            
            # Extract features
            child_features = self.extract_all_features(child_id, coloring_files)
            all_features.extend(child_features)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(child_folders)} children")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Save features
        output_file = OUTPUT_PATH / 'coloring_features.csv'
        features_df.to_csv(output_file, index=False)
        print(f"\nSaved {len(features_df)} session features to {output_file}")
        
        # Save feature summary
        self.save_feature_summary(features_df)
        
        return features_df
    
    def save_feature_summary(self, features_df):
        """Save summary statistics of features by group"""
        # Get numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        # Group by diagnosis
        summary_stats = []
        for group in ['ASD', 'TD', 'DD']:
            group_df = features_df[features_df['group'] == group]
            
            stats_dict = {'group': group, 'n_sessions': len(group_df), 
                         'n_children': group_df['child_id'].nunique()}
            
            for col in numeric_cols:
                if col not in ['session_number']:
                    stats_dict[f'{col}_mean'] = group_df[col].mean()
                    stats_dict[f'{col}_std'] = group_df[col].std()
            
            summary_stats.append(stats_dict)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(OUTPUT_PATH / 'feature_summary_by_group.csv', index=False)
        
        # Save feature importance hints
        feature_importance = {
            'motor_control': [
                'stroke_velocity_mean', 'stroke_velocity_std',
                'stroke_acceleration_std', 'stroke_direction_changes',
                'stroke_smoothness_sal', 'stroke_straightness'
            ],
            'planning': [
                'zone_transition_rate', 'color_switch_rate',
                'progress_linearity', 'out_of_bounds_ratio'
            ],
            'tremor': [
                'stroke_acc_mag_tremor_ratio', 'stroke_acc_z_power_high',
                'stroke_acc_mag_std'
            ],
            'attention': [
                'session_duration', 'final_completion',
                'inter_stroke_interval_std', 'num_strokes'
            ]
        }
        
        with open(OUTPUT_PATH / 'feature_categories.json', 'w') as f:
            json.dump(feature_importance, f, indent=2)

def main():
    """Main extraction function"""
    extractor = ColoringFeatureExtractor()
    features_df = extractor.process_all_children()
    
    # Print summary
    print(f"\nExtraction complete!")
    print(f"Total sessions: {len(features_df)}")
    print(f"Feature columns: {len(features_df.columns)}")
    print(f"\nGroup distribution:")
    print(features_df['group'].value_counts())
    
    return features_df

if __name__ == "__main__":
    features_df = main()
