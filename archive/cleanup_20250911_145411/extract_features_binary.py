#!/usr/bin/env python3
"""
Improved Feature Extraction for Kidaura Coloring Game
Binary Classification: ASD+DD vs TD (3-6 years old children)
Focuses on comprehensive representation of touchscreen behavior
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats, signal
from scipy.spatial import distance, ConvexHull
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Paths
from rag_system.config import config as RAG_CFG  # reuse project-level paths

DATA_PATH = RAG_CFG.RAW_DATA_PATH
LABELS_PATH = RAG_CFG.LABELS_PATH
OUTPUT_PATH = RAG_CFG.PROJECT_PATH / "features_binary"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

class BinaryColoringFeatureExtractor:
    """Extract features for binary classification: ASD+DD vs TD"""
    
    def __init__(self):
        # Load labels and convert to binary
        self.labels_df = pd.read_csv(LABELS_PATH)
        self.labels_df['binary_group'] = self.labels_df['Group'].apply(
            lambda x: 'TD' if x == 'TD' else 'ASD_DD' if x in ['ASD', 'DD'] or 'ASD' in str(x) else 'Unknown'
        )
        self.labels_dict = dict(zip(self.labels_df['Unity_id'], self.labels_df['binary_group']))
        
        # Template zones for area calculation
        self.template_zones = {
            'cake': ['TopCake', 'BottomCake', 'TopIcing', 'BottomIcing', 'Cherry'],
            'hut': ['Wall', 'Roof', 'Door', 'Window'],
            'face': ['Head', 'Face', 'Hair', 'Eyes', 'Mouth']
        }
    
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
    
    def detect_template_type(self, touch_data):
        """Detect which template (hut or cake) based on zones"""
        all_zones = []
        for stroke_id, stroke_points in touch_data.items():
            for point in stroke_points:
                all_zones.append(point.get('zone', ''))
        
        zone_set = set(all_zones)
        
        # Check which template
        cake_zones = set(self.template_zones['cake'])
        hut_zones = set(self.template_zones['hut'])
        
        cake_overlap = len(zone_set.intersection(cake_zones))
        hut_overlap = len(zone_set.intersection(hut_zones))
        
        if cake_overlap > hut_overlap:
            return 'cake'
        elif hut_overlap > cake_overlap:
            return 'hut'
        else:
            return 'unknown'
    
    def extract_session_metadata(self, filepath):
        """Extract metadata including timestamp for session ordering"""
        filename = filepath.name
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
    
    def calculate_area_coverage(self, stroke_points):
        """Calculate area covered by a stroke using convex hull"""
        if len(stroke_points) < 3:
            return 0
        
        try:
            points = np.array([[p['x'], p['y']] for p in stroke_points])
            hull = ConvexHull(points)
            return hull.volume  # In 2D, volume is actually area
        except:
            return 0
    
    def detect_palm_touch(self, stroke_points):
        """Detect potential palm touches based on point density and area"""
        if len(stroke_points) < 5:
            return False
        
        # Calculate inter-point distances
        points = np.array([[p['x'], p['y']] for p in stroke_points])
        
        # If points are very close together in a large cluster, might be palm
        distances = []
        for i in range(len(points)-1):
            dist = np.linalg.norm(points[i+1] - points[i])
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        area = self.calculate_area_coverage(stroke_points)
        
        # Palm touches typically have small inter-point distances but large area
        if avg_distance < 10 and area > 5000:
            return True
        
        return False
    
    def calculate_stroke_features(self, stroke_points):
        """Enhanced stroke features including area coverage"""
        if len(stroke_points) < 2:
            return None
            
        features = {}
        
        # Extract basic data
        times = np.array([p['time'] for p in stroke_points])
        xs = np.array([p['x'] for p in stroke_points])
        ys = np.array([p['y'] for p in stroke_points])
        
        # Stroke duration
        features['duration'] = times[-1] - times[0]
        features['num_points'] = len(stroke_points)
        
        # Area coverage
        features['area_covered'] = self.calculate_area_coverage(stroke_points)
        features['is_palm_touch'] = self.detect_palm_touch(stroke_points)
        
        # Path features
        path_length = 0
        velocities = []
        accelerations = []
        angles = []
        curvatures = []
        
        for i in range(1, len(stroke_points)):
            dx = xs[i] - xs[i-1]
            dy = ys[i] - ys[i-1]
            dt = times[i] - times[i-1]
            
            dist = np.sqrt(dx**2 + dy**2)
            path_length += dist
            
            if dt > 0:
                velocity = dist / dt
                velocities.append(velocity)
                
                # Direction angle
                if dx != 0 or dy != 0:
                    angle = np.arctan2(dy, dx)
                    angles.append(angle)
        
        features['path_length'] = path_length
        
        # Efficiency: direct distance / path length
        if path_length > 0:
            direct_dist = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
            features['path_efficiency'] = direct_dist / path_length
        else:
            features['path_efficiency'] = 0
        
        # Velocity features
        if velocities:
            features['velocity_mean'] = np.mean(velocities)
            features['velocity_std'] = np.std(velocities)
            features['velocity_max'] = np.max(velocities)
            features['velocity_min'] = np.min(velocities)
            features['velocity_cv'] = features['velocity_std'] / features['velocity_mean'] if features['velocity_mean'] > 0 else 0
            
            # Acceleration
            for i in range(1, len(velocities)):
                dt = times[i+1] - times[i]
                if dt > 0:
                    acc = (velocities[i] - velocities[i-1]) / dt
                    accelerations.append(acc)
        
        # Acceleration features
        if accelerations:
            features['acceleration_mean'] = np.mean(accelerations)
            features['acceleration_std'] = np.std(accelerations)
            features['acceleration_max'] = np.max(np.abs(accelerations))
            features['jerk_score'] = np.sum(np.abs(accelerations)) / len(accelerations)
        
        # Angular features (curvature)
        if len(angles) > 1:
            for i in range(1, len(angles)):
                angle_change = angles[i] - angles[i-1]
                # Normalize to [-pi, pi]
                angle_change = (angle_change + np.pi) % (2 * np.pi) - np.pi
                curvatures.append(angle_change)
            
            features['curvature_mean'] = np.mean(np.abs(curvatures))
            features['curvature_std'] = np.std(curvatures)
            features['direction_changes'] = np.sum(np.abs(curvatures) > np.pi/4)
            features['curvature_energy'] = np.sum(np.array(curvatures)**2)
        
        # Pressure proxy from accelerometer
        acc_features = self.calculate_pressure_features(stroke_points)
        features.update(acc_features)
        
        # Touch phase analysis
        phases = [p.get('touchPhase', '') for p in stroke_points]
        phase_counts = Counter(phases)
        features['has_canceled'] = phase_counts.get('Canceled', 0) > 0
        
        return features
    
    def calculate_pressure_features(self, stroke_points):
        """Extract pressure-related features from accelerometer"""
        features = {}
        
        accx = np.array([p.get('accx', 0) for p in stroke_points])
        accy = np.array([p.get('accy', 0) for p in stroke_points])
        accz = np.array([p.get('accz', 0) for p in stroke_points])
        
        # Z-axis is most relevant for pressure (perpendicular to screen)
        # More negative z values typically indicate more pressure
        features['pressure_mean'] = -np.mean(accz)  # Invert so higher = more pressure
        features['pressure_std'] = np.std(accz)
        features['pressure_range'] = np.max(accz) - np.min(accz)
        
        # Device tilt from accelerometer
        acc_magnitude = np.sqrt(accx**2 + accy**2 + accz**2)
        features['device_stability'] = 1.0 / (np.std(acc_magnitude) + 0.001)
        
        # Tremor detection
        if len(accz) > 10:
            # FFT for frequency analysis
            try:
                freqs = np.fft.fftfreq(len(accz), d=1/30.0)  # Assume 30Hz sampling
                fft = np.abs(np.fft.fft(accz))
                
                # Tremor typically 4-12 Hz
                tremor_mask = (np.abs(freqs) >= 4) & (np.abs(freqs) <= 12)
                normal_mask = (np.abs(freqs) < 4)
                
                if np.sum(fft[normal_mask]) > 0:
                    features['tremor_ratio'] = np.sum(fft[tremor_mask]) / np.sum(fft[normal_mask])
                else:
                    features['tremor_ratio'] = 0
                    
                # Peak tremor frequency
                if np.any(tremor_mask):
                    tremor_freqs = freqs[tremor_mask]
                    tremor_powers = fft[tremor_mask]
                    peak_idx = np.argmax(tremor_powers)
                    features['tremor_peak_freq'] = abs(tremor_freqs[peak_idx])
                else:
                    features['tremor_peak_freq'] = 0
            except:
                features['tremor_ratio'] = 0
                features['tremor_peak_freq'] = 0
        
        return features
    
    def calculate_multitouch_features(self, touch_data):
        """Analyze multi-touch patterns"""
        features = {}
        
        # Finger ID analysis
        all_finger_ids = []
        finger_strokes = defaultdict(int)
        
        for stroke_id, stroke_points in touch_data.items():
            finger_ids = set()
            for point in stroke_points:
                fid = point.get('fingerId', 0)
                finger_ids.add(fid)
                all_finger_ids.append(fid)
            
            # Count strokes per finger
            for fid in finger_ids:
                finger_strokes[fid] += 1
        
        features['num_unique_fingers'] = len(set(all_finger_ids))
        features['multitouch_used'] = features['num_unique_fingers'] > 1
        features['max_finger_id'] = max(all_finger_ids) if all_finger_ids else 0
        
        # Palm touch indicator (high finger IDs suggest palm)
        features['potential_palm_touches'] = sum(1 for fid in all_finger_ids if fid > 3)
        
        # Finger switching frequency
        if len(all_finger_ids) > 1:
            finger_switches = sum(1 for i in range(1, len(all_finger_ids)) 
                                if all_finger_ids[i] != all_finger_ids[i-1])
            features['finger_switch_rate'] = finger_switches / len(all_finger_ids)
        else:
            features['finger_switch_rate'] = 0
        
        return features
    
    def calculate_zone_coverage_features(self, touch_data, template_type):
        """Calculate how well child covers different zones"""
        features = {}
        
        # Zone-specific features
        zone_points = defaultdict(list)
        zone_completion = defaultdict(float)
        
        for stroke_id, stroke_points in touch_data.items():
            for point in stroke_points:
                zone = point.get('zone', 'Unknown')
                zone_points[zone].append(point)
                
                # Track max completion per zone
                completion = point.get('completionPerc', 0)
                zone_completion[zone] = max(zone_completion[zone], completion)
        
        # Get template-specific zones
        if template_type in self.template_zones:
            target_zones = self.template_zones[template_type]
            
            # Coverage ratio for each target zone
            total_target_points = 0
            total_outside_points = 0
            
            for zone, points in zone_points.items():
                if zone in target_zones:
                    total_target_points += len(points)
                elif zone in ['Outside', 'Super outside', 'Bound']:
                    total_outside_points += len(points)
            
            total_points = sum(len(points) for points in zone_points.values())
            
            features['target_zone_coverage'] = total_target_points / total_points if total_points > 0 else 0
            features['outside_ratio'] = total_outside_points / total_points if total_points > 0 else 0
            
            # Zone completion uniformity
            if target_zones:
                zone_coverages = []
                for zone in target_zones:
                    coverage = len(zone_points.get(zone, [])) / total_points if total_points > 0 else 0
                    zone_coverages.append(coverage)
                
                features['zone_coverage_std'] = np.std(zone_coverages)
                features['zone_coverage_entropy'] = stats.entropy(zone_coverages + [0.001])  # Add small value to avoid log(0)
        
        # Zone transition patterns
        zone_sequence = []
        for stroke_id, stroke_points in touch_data.items():
            for point in stroke_points:
                zone_sequence.append(point.get('zone', 'Unknown'))
        
        if len(zone_sequence) > 1:
            transitions = sum(1 for i in range(1, len(zone_sequence)) 
                            if zone_sequence[i] != zone_sequence[i-1])
            features['zone_transitions'] = transitions
            features['zone_transition_rate'] = transitions / len(zone_sequence)
            
            # Zone revisits
            zone_visits = []
            current_zone = None
            for zone in zone_sequence:
                if zone != current_zone:
                    zone_visits.append(zone)
                    current_zone = zone
            
            zone_visit_counts = Counter(zone_visits)
            features['zones_revisited'] = sum(1 for count in zone_visit_counts.values() if count > 1)
            features['avg_zone_visits'] = np.mean(list(zone_visit_counts.values()))
        
        return features
    
    def calculate_temporal_features(self, touch_data):
        """Extract temporal patterns and progression features"""
        features = {}
        
        # Get all timestamps and completion values
        time_completion_pairs = []
        all_times = []
        
        for stroke_id, stroke_points in touch_data.items():
            for point in stroke_points:
                time = point['time']
                completion = point.get('completionPerc', 0)
                time_completion_pairs.append((time, completion))
                all_times.append(time)
        
        if not all_times:
            return features
        
        # Sort by time
        time_completion_pairs.sort(key=lambda x: x[0])
        times, completions = zip(*time_completion_pairs)
        times = np.array(times)
        completions = np.array(completions)
        
        # Session duration
        session_duration = times[-1] - times[0]
        features['session_duration'] = session_duration
        
        # Completion features
        features['final_completion'] = completions[-1]
        features['completion_rate'] = features['final_completion'] / session_duration if session_duration > 0 else 0
        
        # Progress consistency
        if len(times) > 10:
            # Fit linear regression to see how linear progress is
            slope, intercept, r_value, _, _ = stats.linregress(times, completions)
            features['progress_linearity'] = r_value**2
            features['progress_slope'] = slope
            
            # Completion plateaus (periods of no progress)
            completion_diffs = np.diff(completions)
            plateau_lengths = []
            current_plateau = 0
            
            for diff in completion_diffs:
                if abs(diff) < 0.1:  # No significant progress
                    current_plateau += 1
                else:
                    if current_plateau > 0:
                        plateau_lengths.append(current_plateau)
                    current_plateau = 0
            
            if plateau_lengths:
                features['avg_plateau_length'] = np.mean(plateau_lengths)
                features['max_plateau_length'] = np.max(plateau_lengths)
            else:
                features['avg_plateau_length'] = 0
                features['max_plateau_length'] = 0
        
        # Stroke timing patterns
        stroke_times = []
        for stroke_id, stroke_points in touch_data.items():
            if stroke_points:
                stroke_start = min(p['time'] for p in stroke_points)
                stroke_times.append(stroke_start)
        
        stroke_times.sort()
        
        if len(stroke_times) > 1:
            inter_stroke_intervals = np.diff(stroke_times)
            features['inter_stroke_mean'] = np.mean(inter_stroke_intervals)
            features['inter_stroke_std'] = np.std(inter_stroke_intervals)
            features['inter_stroke_cv'] = features['inter_stroke_std'] / features['inter_stroke_mean'] if features['inter_stroke_mean'] > 0 else 0
            
            # Detect pauses (long inter-stroke intervals)
            pause_threshold = np.mean(inter_stroke_intervals) + 2 * np.std(inter_stroke_intervals)
            features['num_pauses'] = sum(1 for interval in inter_stroke_intervals if interval > pause_threshold)
        
        return features
    
    def calculate_session_features(self, touch_data, template_type):
        """Calculate comprehensive session-level features"""
        features = {
            'template_type': template_type,
            'num_strokes': len(touch_data),
            'num_points': sum(len(points) for points in touch_data.values())
        }
        
        # Multitouch features
        multitouch_features = self.calculate_multitouch_features(touch_data)
        features.update(multitouch_features)
        
        # Zone coverage features
        zone_features = self.calculate_zone_coverage_features(touch_data, template_type)
        features.update(zone_features)
        
        # Temporal features
        temporal_features = self.calculate_temporal_features(touch_data)
        features.update(temporal_features)
        
        # Color usage patterns
        color_sequence = []
        for stroke_id, stroke_points in touch_data.items():
            if stroke_points:
                colors = [p.get('color', 'Unknown') for p in stroke_points]
                dominant_color = max(set(colors), key=colors.count)
                color_sequence.append(dominant_color)
        
        features['num_colors_used'] = len(set(color_sequence))
        
        if len(color_sequence) > 1:
            color_switches = sum(1 for i in range(1, len(color_sequence)) 
                               if color_sequence[i] != color_sequence[i-1])
            features['color_switches'] = color_switches
            features['color_switch_rate'] = color_switches / len(color_sequence)
        
        # Aggregate stroke features
        stroke_features_list = []
        total_area = 0
        palm_touches = 0
        
        for stroke_id, stroke_points in touch_data.items():
            if stroke_points and len(stroke_points) > 1:
                stroke_feat = self.calculate_stroke_features(stroke_points)
                if stroke_feat:
                    stroke_features_list.append(stroke_feat)
                    total_area += stroke_feat.get('area_covered', 0)
                    palm_touches += int(stroke_feat.get('is_palm_touch', False))
        
        features['total_area_covered'] = total_area
        features['num_palm_touches'] = palm_touches
        
        # Aggregate stroke statistics
        if stroke_features_list:
            stroke_df = pd.DataFrame(stroke_features_list)
            
            # Key metrics to aggregate
            metrics = [
                'velocity_mean', 'velocity_std', 'velocity_cv',
                'acceleration_std', 'jerk_score',
                'path_efficiency', 'curvature_mean', 'direction_changes',
                'pressure_mean', 'pressure_std', 'tremor_ratio',
                'device_stability'
            ]
            
            for metric in metrics:
                if metric in stroke_df.columns:
                    values = stroke_df[metric].dropna()
                    if len(values) > 0:
                        features[f'{metric}_session_mean'] = values.mean()
                        features[f'{metric}_session_std'] = values.std()
                        features[f'{metric}_session_max'] = values.max()
                        features[f'{metric}_session_min'] = values.min()
        
        return features
    
    def extract_child_level_features(self, child_id, session_features_list):
        """Aggregate session features to child level"""
        if not session_features_list:
            return None
        
        # Sort sessions by timestamp to analyze progression
        session_features_list.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        child_features = {
            'child_id': child_id,
            'num_sessions': len(session_features_list),
            'group': session_features_list[0]['group']
        }
        
        # Aggregate numeric features
        numeric_cols = []
        for feat in session_features_list[0].keys():
            if isinstance(session_features_list[0][feat], (int, float)):
                numeric_cols.append(feat)
        
        # Calculate statistics across sessions
        for col in numeric_cols:
            values = [s[col] for s in session_features_list if col in s]
            if values:
                child_features[f'{col}_mean'] = np.mean(values)
                child_features[f'{col}_std'] = np.std(values)
                child_features[f'{col}_min'] = np.min(values)
                child_features[f'{col}_max'] = np.max(values)
                
                # Progression features (change over time)
                if len(values) > 1:
                    child_features[f'{col}_first_last_diff'] = values[-1] - values[0]
                    child_features[f'{col}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
        
        # Session consistency
        if len(session_features_list) > 1:
            # Calculate consistency of key metrics across sessions
            consistency_metrics = ['final_completion', 'session_duration', 'velocity_mean_session_mean']
            
            for metric in consistency_metrics:
                values = [s.get(metric, 0) for s in session_features_list]
                if len(values) > 1 and np.std(values) > 0:
                    child_features[f'{metric}_consistency'] = 1.0 / (np.std(values) / np.mean(values))
        
        return child_features
    
    def process_all_children(self):
        """Process all children and extract features"""
        all_child_features = []
        all_session_features = []
        
        # Get all child folders
        child_folders = [f for f in DATA_PATH.iterdir() if f.is_dir()]
        
        print(f"Processing {len(child_folders)} children...")
        
        for i, child_folder in enumerate(child_folders):
            child_id = child_folder.name
            
            # Skip if not in labels or unknown group
            if child_id not in self.labels_dict or self.labels_dict[child_id] == 'Unknown':
                continue
            
            # Get coloring files
            coloring_files = list(child_folder.glob("Coloring_*.json"))
            
            if not coloring_files:
                continue
            
            group = self.labels_dict[child_id]
            print(f"  Processing {child_id} ({group}) - {len(coloring_files)} files")
            
            # Extract features from each session
            session_features_list = []
            
            for filepath in sorted(coloring_files):
                # Parse file
                touch_data = self.parse_coloring_file(filepath)
                if not touch_data:
                    continue
                
                # Get metadata
                metadata = self.extract_session_metadata(filepath)
                
                # Detect template type
                template_type = self.detect_template_type(touch_data)
                
                # Extract session features
                features = self.calculate_session_features(touch_data, template_type)
                features['child_id'] = child_id
                features['filename'] = metadata['filename']
                features['timestamp'] = metadata['timestamp']
                features['group'] = group
                
                session_features_list.append(features)
                all_session_features.append(features)
            
            # Extract child-level features
            if session_features_list:
                child_features = self.extract_child_level_features(child_id, session_features_list)
                if child_features:
                    all_child_features.append(child_features)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(child_folders)} children")
        
        # Save both session and child level features
        if all_session_features:
            session_df = pd.DataFrame(all_session_features)
            session_df.to_csv(OUTPUT_PATH / 'session_features_binary.csv', index=False)
            print(f"\nSaved {len(session_df)} session features")
        
        if all_child_features:
            child_df = pd.DataFrame(all_child_features)
            child_df.to_csv(OUTPUT_PATH / 'child_features_binary.csv', index=False)
            print(f"Saved {len(child_df)} child-level features")
            
            # Save summary by group
            self.save_feature_summary(child_df)
        
        return session_df, child_df
    
    def save_feature_summary(self, features_df):
        """Save summary statistics by group"""
        summary = features_df.groupby('group').agg(['mean', 'std', 'count']).round(4)
        summary.to_csv(OUTPUT_PATH / 'feature_summary_by_group.csv')
        
        # Key features comparison
        key_features = [
            'velocity_mean_session_mean_mean',
            'acceleration_std_session_mean_mean', 
            'tremor_ratio_session_mean_mean',
            'outside_ratio_mean',
            'final_completion_mean',
            'zone_transition_rate_mean',
            'pressure_std_session_mean_mean'
        ]
        
        print("\nKey Features Comparison (ASD+DD vs TD):")
        print("="*60)
        
        for feat in key_features:
            if feat in features_df.columns:
                asd_dd = features_df[features_df['group'] == 'ASD_DD'][feat].mean()
                td = features_df[features_df['group'] == 'TD'][feat].mean()
                
                if td != 0:
                    ratio = asd_dd / td
                    print(f"{feat:40} ASD_DD/TD ratio: {ratio:.3f}")

def main():
    """Main extraction function"""
    extractor = BinaryColoringFeatureExtractor()
    session_df, child_df = extractor.process_all_children()
    
    print("\nExtraction complete!")
    if child_df is not None and len(child_df) > 0:
        print(f"\nGroup distribution:")
        print(child_df['group'].value_counts())
    
    return session_df, child_df

if __name__ == "__main__":
    session_df, child_df = main()
