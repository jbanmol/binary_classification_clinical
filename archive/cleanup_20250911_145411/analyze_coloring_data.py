#!/usr/bin/env python3
"""
Kidaura Coloring Game Data Analysis
Analyzes touchscreen data from coloring game sessions
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = Path("/Users/jbanmol/Desktop/Kidaura/S3_extration/data/fileKeys")
REPORT_PATH = Path("/Users/jbanmol/binary-classification-project/coloring_data_report")
REPORT_PATH.mkdir(exist_ok=True)

def get_file_statistics():
    """Get basic statistics about the files"""
    stats = {
        'total_children': 0,
        'total_coloring_files': 0,
        'files_per_child': [],
        'file_sizes': [],
        'child_ids': []
    }
    
    # Iterate through each child folder
    for child_folder in DATA_PATH.iterdir():
        if child_folder.is_dir():
            stats['total_children'] += 1
            stats['child_ids'].append(child_folder.name)
            
            # Count coloring files for this child
            coloring_files = list(child_folder.glob("Coloring_*.json"))
            num_files = len(coloring_files)
            stats['files_per_child'].append(num_files)
            stats['total_coloring_files'] += num_files
            
            # Get file sizes
            for file in coloring_files:
                stats['file_sizes'].append(file.stat().st_size / 1024)  # KB
    
    return stats

def parse_coloring_file(filepath):
    """Parse a single coloring file and extract data"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'json' in data and 'touchData' in data['json']:
            touch_data = data['json']['touchData']
            return touch_data
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def analyze_touch_data(touch_data):
    """Analyze touch data from a coloring session"""
    session_stats = {
        'total_strokes': len(touch_data),
        'total_points': 0,
        'session_duration': 0,
        'colors_used': set(),
        'zones_touched': set(),
        'completion_progress': [],
        'stroke_lengths': [],
        'stroke_durations': []
    }
    
    for stroke_id, stroke_points in touch_data.items():
        if not stroke_points:
            continue
            
        session_stats['total_points'] += len(stroke_points)
        
        # Get stroke statistics
        first_point = stroke_points[0]
        last_point = stroke_points[-1]
        
        # Duration of this stroke
        stroke_duration = last_point['time'] - first_point['time']
        session_stats['stroke_durations'].append(stroke_duration)
        
        # Track colors and zones
        for point in stroke_points:
            session_stats['colors_used'].add(point.get('color', 'Unknown'))
            session_stats['zones_touched'].add(point.get('zone', 'Unknown'))
            session_stats['completion_progress'].append(point.get('completionPerc', 0))
        
        # Calculate stroke length
        if len(stroke_points) > 1:
            length = 0
            for i in range(1, len(stroke_points)):
                dx = stroke_points[i]['x'] - stroke_points[i-1]['x']
                dy = stroke_points[i]['y'] - stroke_points[i-1]['y']
                length += np.sqrt(dx**2 + dy**2)
            session_stats['stroke_lengths'].append(length)
    
    # Calculate total session duration
    all_times = []
    for stroke_points in touch_data.values():
        for point in stroke_points:
            all_times.append(point['time'])
    
    if all_times:
        session_stats['session_duration'] = max(all_times) - min(all_times)
    
    return session_stats

def analyze_sample_files(sample_size=10):
    """Analyze a sample of coloring files"""
    all_sessions = []
    
    # Get sample of children
    child_folders = list(DATA_PATH.iterdir())[:sample_size]
    
    for child_folder in child_folders:
        if not child_folder.is_dir():
            continue
            
        coloring_files = list(child_folder.glob("Coloring_*.json"))
        
        for file_path in coloring_files[:2]:  # Max 2 files per child for sample
            touch_data = parse_coloring_file(file_path)
            
            if touch_data:
                session_stats = analyze_touch_data(touch_data)
                session_stats['child_id'] = child_folder.name
                session_stats['filename'] = file_path.name
                all_sessions.append(session_stats)
    
    return all_sessions

def extract_point_features(sample_files=5):
    """Extract detailed features from touch points"""
    all_points = []
    
    # Get a few sample files
    for child_folder in list(DATA_PATH.iterdir())[:3]:
        if not child_folder.is_dir():
            continue
            
        coloring_files = list(child_folder.glob("Coloring_*.json"))[:2]
        
        for file_path in coloring_files:
            touch_data = parse_coloring_file(file_path)
            
            if touch_data:
                for stroke_id, stroke_points in touch_data.items():
                    for i, point in enumerate(stroke_points):
                        point_features = {
                            'child_id': child_folder.name,
                            'stroke_id': stroke_id,
                            'point_index': i,
                            'x': point.get('x', 0),
                            'y': point.get('y', 0),
                            'time': point.get('time', 0),
                            'touchPhase': point.get('touchPhase', ''),
                            'fingerId': point.get('fingerId', 0),
                            'accx': point.get('accx', 0),
                            'accy': point.get('accy', 0),
                            'accz': point.get('accz', 0),
                            'color': point.get('color', ''),
                            'zone': point.get('zone', ''),
                            'completionPerc': point.get('completionPerc', 0)
                        }
                        all_points.append(point_features)
    
    return pd.DataFrame(all_points)

def create_visualizations(stats, session_data, points_df):
    """Create visualizations for the report"""
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    
    # 1. Distribution of files per child
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.hist(stats['files_per_child'], bins=range(0, max(stats['files_per_child'])+2), 
             edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Number of Coloring Files')
    ax1.set_ylabel('Number of Children')
    ax1.set_title('Distribution of Coloring Files per Child')
    ax1.grid(True, alpha=0.3)
    
    # 2. File size distribution
    ax2.hist(stats['file_sizes'], bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('File Size (KB)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Coloring File Sizes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORT_PATH / 'file_statistics.png', dpi=150)
    plt.close()
    
    # 3. Touch zones frequency
    if not points_df.empty:
        plt.figure(figsize=(12, 6))
        zone_counts = points_df['zone'].value_counts()
        zone_counts.plot(kind='bar')
        plt.xlabel('Touch Zone')
        plt.ylabel('Frequency')
        plt.title('Frequency of Touch Points by Zone')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(REPORT_PATH / 'zone_distribution.png', dpi=150)
        plt.close()
        
        # 4. Touch phase distribution
        plt.figure(figsize=(10, 6))
        phase_counts = points_df['touchPhase'].value_counts()
        phase_counts.plot(kind='bar')
        plt.xlabel('Touch Phase')
        plt.ylabel('Frequency')
        plt.title('Distribution of Touch Phases')
        plt.tight_layout()
        plt.savefig(REPORT_PATH / 'touch_phases.png', dpi=150)
        plt.close()
        
        # 5. Completion progress over time (sample)
        sample_child = points_df['child_id'].unique()[0]
        sample_data = points_df[points_df['child_id'] == sample_child].copy()
        
        if not sample_data.empty:
            plt.figure(figsize=(12, 6))
            plt.scatter(sample_data['time'], sample_data['completionPerc'], 
                       alpha=0.5, s=10)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Completion Percentage')
            plt.title(f'Completion Progress Over Time - Sample Child')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(REPORT_PATH / 'completion_progress.png', dpi=150)
            plt.close()

def generate_report(stats, session_data, points_df):
    """Generate comprehensive analysis report"""
    report = f"""# Kidaura Coloring Game Data Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Dataset Overview

### File Statistics
- **Total number of children**: {stats['total_children']}
- **Total coloring files**: {stats['total_coloring_files']}
- **Average files per child**: {np.mean(stats['files_per_child']):.2f}
- **File size range**: {min(stats['file_sizes']):.1f} - {max(stats['file_sizes']):.1f} KB
- **Average file size**: {np.mean(stats['file_sizes']):.1f} KB

### Files per Child Distribution
- Minimum: {min(stats['files_per_child'])}
- Maximum: {max(stats['files_per_child'])}
- Median: {np.median(stats['files_per_child']):.0f}

## 2. Data Structure Analysis

### JSON Structure
Each coloring file contains:
- **message**: "gameData"
- **json**: Object containing:
  - **dataSet**: "Coloring"
  - **touchData**: Dictionary of strokes

### Touch Point Fields
Each touch point contains the following fields:
- **x, y**: Screen coordinates (pixels)
- **time**: Timestamp in seconds since session start
- **touchPhase**: Touch event type (Began, Moved, Stationary, Ended, Canceled)
- **fingerId**: Identifier for multi-touch support
- **accx, accy, accz**: Accelerometer data (device motion)
- **color**: Current color being used (e.g., "RedDefault", "Green")
- **zone**: Area being colored (e.g., "TopCake", "BottomIcing", "Cherry", "Outside", "Bound")
- **completionPerc**: Percentage of coloring completed

## 3. Touch Data Characteristics
"""

    if not points_df.empty:
        report += f"""
### Coordinate Ranges
- X coordinates: {points_df['x'].min():.0f} - {points_df['x'].max():.0f}
- Y coordinates: {points_df['y'].min():.0f} - {points_df['y'].max():.0f}

### Touch Phases Distribution
{points_df['touchPhase'].value_counts().to_string()}

### Zone Distribution
{points_df['zone'].value_counts().head(10).to_string()}

### Colors Used
{points_df['color'].value_counts().to_string()}

### Accelerometer Data Ranges
- accx: [{points_df['accx'].min():.6f}, {points_df['accx'].max():.6f}]
- accy: [{points_df['accy'].min():.6f}, {points_df['accy'].max():.6f}]
- accz: [{points_df['accz'].min():.6f}, {points_df['accz'].max():.6f}]
"""

    # Session analysis
    if session_data:
        total_strokes = [s['total_strokes'] for s in session_data]
        session_durations = [s['session_duration'] for s in session_data if s['session_duration'] > 0]
        
        report += f"""
## 4. Session-Level Analysis

### Stroke Statistics
- Average strokes per session: {np.mean(total_strokes):.1f}
- Stroke count range: {min(total_strokes)} - {max(total_strokes)}

### Session Duration
- Average session duration: {np.mean(session_durations):.1f} seconds
- Duration range: {min(session_durations):.1f} - {max(session_durations):.1f} seconds
"""

    report += """
## 5. Feature Engineering Recommendations

Based on the data analysis, the following features could be extracted for classification:

### Motion Features
1. **Stroke velocity**: Distance/time between consecutive points
2. **Stroke acceleration**: Change in velocity
3. **Stroke curvature**: Angle changes along the path
4. **Jerkiness**: Frequency of direction changes

### Spatial Features
1. **Stroke length**: Total distance covered in a stroke
2. **Bounding box**: Area covered by each stroke
3. **Zone transitions**: Number of times child moves between zones
4. **Out-of-bounds percentage**: Time spent coloring outside designated areas

### Temporal Features
1. **Time per stroke**: Duration of each stroke
2. **Inter-stroke interval**: Time between strokes
3. **Coloring speed**: Area covered per unit time
4. **Completion rate**: Change in completion percentage over time

### Pressure/Motion Features
1. **Accelerometer variability**: Standard deviation of acc values during strokes
2. **Device stability**: Average accelerometer magnitude
3. **Touch pressure proxy**: Using accelerometer z-axis variations

### Behavioral Features
1. **Color switching frequency**: How often the child changes colors
2. **Zone preference**: Time spent in each zone
3. **Completion pattern**: Order of zones colored
4. **Revisit frequency**: How often child returns to already colored areas

### Statistical Aggregates
1. **Mean, std, min, max** of velocities, accelerations
2. **Percentiles** of stroke lengths and durations
3. **Entropy** of zone sequences
4. **Autocorrelation** of motion patterns

## 6. Data Quality Observations

1. **Multi-touch Events**: Some sessions show multiple finger IDs, indicating multi-touch usage
2. **Incomplete Sessions**: Completion percentage doesn't always reach 100%
3. **Touch Phases**: "Canceled" phases might indicate interruptions or errors
4. **Coordinate System**: Appears to be standard screen coordinates with (0,0) at top-left

## 7. Next Steps

1. **Feature Extraction Pipeline**: Implement the recommended features
2. **Data Cleaning**: Handle multi-touch events and incomplete sessions
3. **Normalization**: Account for different screen sizes/resolutions
4. **Temporal Alignment**: Standardize time series for comparison across children
5. **Label Integration**: Merge with any available labels for classification
"""

    # Save report
    with open(REPORT_PATH / 'analysis_report.md', 'w') as f:
        f.write(report)
    
    # Also save as text
    with open(REPORT_PATH / 'analysis_report.txt', 'w') as f:
        f.write(report)
    
    return report

def main():
    """Main analysis function"""
    print("Starting Kidaura Coloring Data Analysis...")
    
    # 1. Get file statistics
    print("\n1. Collecting file statistics...")
    stats = get_file_statistics()
    print(f"   Found {stats['total_children']} children with {stats['total_coloring_files']} coloring files")
    
    # 2. Analyze sample sessions
    print("\n2. Analyzing sample sessions...")
    session_data = analyze_sample_files(sample_size=10)
    print(f"   Analyzed {len(session_data)} sessions")
    
    # 3. Extract detailed point features
    print("\n3. Extracting point-level features...")
    points_df = extract_point_features()
    print(f"   Extracted {len(points_df)} touch points")
    
    # 4. Create visualizations
    print("\n4. Creating visualizations...")
    create_visualizations(stats, session_data, points_df)
    print(f"   Saved visualizations to {REPORT_PATH}")
    
    # 5. Generate report
    print("\n5. Generating analysis report...")
    report = generate_report(stats, session_data, points_df)
    print(f"   Report saved to {REPORT_PATH}/analysis_report.md")
    
    # Save sample data for inspection
    if not points_df.empty:
        points_df.head(1000).to_csv(REPORT_PATH / 'sample_touch_points.csv', index=False)
        print(f"   Sample data saved to {REPORT_PATH}/sample_touch_points.csv")
    
    print("\nAnalysis complete!")
    return stats, session_data, points_df

if __name__ == "__main__":
    stats, session_data, points_df = main()
