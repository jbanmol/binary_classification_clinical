# Kidaura Coloring Game Data Analysis Report

Generated: 2025-09-07 12:27:08

## 1. Dataset Overview

### File Statistics
- **Total number of children**: 286
- **Total coloring files**: 867
- **Average files per child**: 3.03
- **File size range**: 0.9 - 1015.4 KB
- **Average file size**: 332.3 KB

### Files per Child Distribution
- Minimum: 0
- Maximum: 7
- Median: 3

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

### Coordinate Ranges
- X coordinates: 0 - 2048
- Y coordinates: 16 - 1536

### Touch Phases Distribution
touchPhase
Moved         5931
Stationary    1227
Began          299
Ended          231
Canceled       125

### Zone Distribution
zone
Outside          3242
Super outside    1219
Wall              931
Roof              566
Bound             410
TopIcing          241
BottomCake        222
Cherry            209
BottomIcing       205
Door              146

### Colors Used
color
Yellow        2684
RedDefault    2524
Orange         845
Violet         628
Pink           551
Red            295
Blue           167
Green          119

### Accelerometer Data Ranges
- accx: [-1.185852, 0.943893]
- accy: [-0.485855, 0.987900]
- accz: [-3.818359, 0.935959]

## 4. Session-Level Analysis

### Stroke Statistics
- Average strokes per session: 38.6
- Stroke count range: 10 - 103

### Session Duration
- Average session duration: 78.5 seconds
- Duration range: 34.7 - 145.1 seconds

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
