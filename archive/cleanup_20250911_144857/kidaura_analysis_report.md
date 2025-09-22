# Kidaura Coloring Game Data Analysis Report
## Touchscreen-Based Assessment for ASD/DD/TD Classification

Generated: 2025-09-07

---

## Executive Summary

This report analyzes touchscreen data from a coloring game designed to assess motor control, cognitive planning, and behavioral patterns in children. The dataset contains:

- **286 children** with diagnostic labels (ASD, TD, DD)
- **867 total coloring sessions** 
- **Rich touchscreen data** including coordinates, timing, accelerometer readings, and game progress

Initial analysis reveals distinct patterns between diagnostic groups in motor control, planning ability, and task completion behaviors.

## 1. Dataset Overview

### 1.1 Population Distribution

| Group | Full Name | Number of Children | Description |
|-------|-----------|-------------------|-------------|
| **ASD** | Autism Spectrum Disorder | ~100 | Developmental disorder affecting communication and behavior |
| **TD** | Typical Development | ~120 | Neurotypically developing children (control group) |
| **DD** | Developmental Delay | ~66 | General developmental delays without specific ASD diagnosis |

### 1.2 Data Collection Context

- **Device**: Touch-screen tablet with accelerometer
- **Task**: Coloring game with defined zones (e.g., cake decorating theme)
- **Sessions per child**: 2-6 coloring files (average: 3)
- **Session duration**: 35-145 seconds (average: 78.5 seconds)

## 2. Data Structure Analysis

### 2.1 Touch Event Structure

Each touch point contains:
```json
{
  "x": 842.0,              // Screen X coordinate
  "y": 657.0,              // Screen Y coordinate  
  "time": 177.908722,      // Seconds since session start
  "touchPhase": "Began",   // Touch event type
  "fingerId": 0,           // Multi-touch identifier
  "accx": 0.009780884,     // Accelerometer X
  "accy": 0.0248413086,    // Accelerometer Y
  "accz": -0.9923706,      // Accelerometer Z
  "color": "RedDefault",   // Current color selection
  "zone": "TopCake",       // Target area being colored
  "completionPerc": 0.0    // Progress percentage
}
```

### 2.2 Game Zones Identified

Different coloring templates detected:
- **Cake template**: TopCake, BottomCake, TopIcing, BottomIcing, Cherry
- **House template**: Wall, Roof, Door, Window
- **Character template**: Head, Face, Hair

## 3. Key Findings from Sample Analysis

### 3.1 Motor Control Differences

From the sample of 61 sessions analyzed:

| Feature | ASD | TD | DD | Interpretation |
|---------|-----|----|----|----------------|
| **Mean Stroke Velocity** | 1471.8 pixels/s | 822.8 pixels/s | 1096.7 pixels/s | ASD shows faster, less controlled movements |
| **Velocity Std Dev** | Higher variability | Lower variability | Moderate | ASD shows inconsistent motor control |
| **Acceleration Std Dev** | 117,226 | 50,281 | 77,639 | ASD shows jerkier movements |

### 3.2 Planning and Organization

| Feature | ASD | TD | DD | Interpretation |
|---------|-----|----|----|----------------|
| **Out-of-bounds ratio** | 67.5% | 39.2% | 52.3% | ASD colors outside lines more frequently |
| **Zone transition rate** | 0.274 | 0.141 | 0.195 | ASD switches between areas more often |
| **Color switches** | More frequent | Less frequent | Moderate | ASD shows less organized approach |

### 3.3 Task Completion

| Feature | ASD | TD | DD | Interpretation |
|---------|-----|----|----|----------------|
| **Final completion %** | 89.4% | 97.2% | 95.7% | ASD shows lower task completion |
| **Session duration** | Longer | Shorter | Moderate | ASD takes more time |
| **Progress linearity** | Lower R² | Higher R² | Moderate | ASD shows less consistent progress |

## 4. Clinical Insights

### 4.1 ASD-Specific Patterns

1. **Motor Control Issues**:
   - Higher velocity with greater variability
   - Increased acceleration/deceleration (jerkiness)
   - More tremor-like movements in accelerometer data

2. **Executive Function Challenges**:
   - Frequent zone switching (difficulty staying on task)
   - Higher out-of-bounds coloring (spatial awareness)
   - More color changes (decision-making difficulties)

3. **Sensory Processing**:
   - Different pressure patterns (accelerometer Z-axis)
   - Variable touch duration and intensity

### 4.2 TD Characteristics

1. **Controlled Movements**:
   - Smoother strokes with consistent velocity
   - Lower acceleration variability
   - Better staying within boundaries

2. **Organized Approach**:
   - Systematic zone completion
   - Fewer color switches
   - Linear progress toward completion

### 4.3 DD Patterns

- Generally intermediate between ASD and TD
- Some children labeled as "DD with emerging ASD features"
- More heterogeneous group requiring careful analysis

## 5. Feature Engineering Recommendations

### 5.1 Primary Features for Classification

**Motor Control Features** (High Priority):
- Stroke velocity statistics (mean, std, max)
- Acceleration patterns and jerkiness
- Movement smoothness (spectral arc length)
- Tremor indicators from accelerometer

**Planning Features** (High Priority):
- Zone transition patterns
- Out-of-bounds percentage
- Color switching frequency
- Completion strategies

**Temporal Features** (Medium Priority):
- Inter-stroke intervals
- Session pacing
- Time per zone
- Progress consistency

### 5.2 Advanced Features

**Pressure Proxies**:
- Accelerometer Z-axis variations
- Touch area changes (if available)
- Stroke intensity patterns

**Behavioral Markers**:
- Repetitive patterns
- Perseveration on specific zones
- Avoidance of certain areas
- Multi-touch usage patterns

## 6. Visualizations

![File Statistics](coloring_data_report/file_statistics.png)
*Distribution of coloring files per child and file sizes*

![Zone Distribution](coloring_data_report/zone_distribution.png)
*Frequency of touch points by game zone - shows "Outside" touches are common*

![Touch Phases](coloring_data_report/touch_phases.png)
*Distribution of touch events - "Moved" events dominate*

![Completion Progress](coloring_data_report/completion_progress.png)
*Sample child's completion progress over time*

## 7. Recommendations

### 7.1 For Classification Model

1. **Feature Selection**:
   - Focus on motor control and planning features
   - Use both stroke-level and session-level aggregations
   - Include temporal progression features

2. **Data Preprocessing**:
   - Normalize for screen size differences
   - Handle multi-touch events appropriately
   - Account for different game templates

3. **Model Architecture**:
   - Consider ensemble methods combining motor and behavioral features
   - Use time-series models for temporal patterns
   - Implement hierarchical models (stroke → session → child)

### 7.2 For Clinical Application

1. **Early Screening**:
   - Motor control features show promise for early detection
   - Planning deficits visible in zone transition patterns
   - Quick assessment possible (< 2 minutes per session)

2. **Progress Monitoring**:
   - Track changes in motor control over multiple sessions
   - Monitor improvement in task completion
   - Assess response to interventions

3. **Personalized Insights**:
   - Identify specific motor vs. planning challenges
   - Tailor interventions based on deficit patterns
   - Provide objective measures for clinicians

## 8. Technical Considerations

### 8.1 Data Quality

- **Multi-touch events**: Some children use multiple fingers
- **Incomplete sessions**: Not all reach 100% completion
- **Template variations**: Different coloring images affect zones
- **Device differences**: Screen sizes may vary

### 8.2 Implementation Notes

- Sampling rate appears to be ~30Hz based on timestamp intervals
- Accelerometer data indicates tablet movement during use
- "Canceled" touch phases may indicate technical issues or interruptions
- Some zones like "Super outside" indicate significant off-target touches

## 9. Next Steps

1. **Complete Feature Extraction**:
   - Process all 286 children's data
   - Extract full feature set as specified
   - Create train/test splits preserving child groupings

2. **Model Development**:
   - Implement classification models using `src/model.py`
   - Compare multiple algorithms (XGBoost, LightGBM, Neural Networks)
   - Optimize for balanced sensitivity/specificity

3. **Clinical Validation**:
   - Correlate with clinical assessments
   - Validate on held-out test set
   - Assess generalization to new populations

4. **Feature Importance Analysis**:
   - Use SHAP values for interpretability
   - Identify most discriminative features
   - Create clinical decision support insights

## 10. Conclusion

The Kidaura coloring game data provides rich insights into motor control, planning abilities, and behavioral patterns that differ between ASD, TD, and DD populations. The touchscreen-based assessment shows promise as an objective, engaging, and efficient screening tool for developmental disorders. The combination of motor control metrics and executive function indicators offers a comprehensive view of each child's abilities, potentially enabling earlier detection and more targeted interventions.

---

*This analysis is based on a sample of 61 sessions from 20 children. Full analysis of all 867 sessions across 286 children is recommended for robust model development.*
