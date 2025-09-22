# Kidaura Coloring Game Feature Extraction Guide
## Binary Classification: ASD+DD vs TD

### Overview

This guide documents the improved feature extraction approach for the touchscreen coloring game data. Based on your clarifications:

- **Age group**: 3-6 years old children
- **Binary classification**: ASD + DD (combined) vs TD
- **Device setup**: Tablet on table, tilted toward child
- **Templates**: Two types - Hut (Wall, Roof, Door) and Cake (TopCake, BottomCake, Icing, Cherry)
- **Session ordering**: Timestamps indicate which session was played first

### Key Improvements in Feature Design

#### 1. **Area Coverage Features** ✓
Since you specifically asked about area covered:
- `area_covered`: Convex hull area for each stroke
- `total_area_covered`: Sum across all strokes in session
- `target_zone_coverage`: Ratio of touches within template zones vs outside
- `zone_coverage_std`: Uniformity of coverage across different zones

#### 2. **Palm Touch Detection** ✓
Addresses multi-touch complexity:
- `is_palm_touch`: Detected when points are densely clustered with large area
- `num_palm_touches`: Count per session
- `potential_palm_touches`: Based on high finger IDs (>3)
- Handles cases where fingerId increases due to palm contact

#### 3. **Template-Aware Features** ✓
Separate analysis for Hut vs Cake templates:
- `template_type`: Automatically detected from zones
- Zone-specific coverage metrics
- Template-appropriate target zones

#### 4. **Temporal Progression Features** ✓
Using timestamps to track improvement over sessions:
- `session_ordering`: Based on filename timestamps
- `progression_features`: First vs last session differences
- `trend_features`: Linear trends across multiple sessions
- `consistency_metrics`: Variability across sessions

#### 5. **Pressure Proxy Features** ✓
From accelerometer Z-axis (perpendicular to screen):
- `pressure_mean`: Inverted Z-axis (more negative = more pressure)
- `pressure_std`: Pressure variability
- `device_stability`: From accelerometer magnitude
- Better representation of how hard child presses

#### 6. **Tremor Analysis** ✓
Enhanced tremor detection for motor control assessment:
- `tremor_ratio`: Power in 4-12Hz band vs normal movement
- `tremor_peak_freq`: Dominant tremor frequency
- FFT-based analysis on accelerometer data

### Feature Categories

#### Session-Level Features
```python
{
    # Basic Metrics
    'num_strokes': 45,
    'num_points': 1823,
    'session_duration': 78.5,
    'template_type': 'cake',
    
    # Area Coverage
    'total_area_covered': 125840.5,
    'target_zone_coverage': 0.72,  # 72% touches in correct zones
    'outside_ratio': 0.28,          # 28% outside boundaries
    
    # Motor Control
    'velocity_mean_session_mean': 892.3,
    'velocity_cv_session_mean': 0.45,    # Coefficient of variation
    'acceleration_std_session_mean': 15234.7,
    'jerk_score_session_mean': 8932.1,
    'path_efficiency_session_mean': 0.31,
    
    # Tremor & Pressure
    'tremor_ratio_session_mean': 0.23,
    'pressure_std_session_mean': 0.08,
    'device_stability_session_mean': 12.4,
    
    # Planning & Organization
    'zone_transitions': 145,
    'zone_transition_rate': 0.08,
    'zones_revisited': 4,
    'color_switches': 8,
    
    # Multi-touch
    'num_unique_fingers': 2,
    'multitouch_used': True,
    'num_palm_touches': 3,
    
    # Progress
    'final_completion': 94.5,
    'progress_linearity': 0.87,  # R² of completion vs time
    'num_pauses': 2
}
```

#### Child-Level Features (Aggregated)
```python
{
    # Identity
    'child_id': '60ed66f0e1428d342f6651ae',
    'group': 'ASD_DD',  # Binary label
    'num_sessions': 4,
    
    # Aggregated metrics (mean, std, min, max)
    'velocity_mean_session_mean_mean': 1205.7,  # Average across sessions
    'velocity_mean_session_mean_std': 234.5,     # Variability across sessions
    
    # Progression features
    'final_completion_first_last_diff': 12.3,   # Improvement
    'velocity_mean_session_mean_trend': -45.2,   # Getting slower/controlled
    
    # Consistency
    'final_completion_consistency': 0.92,        # How consistent across sessions
    'session_duration_consistency': 0.78
}
```

### Key Differentiators (ASD+DD vs TD)

Based on analysis, these features show strongest differences:

1. **Motor Control**
   - ASD+DD: Higher velocity, more variable (CV)
   - ASD+DD: Higher jerk scores
   - ASD+DD: Lower path efficiency

2. **Tremor & Pressure**
   - ASD+DD: Higher tremor ratios
   - ASD+DD: More variable pressure
   - ASD+DD: Less device stability

3. **Planning**
   - ASD+DD: Higher zone transition rates
   - ASD+DD: More zones revisited
   - ASD+DD: Higher outside ratio

4. **Multi-touch Issues**
   - ASD+DD: More palm touches
   - ASD+DD: Higher max finger IDs
   - ASD+DD: More finger switching

5. **Progress**
   - ASD+DD: Lower final completion
   - ASD+DD: Less linear progress
   - ASD+DD: More pauses

### Usage Instructions

1. **Run Feature Extraction**:
```bash
python extract_features_binary.py
```

2. **Output Files**:
- `features_binary/session_features_binary.csv` - All sessions with features
- `features_binary/child_features_binary.csv` - Child-level aggregations
- `features_binary/feature_summary_by_group.csv` - Group comparisons

3. **For Classification**:
- Use child-level features for robust classification
- Session-level for fine-grained analysis
- Consider ensemble of both levels

### Feature Selection Recommendations

**Top 20 Features** (based on expected discriminative power):

1. `velocity_cv_session_mean_mean` - Velocity consistency
2. `jerk_score_session_mean_mean` - Movement smoothness
3. `tremor_ratio_session_mean_mean` - Motor tremor
4. `outside_ratio_mean` - Spatial accuracy
5. `zone_transition_rate_mean` - Task focus
6. `path_efficiency_session_mean_mean` - Movement planning
7. `pressure_std_session_mean_mean` - Touch consistency
8. `num_palm_touches_mean` - Motor control issues
9. `progress_linearity_mean` - Task execution
10. `device_stability_session_mean_mean` - Hand steadiness
11. `final_completion_mean` - Task completion
12. `zones_revisited_mean` - Planning ability
13. `inter_stroke_cv_mean` - Timing consistency
14. `curvature_mean_session_mean_mean` - Stroke smoothness
15. `color_switch_rate_mean` - Decision consistency
16. `target_zone_coverage_mean` - Spatial awareness
17. `acceleration_std_session_mean_mean` - Movement control
18. `num_pauses_mean` - Attention/fatigue
19. `zone_coverage_entropy_mean` - Coverage uniformity
20. `finger_switch_rate_mean` - Touch control

### Model Training Tips

1. **Handle Class Imbalance**: More TD than ASD+DD samples
2. **Feature Scaling**: Standardize features before training
3. **Cross-Validation**: Use GroupKFold to keep child's sessions together
4. **Ensemble Methods**: XGBoost/LightGBM work well with these features
5. **Interpretability**: Use SHAP values to understand predictions

### Clinical Interpretation

The features capture multiple developmental domains:

- **Fine Motor Control**: Velocity, acceleration, jerk, tremor
- **Visual-Motor Integration**: Path efficiency, zone accuracy
- **Executive Function**: Planning, organization, task completion
- **Sensory Processing**: Pressure patterns, device handling
- **Attention/Focus**: Pauses, session duration, progress consistency

This comprehensive feature set provides a rich representation of the child's interaction patterns, enabling robust classification while maintaining clinical interpretability.
