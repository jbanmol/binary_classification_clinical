# ASD/TD Model — Complete Technical Guide

This comprehensive guide explains what the model is, how it was trained, and exactly how to use it.

## 1) What the model does
- **Task**: Binary classification — ASD (including DD) vs TD at the child level
- **Clinical requirement**: meet both Sensitivity ≥ 0.86 and Specificity ≥ 0.70
- **Final performance** (fixed test split):
  - Sensitivity ≈ 0.882, Specificity ≈ 0.800, AUC ≈ 0.906 (bagged across seeds)

## 2) What's inside the model bundle
Saved under `models/final_np_iqrmid_u16n50_k2/`:
- **preprocess/**
  - `scaler_all.joblib` — StandardScaler fit on training set
  - `umap_all.joblib` — UMAP (cosine) embedding with n_components=16, n_neighbors=50
- **models/**
  - Per‑model estimators used by the ensemble (LightGBM, XGBoost, BRF, ExtraTrees)
  - `meta_combiner.joblib` (only if meta used; otherwise alpha blend is recorded)
- **bundle.json** — manifest containing:
  - `feature_columns` — order of input columns used during training
  - `feature_config` — representation settings (UMAP), etc.
  - `models_used` and their file paths
  - `ensembles` — e_sens and e_spec model lists
  - `combiner` — either {type: 'alpha', best_alpha: ...} or {type: 'meta', path: ...}; includes temperature T if used
  - `threshold` — final decision threshold τ to convert probability to ASD/TD
  - `holdout_metrics` and `cv_summary` for traceability

## 3) Complete Model Training Process

### 3.1 Data Ingestion and Preprocessing

#### Raw Data Sources
- **Input**: Raw behavioral data from multiple sessions per child
- **Data Engine**: Uses `rag_system.research_engine` to ingest and index behavioral data
- **Filtering**: Only includes children with known labels (ASD or TD)

#### Child-Level Aggregation
The model works at the **child level**, not session level, to prevent data leakage:

```python
# For each child, compute mean values across all their sessions
agg = df.groupby('child_id')[numeric_features].mean().reset_index()
sess_count = df.groupby('child_id').size().rename('session_count').reset_index()
```

#### Feature Engineering (Domain-Specific)
**Core Numeric Features** (23 features):
- `velocity_mean`, `velocity_std`, `velocity_max`, `velocity_cv`
- `tremor_indicator`, `acc_magnitude_mean`, `acc_magnitude_std`
- `palm_touch_ratio`, `unique_fingers`, `max_finger_id`
- `session_duration`, `stroke_count`, `total_touch_points`
- `unique_zones`, `unique_colors`, `final_completion`
- `completion_progress_rate`, `avg_time_between_points`, `canceled_touches`

**Engineered Domain Features** (12+ features):
- **Per-zone dynamics**: `touches_per_zone`, `strokes_per_zone`, `zones_per_minute`
- **Jitter/jerk proxies**: `vel_std_over_mean`, `acc_std_over_mean`
- **Temporal stability**: `avg_ibp_norm`, `interpoint_rate`, `touch_rate`, `stroke_rate`

**Quantile Binning** (24+ features):
- Creates quartile-based one-hot indicators for key ratios
- Applied to: `touch_rate`, `strokes_per_zone`, `vel_std_over_mean`, `acc_std_over_mean`, `zones_per_minute`, `interpoint_rate`

### 3.2 Data Splitting Strategy

#### Holdout Split (Child-Level)
```python
# 80% train, 20% test split by child (not by session)
GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=777)
```

#### Cross-Validation (Leakage-Safe)
```python
# 5-fold stratified group K-fold on training data
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

### 3.3 Feature Preprocessing Pipeline

#### Step 1: Standardization
```python
# Fit StandardScaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Step 2: Representation Learning (UMAP)
```python
# UMAP with cosine metric for non-linear dimensionality reduction
umap = UMAP(
    n_components=16,
    n_neighbors=50,
    metric='cosine',
    random_state=42
)
# Fit only on training data, transform both train and test
U_train = umap.fit_transform(X_train_scaled)
U_test = umap.transform(X_test_scaled)
# Concatenate original + UMAP features
X_final = np.concatenate([X_train_scaled, U_train], axis=1)
```

**Why UMAP over PCA?**
- UMAP preserves both local and global structure
- Cosine metric captures similarity patterns in behavioral data
- Non-linear relationships better captured than linear PCA

### 3.4 Model Training and Selection

#### Individual Models Trained
1. **LightGBM** (Gradient Boosting):
   ```python
   LGBMClassifier(
       n_estimators=400, learning_rate=0.05,
       subsample=0.8, colsample_bytree=0.8,
       feature_fraction=0.8, min_child_samples=30,
       max_depth=-1, random_state=seed
   )
   ```

2. **XGBoost** (Gradient Boosting):
   ```python
   XGBClassifier(
       n_estimators=400, learning_rate=0.05,
       max_depth=4, min_child_weight=3,
       subsample=0.8, colsample_bytree=0.8,
       reg_lambda=1.0, tree_method='hist'
   )
   ```

3. **Balanced Random Forest**:
   ```python
   BalancedRandomForestClassifier(
       n_estimators=200, max_depth=10,
       min_samples_split=5, min_samples_leaf=2,
       random_state=seed
   )
   ```

4. **Extra Trees**:
   ```python
   ExtraTreesClassifier(
       n_estimators=200, max_depth=10,
       min_samples_split=5, min_samples_leaf=2,
       random_state=seed
   )
   ```

#### Model Selection Process
1. **Per-fold training**: Each model trained on 4 folds, validated on 1 fold
2. **Out-of-fold (OOF) scoring**: Each model gets OOF AUC score
3. **Top-K selection**: Select top 2 models by OOF AUC for ensemble

### 3.5 Probability Calibration

#### Per-Model Calibration
```python
# Isotonic regression calibration on each model
CalibratedClassifierCV(model, method='isotonic', cv=3)
```

#### Ensemble Calibration (Temperature Scaling)
```python
# Temperature scaling on final ensemble probabilities
T = optimal_temperature  # Found via optimization
P_calibrated = 1 / (1 + exp(-log(P/(1-P)) / T))
```

### 3.6 Ensemble Strategy

#### Dual Ensemble Approach
1. **Sensitivity-Oriented Ensemble**: Models optimized for high sensitivity
2. **Specificity-Oriented Ensemble**: Models optimized for high specificity

#### Alpha Blending
```python
# Weighted combination of the two ensembles
P_final = alpha * P_sens + (1 - alpha) * P_spec
# Alpha optimized to meet both clinical targets
```

#### Meta-Learning (Optional)
```python
# Logistic regression meta-learner
meta_features = [P_sens, P_spec, |P_sens - P_spec|]
meta_learner = LogisticRegression()
P_final = meta_learner.predict_proba(meta_features)[:, 1]
```

### 3.7 Threshold Optimization

#### Neyman-Pearson Policy
```python
# Maximize True Positive Rate (sensitivity) 
# while keeping False Positive Rate ≤ 0.30
threshold = find_threshold_meeting_constraints(
    fpr_limit=0.30,
    target_sensitivity=0.86,
    target_specificity=0.70
)
```

#### Threshold Transfer Strategy
- **IQR-Mid**: Use median threshold across CV folds
- **Quantile Mapping**: Map CV threshold distribution to holdout
- **KS Guard**: Only apply mapping if distributions are similar

### 3.8 Bagging for Reliability

#### Multi-Seed Training
```python
# Train on same data split with different random seeds
seeds = [17, 29, 43]
holdout_seed = 777  # Fixed holdout split

for seed in seeds:
    # Train complete pipeline with this seed
    results[seed] = train_pipeline(seed, holdout_seed)
```

#### Final Prediction
```python
# Average probabilities across all seeds
P_bagged = np.mean([results[s]['probabilities'] for s in seeds], axis=0)
```

### 3.9 Complete Training Pipeline Summary

1. **Data Ingestion**: Raw behavioral data → child-level aggregation
2. **Feature Engineering**: 23 core + 12+ domain + 24+ quantile features
3. **Preprocessing**: StandardScaler + UMAP (16 components, cosine metric)
4. **Model Training**: 4 model types × 5-fold CV with calibration
5. **Ensemble Building**: Dual ensemble (sens/spec) + alpha blending
6. **Threshold Optimization**: Neyman-Pearson policy with IQR-mid transfer
7. **Bagging**: 3 seeds × same holdout = final reliable predictions

## 4) Raw Data Processing and Feature Extraction

### 4.1 Raw Data Ingestion Process
**Data Source**: Raw Coloring session JSON files organized by child in `data/raw/` folders
**Processing Engine**: `rag_system/research_engine.py` with `ColoringDataProcessor` class

#### Raw Data Structure
- **File Organization**: Each child has a folder containing multiple session JSON files
- **File Types**: `Coloring_*.json` (main data) and `Tracing_*.json` (auxiliary)
- **Data Format**: JSON files contain touch interaction data with timestamps, coordinates, pressure, and finger IDs

#### JSON Data Content
Each Coloring session JSON contains:
- **Touch Data**: Array of touch points with x/y coordinates, timestamps, pressure values
- **Stroke Information**: Grouped touch points representing drawing strokes
- **Metadata**: Session timestamps, child identification, file paths

### 4.2 Session-Level Feature Extraction
**Process**: Each JSON file is processed to extract comprehensive behavioral features

#### Movement Analysis Features (4 features)
- **Velocity Statistics**: Mean, standard deviation, maximum, and coefficient of variation of drawing velocities
- **Purpose**: Captures movement smoothness and consistency patterns that may indicate motor control differences

#### Touch Pattern Features (3 features)
- **Palm Touch Ratio**: Proportion of touches using palm vs. fingers
- **Unique Fingers**: Count of different fingers used during the session
- **Max Finger ID**: Highest finger identifier used
- **Purpose**: Analyzes fine motor control and finger dexterity patterns

#### Session Statistics (8 features)
- **Duration Metrics**: Total session time and average time between touch points
- **Interaction Counts**: Number of strokes, total touch points, unique zones touched
- **Completion Metrics**: Final completion percentage and progress rate
- **Color Usage**: Number of unique colors used during coloring
- **Purpose**: Captures overall engagement, attention span, and task completion patterns

#### Tremor and Acceleration Features (3 features)
- **Tremor Detection**: Algorithmic detection of tremor-like movement patterns
- **Acceleration Analysis**: Mean and standard deviation of acceleration magnitudes
- **Purpose**: Identifies motor control issues and movement stability patterns

### 4.3 Child-Level Aggregation (Critical for Leakage Prevention)
**Why Child-Level?**: Prevents data leakage by ensuring no child appears in both training and test sets

#### Aggregation Process
- **Mean Calculation**: Average all session-level features across all sessions per child
- **Session Count**: Track number of sessions per child for additional context
- **Label Assignment**: Use majority vote across sessions to assign child-level labels
- **Leakage Prevention**: Ensures realistic clinical scenario where we predict one label per child

#### Statistical Safety
- **Independence**: Each child is treated as an independent unit
- **Realistic Scenario**: Mirrors real-world clinical assessment where one diagnosis is made per child
- **Validation Integrity**: Maintains proper statistical validation without data contamination

### 4.4 Domain-Specific Feature Engineering

#### Per-Zone Dynamics (3 features)
- **Touches per Zone**: Average number of touches per unique zone touched
- **Strokes per Zone**: Average number of strokes per unique zone
- **Zones per Minute**: Rate of zone exploration over time
- **Purpose**: Captures spatial exploration patterns and attention distribution

#### Movement Smoothness Indicators (2 features)
- **Velocity Variability**: Standard deviation over mean velocity ratio
- **Acceleration Variability**: Standard deviation over mean acceleration ratio
- **Purpose**: Quantifies movement smoothness and motor control precision

#### Temporal Stability Patterns (4 features)
- **Normalized Inter-Point Time**: Average time between points relative to session duration
- **Inter-Point Rate**: Inverse of normalized inter-point time
- **Touch Rate**: Touches per unit time
- **Stroke Rate**: Strokes per unit time
- **Purpose**: Captures timing patterns and behavioral rhythm

#### Quantile Binning (24+ features)
- **Process**: Creates quartile-based one-hot indicators for key behavioral ratios
- **Features**: 6 ratio features × 4 quartiles = 24 binary indicators
- **Purpose**: Captures non-linear relationships and behavioral thresholds
- **Applied To**: Touch rate, strokes per zone, velocity variability, acceleration variability, zones per minute, inter-point rate

**Total Feature Count**: 23 core + 12+ domain + 24+ quantile = **59+ features per child**

## 5) Complete Model Training Process

### 5.1 Data Splitting Strategy

#### Holdout Split (Child-Level)
- **Method**: GroupShuffleSplit with 80% train, 20% test
- **Grouping**: Split by child ID to prevent session-level leakage
- **Fixed Seed**: Holdout seed = 777 for reproducibility
- **Purpose**: Creates realistic train/test split that mirrors clinical deployment

#### Cross-Validation (Leakage-Safe)
- **Method**: 5-fold StratifiedGroupKFold on training data
- **Grouping**: Each fold contains different children
- **Stratification**: Maintains class balance across folds
- **Purpose**: Provides robust model evaluation without data leakage

### 5.2 Feature Preprocessing Pipeline

#### Step 1: Standardization
- **Method**: StandardScaler (zero mean, unit variance)
- **Fitting**: Only on training data to prevent leakage
- **Purpose**: Normalizes features for optimal model performance
- **Application**: Applied to all numeric features before model training

#### Step 2: Representation Learning (UMAP)
- **Method**: UMAP with cosine metric
- **Parameters**: 16 components, 50 neighbors, cosine distance
- **Fitting**: Only on training data, then transform test data
- **Purpose**: Captures non-linear relationships and reduces dimensionality
- **Why UMAP over PCA**: Preserves both local and global structure, better for behavioral data

### 5.3 Model Training and Selection

#### Individual Models (4 Types)

**1. LightGBM (Gradient Boosting)**
- **Type**: Gradient boosting with decision trees
- **Key Parameters**: 400 estimators, 0.05 learning rate, 0.8 subsampling
- **Strengths**: Fast training, handles categorical features well, good with imbalanced data
- **Purpose**: Primary gradient boosting model for ensemble

**2. XGBoost (Gradient Boosting)**
- **Type**: Extreme gradient boosting with regularization
- **Key Parameters**: 400 estimators, 0.05 learning rate, L2 regularization
- **Strengths**: Robust performance, built-in regularization, handles missing values
- **Purpose**: Secondary gradient boosting model with different regularization

**3. Balanced Random Forest**
- **Type**: Random forest with class balancing
- **Key Parameters**: 200 trees, balanced class weights
- **Strengths**: Handles class imbalance naturally, robust to outliers
- **Purpose**: Provides tree-based ensemble with built-in imbalance handling

**4. Extra Trees (Extremely Randomized Trees)**
- **Type**: Random forest with extra randomization
- **Key Parameters**: 200 trees, random splits at each node
- **Strengths**: Fast training, reduces overfitting, good variance reduction
- **Purpose**: Provides additional tree-based diversity to ensemble

#### Model Selection Process
1. **Cross-Validation Training**: Each model trained on 4 folds, validated on 1 fold
2. **Out-of-Fold Scoring**: Each model evaluated using OOF AUC scores
3. **Top-K Selection**: Best 2 models by OOF AUC selected for ensemble
4. **Purpose**: Ensures only best-performing models contribute to final prediction

### 5.4 Probability Calibration

#### Per-Model Calibration
- **Method**: Isotonic regression calibration
- **Process**: 3-fold cross-validation within each model
- **Purpose**: Improves probability quality and reliability
- **Benefit**: Makes probabilities more interpretable and trustworthy

#### Ensemble Calibration (Temperature Scaling)
- **Method**: Temperature scaling on final ensemble probabilities
- **Process**: Grid search to find optimal temperature parameter
- **Formula**: P_calibrated = 1 / (1 + exp(-log(P/(1-P)) / T))
- **Purpose**: Further improves probability calibration at ensemble level

### 5.5 Ensemble Strategy

#### Dual Ensemble Approach
**Sensitivity-Oriented Ensemble (E_sens)**:
- **Goal**: Maximize sensitivity (catch ASD cases)
- **Models**: Typically LightGBM, XGBoost, Balanced Random Forest
- **Purpose**: Ensures we don't miss children who need diagnosis

**Specificity-Oriented Ensemble (E_spec)**:
- **Goal**: Maximize specificity (avoid false positives)
- **Models**: Typically LightGBM, XGBoost, Extra Trees
- **Purpose**: Reduces unnecessary referrals and anxiety

#### Alpha Blending
- **Method**: Weighted combination of sensitivity and specificity ensembles
- **Formula**: P_final = α × P_sens + (1-α) × P_spec
- **Optimization**: Grid search over α values to meet clinical targets
- **Purpose**: Balances sensitivity and specificity based on clinical requirements

#### Meta-Learning (Optional)
- **Method**: Logistic regression meta-learner
- **Features**: E_sens, E_spec, and their disagreement measure
- **Purpose**: Learns optimal combination strategy from data
- **Benefit**: Can adapt combination strategy based on prediction confidence

### 5.6 Threshold Optimization

#### Neyman-Pearson Policy
- **Goal**: Maximize sensitivity while keeping false positive rate ≤ 0.30
- **Process**: Find threshold that maximizes true positive rate under FPR constraint
- **Fallback**: If no feasible threshold, use specificity-first approach
- **Purpose**: Prioritizes catching ASD cases while controlling false alarms

#### Threshold Transfer Strategy
**IQR-Mid Method**:
- **Process**: Use median threshold across CV folds
- **Purpose**: Robust transfer that reduces fold-to-fold variance
- **Benefit**: More stable than using individual fold thresholds

**Quantile Mapping**:
- **Process**: Map CV threshold distribution to holdout using quantiles
- **Purpose**: Accounts for distribution differences between CV and holdout
- **Benefit**: More sophisticated than simple median transfer

### 5.7 Bagging for Reliability

#### Multi-Seed Training
- **Seeds**: [17, 29, 43] for different random initializations
- **Holdout**: Fixed holdout seed = 777 across all runs
- **Process**: Train complete pipeline with each seed independently
- **Purpose**: Reduces variance and improves reliability

#### Final Prediction
- **Method**: Average probabilities across all seeds
- **Formula**: P_bagged = mean([P_seed1, P_seed2, P_seed3])
- **Purpose**: Provides more stable and reliable predictions
- **Benefit**: Reduces impact of random initialization on final results

## 6) How to Run Predictions (CLI)

### Prerequisites
- **Python Version**: 3.10 or higher
- **Dependencies**: Install using `pip install -r requirements.txt`
- **Model Bundle**: Ensure `models/final_np_iqrmid_u16n50_k2/` directory exists

### Input Format Requirements
- **Data Format**: CSV file with one row per child
- **Column Order**: Must match `bundle.json` → `feature_columns` exactly
- **Feature Count**: 59+ features per child (23 core + 12+ domain + 24+ quantile)
- **Data Quality**: No missing values, numeric data only

### Prediction Process Overview
1. **Feature Loading**: Load and align input features to model schema
2. **Preprocessing**: Apply StandardScaler and UMAP transformations
3. **Model Inference**: Run through all 4 trained models
4. **Ensemble Building**: Combine sensitivity and specificity ensembles
5. **Calibration**: Apply temperature scaling if configured
6. **Thresholding**: Convert probabilities to binary predictions
7. **Output**: Generate CSV with probabilities and predictions

### Quick Prediction Example
```bash
# 1) Ensure dependencies
./venv/bin/pip install -r requirements.txt

# 2) Score a CSV of features (one row per child)
python - <<'PY'
import json, joblib, pandas as pd, numpy as np
from pathlib import Path

B = Path('models/final_np_iqrmid_u16n50_k2/bundle.json')
P = B.parent
bundle = json.load(open(B))
cols = bundle['feature_columns']

# Load input data (replace with your file)
df = pd.read_csv('data/processed/child_level_features.csv')
X = df[cols].copy()

# Preprocess
scaler = joblib.load(P/'preprocess'/'scaler_all.joblib')
X_s = scaler.transform(X)
# Optional UMAP
try:
    umap = joblib.load(P/'preprocess'/'umap_all.joblib')
    U = umap.transform(X_s)
    X_s = np.concatenate([X_s, U], axis=1)
except Exception:
    pass

# Per‑model probabilities
import numpy as np
models_used = bundle['models_used']
probs = []
for name in models_used:
    mdl = joblib.load(P/'models'/f'{name}.joblib')
    if hasattr(mdl, 'predict_proba'):
        p = mdl.predict_proba(X_s)[:,1]
    elif hasattr(mdl, 'decision_function'):
        s = mdl.decision_function(X_s)
        p = (s - s.min())/(s.max()-s.min()+1e-8)
    else:
        p = np.full(len(X_s), 0.5)
    probs.append(p)

# Build ensemble logits
def mean_of(names):
    arr = [probs[models_used.index(n)] for n in names if n in models_used]
    return np.mean(np.vstack(arr), axis=0)

E_sens = mean_of(bundle['ensembles']['e_sens_names'])
E_spec = mean_of(bundle['ensembles']['e_spec_names'])
alpha_info = bundle['combiner']
if alpha_info['type'] == 'alpha':
    alpha = alpha_info['best_alpha']
    P_ens = alpha*E_sens + (1-alpha)*E_spec
else:
    # meta
    from numpy import abs as nabs
    Z = np.column_stack([E_sens, E_spec, np.abs(E_sens - E_spec)])
    meta = joblib.load(P/'models'/'meta_combiner.joblib')
    P_ens = meta.predict_proba(Z)[:,1]

# Temperature (if used)
if alpha_info.get('temperature_applied') and 'temperature_T' in alpha_info:
    T = float(alpha_info['temperature_T'])
    P_ens = 1/(1+np.exp(-np.log(P_ens/(1-P_ens+1e-12))/T))

# Apply final threshold τ
tau = float(bundle['threshold'])
labels = (P_ens >= tau).astype(int)

out = df.copy()
out['prob_asd'] = P_ens
out['pred_label'] = labels  # 1=ASD, 0=TD
out.to_csv('predictions.csv', index=False)
print('Saved predictions.csv with prob_asd and pred_label columns.')
PY
```

### Output Format
- **prob_asd**: Probability of ASD diagnosis (0.0 to 1.0)
- **pred_label**: Binary prediction (1 = ASD, 0 = TD)
- **Threshold**: Default threshold ≈ 0.5310 (from bundle configuration)

## 7) FAQ (Plain Language)

### Model Performance Questions

**Q: Why is cross-validation noisier than the final test?**
A: Each fold sees a different small set of kids; the final test is one fixed set. We also average multiple runs (bagging) for the final score, which is more stable.

**Q: What number should I trust?**
A: The final test metrics (or the bagged result). Those are the deployment numbers. The bagged results show Sensitivity ≈ 0.882, Specificity ≈ 0.800, AUC ≈ 0.906.

**Q: What if new data looks different?**
A: Monitor sensitivity/specificity. If probabilities shift, re-fit temperature scaling; adjust the threshold slightly only if needed.

### Technical Questions

**Q: Why child-level aggregation instead of session-level?**
A: Prevents data leakage and ensures realistic clinical scenario where we predict one label per child, not per session. This mirrors how clinicians make diagnoses.

**Q: Why UMAP instead of PCA?**
A: UMAP captures non-linear relationships in behavioral data better than linear PCA, and the cosine metric is more appropriate for similarity patterns in touch behavior.

**Q: Why so many models in the ensemble?**
A: Different models capture different patterns. The ensemble combines their strengths while reducing individual model weaknesses through averaging.

**Q: What's the difference between sensitivity and specificity ensembles?**
A: Sensitivity ensemble prioritizes catching ASD cases (avoiding false negatives), while specificity ensemble avoids false positives. The alpha blending balances both goals.

### Clinical Questions

**Q: How confident should I be in these predictions?**
A: The model meets clinical targets (Sens ≥ 0.86, Spec ≥ 0.70) on holdout data. However, these are screening tools - final diagnosis should always involve clinical assessment.

**Q: What if a child has very few sessions?**
A: The model aggregates across all available sessions. More sessions generally provide more reliable predictions, but the model can work with single sessions.

**Q: Can this model be used for other age groups?**
A: The model was trained on specific age ranges. Performance on different age groups would need validation before clinical use.

### Data Questions

**Q: What if I don't have all the features?**
A: The model requires all features in the exact order specified in the bundle. Missing features would need to be imputed or the model retrained.

**Q: How often should the model be retrained?**
A: Regular retraining is recommended as new data becomes available, especially if behavioral patterns change over time or with different populations.

**Q: Can I use this model with different coloring tasks?**
A: The model is specifically trained on the coloring task data. Different tasks would require retraining with appropriate data.

