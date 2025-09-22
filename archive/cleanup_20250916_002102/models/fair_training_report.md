# Fair ASD Detection Model Training Report

Generated: 2025-09-07 13:33:51

## Key Training Considerations
- All sessions from same child kept together (no data leakage)
- Grouped cross-validation respecting child boundaries
- Child-level aggregated features used for training
- Session-level validation performed

## Model Performance (Child-Level)

### RandomForest
- Sensitivity: 0.812
- Specificity: 0.727
- Balanced Accuracy: 0.770
- AUC-ROC: 0.827
- PPV: 0.812
- NPV: 0.727
- MCC: 0.540

Confusion Matrix:
  - True Positives: 26
  - True Negatives: 16
  - False Positives: 6
  - False Negatives: 6

### XGBoost
- Sensitivity: 0.875
- Specificity: 0.636
- Balanced Accuracy: 0.756
- AUC-ROC: 0.795
- PPV: 0.778
- NPV: 0.778
- MCC: 0.533

Confusion Matrix:
  - True Positives: 28
  - True Negatives: 14
  - False Positives: 8
  - False Negatives: 4

### LightGBM
- Sensitivity: 0.781
- Specificity: 0.591
- Balanced Accuracy: 0.686
- AUC-ROC: 0.783
- PPV: 0.735
- NPV: 0.650
- MCC: 0.379

Confusion Matrix:
  - True Positives: 25
  - True Negatives: 13
  - False Positives: 9
  - False Negatives: 7

## Best Model: RandomForest
- Correctly identifies 81% of ASD/DD children
- Correctly identifies 73% of TD children
- Overall balanced accuracy: 77.0%

## Feature Importance (Top 15)