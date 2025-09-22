# High-Accuracy ASD Detection Model Report

Generated: 2025-09-07 13:29:21

## Model Performance Summary

| Model | Sensitivity | Specificity | PPV | NPV | Balanced Acc | AUC-ROC | Custom Score |
|-------|------------|-------------|-----|-----|--------------|---------|--------------|
| ExtraTrees_unbalanced | 0.818 | 0.864 | 0.900 | 0.760 | 0.841 | 0.876 | 0.836 |
| Stacking_unbalanced | 0.848 | 0.818 | 0.875 | 0.783 | 0.833 | 0.877 | 0.830 |
| XGBoost_balanced | 0.879 | 0.773 | 0.853 | 0.810 | 0.826 | 0.888 | 0.815 |
| LightGBM_balanced | 0.879 | 0.773 | 0.853 | 0.810 | 0.826 | 0.886 | 0.815 |
| RandomForest_unbalanced | 0.879 | 0.773 | 0.853 | 0.810 | 0.826 | 0.873 | 0.815 |
| RandomForest_balanced | 0.939 | 0.727 | 0.838 | 0.889 | 0.833 | 0.868 | 0.812 |
| XGBoost_unbalanced | 0.909 | 0.727 | 0.833 | 0.842 | 0.818 | 0.873 | 0.800 |
| Voting_unbalanced | 0.788 | 0.818 | 0.867 | 0.720 | 0.803 | 0.879 | 0.800 |
| Stacking_balanced | 0.879 | 0.727 | 0.829 | 0.800 | 0.803 | 0.882 | 0.788 |
| Voting_balanced | 0.697 | 0.909 | 0.920 | 0.667 | 0.803 | 0.895 | 0.782 |

## Best Model: ExtraTrees_unbalanced

### Performance Metrics:
- **Sensitivity**: 0.818 (81.8% of ASD cases detected)
- **Specificity**: 0.864 (86.4% of TD cases correctly identified)
- **PPV**: 0.900 (When positive, 90.0% actually have ASD)
- **NPV**: 0.760 (When negative, 76.0% actually are TD)
- **Optimal Threshold**: 0.440

### Confusion Matrix:
- True Positives (ASD correctly identified): 27
- True Negatives (TD correctly identified): 19
- False Positives (TD misclassified as ASD): 3
- False Negatives (ASD missed): 6

## Clinical Interpretation
- The model correctly identifies 82 out of 100 children with ASD/DD
- The model correctly identifies 86 out of 100 TD children
- Very high balanced performance: 84.1%