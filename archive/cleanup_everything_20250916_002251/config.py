"""
Configuration Module

Central configuration for the binary classification project.
Includes hyperparameters, paths, and model settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# =====================
# Data Configuration
# =====================
DATA_CONFIG = {
    'raw_data_dir': PROJECT_ROOT / 'data' / 'raw',
    'processed_data_dir': PROJECT_ROOT / 'data' / 'processed',
    'external_data_dir': PROJECT_ROOT / 'data' / 'external',
    
    # File names
    'train_file': 'train.csv',
    'test_file': 'test.csv',
    'validation_file': 'validation.csv',
    
    # Data processing
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'target_column': 'target',  # Update with actual target column name
}

# =====================
# Model Configuration
# =====================
MODEL_CONFIG = {
    'model_dir': PROJECT_ROOT / 'models',
    'best_model_name': 'best_model.pkl',
    'model_comparison_results': 'model_comparison.csv',
    
    # Training settings
    'cv_folds': 5,
    'scoring_metric': 'custom',  # Options: 'accuracy', 'f1', 'custom'
    'sensitivity_weight': 0.7,   # Weight for sensitivity in custom scorer
    
    # Models to train (comment out to exclude)
    'models_to_train': [
        'logistic_regression',
        'random_forest',
        'xgboost',
        'lightgbm',
        # 'decision_tree',
        # 'extra_trees',
        # 'gradient_boosting',
        # 'adaboost',
        # 'svm',
        # 'knn',
        # 'naive_bayes',
        # 'neural_network'
    ],
    
    # Hyperparameter tuning
    'tune_hyperparameters': True,
    'hyperparameter_optimization_metric': 'custom',
    
    # Model calibration
    'calibrate_probabilities': True,
    'calibration_method': 'sigmoid',  # Options: 'sigmoid', 'isotonic'
}

# =====================
# Feature Engineering Configuration
# =====================
FEATURE_CONFIG = {
    # Feature scaling
    'scaler_type': 'standard',  # Options: 'standard', 'minmax', 'robust'
    
    # Missing value imputation
    'imputer_strategy': 'median',  # Options: 'mean', 'median', 'most_frequent', 'knn'
    
    # Feature engineering
    'engineer_features': True,
    'polynomial_features': True,
    'interaction_features': True,
    'log_features': True,
    
    # Categorical encoding
    'categorical_encoding': {
        'enabled': True,
        'method': 'onehot',  # 'onehot' or 'ordinal'
        'max_onehot_categories': 20,
        'id_unique_ratio': 0.8,
        'exclude_columns': ['child_id', 'filename', 'session_id', 'id', 'uid']
    },
    
    # Representation learning
    'representation_learning': {
        'enabled': True,
        # method: 'autoencoder' (requires torch or tensorflow), 'umap', or 'pca'
        'method': 'umap',
        'n_components': 16,
        'random_state': 42,
        # Autoencoder-specific hyperparameters (used only if method == 'autoencoder')
        'ae_hidden_dims': [64, 32],
        'ae_dropout': 0.0,
        'ae_lr': 1e-3,
        'ae_batch_size': 128,
        'ae_epochs': 50,
    },
    
    # Feature selection
    'select_features': True,
    'feature_selection_method': 'mutual_info',  # Options: 'chi2', 'f_classif', 'mutual_info'
    'n_features_to_select': 20,
}

# =====================
# Optimization Configuration
# =====================
OPTIMIZATION_CONFIG = {
    # Threshold optimization
    'optimize_threshold': True,
    'threshold_optimization_metric': 'custom',  # Options: 'balanced', 'f1', 'youden', 'custom'
    'threshold_search_range': (0.1, 0.9),
    'threshold_search_step': 0.01,
    
    # Class imbalance handling
    'handle_imbalance': True,
    'imbalance_strategy': 'class_weight',  # Options: 'class_weight', 'smote', 'adasyn'
    
    # Early stopping (for iterative models)
    'early_stopping': True,
    'early_stopping_rounds': 50,
}

# =====================
# Evaluation Configuration
# =====================
EVALUATION_CONFIG = {
    'results_dir': PROJECT_ROOT / 'results',
    'plots_dir': PROJECT_ROOT / 'results' / 'plots',
    'reports_dir': PROJECT_ROOT / 'results' / 'reports',
    
    # Metrics to calculate
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'specificity',
        'f1_score',
        'balanced_accuracy',
        'mcc',
        'kappa',
        'auc_roc',
        'auc_pr'
    ],
    
    # Plotting settings
    'generate_plots': True,
    'plot_formats': ['png', 'pdf'],
    'plot_dpi': 300,
}

# =====================
# Experiment Tracking Configuration
# =====================
TRACKING_CONFIG = {
    'use_mlflow': True,
    'mlflow_tracking_uri': PROJECT_ROOT / 'mlruns',
    'experiment_name': 'binary_classification_experiment',
    
    # Logging settings
    'log_models': True,
    'log_metrics': True,
    'log_params': True,
    'log_artifacts': True,
}

# =====================
# Advanced Model Configurations
# =====================

# XGBoost specific settings
XGBOOST_CONFIG = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'auto',  # Use GPU if available
    'predictor': 'auto',
    'use_label_encoder': False,
}

# LightGBM specific settings
LIGHTGBM_CONFIG = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'is_unbalance': True,
    'verbose': -1,
}

# Neural Network specific settings
NEURAL_NETWORK_CONFIG = {
    'hidden_layer_sizes': (100, 50, 25),
    'activation': 'relu',
    'solver': 'adam',
    'learning_rate': 'adaptive',
    'max_iter': 1000,
    'early_stopping': True,
    'validation_fraction': 0.1,
}

# =====================
# Hardware Configuration
# =====================
HARDWARE_CONFIG = {
    'n_jobs': -1,  # Use all available CPU cores
    'use_gpu': True,  # Use GPU if available (for supported models)
    'gpu_device': 0,  # GPU device ID
}

# =====================
# Logging Configuration
# =====================
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': PROJECT_ROOT / 'logs' / 'binary_classification.log',
}

# =====================
# Utility Functions
# =====================
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_CONFIG['raw_data_dir'],
        DATA_CONFIG['processed_data_dir'],
        DATA_CONFIG['external_data_dir'],
        MODEL_CONFIG['model_dir'],
        EVALUATION_CONFIG['results_dir'],
        EVALUATION_CONFIG['plots_dir'],
        EVALUATION_CONFIG['reports_dir'],
        PROJECT_ROOT / 'logs',
        PROJECT_ROOT / 'notebooks',
        PROJECT_ROOT / 'tests',
        PROJECT_ROOT / 'docs'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    # Create .gitkeep files in data directories
    gitkeep_dirs = [
        DATA_CONFIG['raw_data_dir'],
        DATA_CONFIG['processed_data_dir'],
        DATA_CONFIG['external_data_dir'],
        MODEL_CONFIG['model_dir']
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_file = directory / '.gitkeep'
        if not gitkeep_file.exists():
            gitkeep_file.touch()

def get_model_path(model_name: str) -> Path:
    """Get the full path for a model file."""
    return MODEL_CONFIG['model_dir'] / f"{model_name}.pkl"

def get_data_path(data_type: str) -> Path:
    """
    Get the full path for a data file.
    
    Args:
        data_type: 'train', 'test', 'validation', 'raw', 'processed'
    """
    if data_type == 'train':
        return DATA_CONFIG['processed_data_dir'] / DATA_CONFIG['train_file']
    elif data_type == 'test':
        return DATA_CONFIG['processed_data_dir'] / DATA_CONFIG['test_file']
    elif data_type == 'validation':
        return DATA_CONFIG['processed_data_dir'] / DATA_CONFIG['validation_file']
    elif data_type == 'raw':
        return DATA_CONFIG['raw_data_dir']
    elif data_type == 'processed':
        return DATA_CONFIG['processed_data_dir']
    else:
        raise ValueError(f"Unknown data type: {data_type}")

# Create directories when config is imported
if __name__ == "__main__":
    create_directories()
    print("Configuration loaded and directories created successfully!")
