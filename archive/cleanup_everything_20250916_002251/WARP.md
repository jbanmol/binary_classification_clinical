# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary project directories
python config.py
```

### Model Training and Evaluation
```bash
# Train a model (main entry point not fully implemented yet)
python main.py --mode train --data-path data/train.csv

# Evaluate model performance
python main.py --mode evaluate --model-path models/best_model.pkl --data-path data/test.csv

# Make predictions on new data
python main.py --mode predict --model-path models/best_model.pkl --data-path data/new_data.csv
```

### Code Quality and Testing
```bash
# Format code with black
black src/ tests/ *.py

# Lint code with flake8
flake8 src/ tests/ *.py

# Type checking with mypy
mypy src/ tests/ *.py

# Run tests with coverage
pytest --cov=src tests/ -v

# Run a specific test
pytest tests/test_file.py::test_function_name -v
```

### MLflow Experiment Tracking
```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Access at http://localhost:5000
```

### Jupyter Notebooks
```bash
# Start Jupyter notebook server
jupyter notebook

# Convert notebook to script
jupyter nbconvert --to script notebooks/experiment.ipynb
```

## Code Architecture

### High-Level Structure
This binary classification project is designed with a modular architecture focused on achieving high sensitivity and specificity. The codebase follows a pipeline pattern with clear separation of concerns:

1. **Data Pipeline** (`src/data_processing.py`): Handles all data ingestion, validation, preprocessing, and feature engineering. The pipeline supports:
   - Missing value imputation strategies
   - Feature scaling and normalization
   - Feature engineering (polynomial, interaction, log features)
   - Train/validation/test splitting with stratification
   - Class imbalance handling

2. **Model Pipeline** (`src/model.py`): Implements a comprehensive model training framework that:
   - Supports 12 different classification algorithms (from logistic regression to neural networks)
   - Includes automated hyperparameter tuning via GridSearchCV
   - Implements custom scoring metrics that balance sensitivity and specificity
   - Provides model calibration for probability estimates
   - Handles both tree-based models (XGBoost, LightGBM) and traditional ML algorithms

3. **Evaluation Framework** (`src/evaluation.py`): Comprehensive evaluation system that:
   - Calculates 10+ performance metrics including sensitivity, specificity, MCC, and AUC scores
   - Implements threshold optimization for optimal sensitivity/specificity trade-off
   - Generates visualization plots (ROC curves, precision-recall curves, confusion matrices)
   - Provides model interpretation via SHAP values
   - Exports detailed evaluation reports

4. **Configuration System** (`config.py`): Centralized configuration that:
   - Defines all hyperparameters, paths, and model settings
   - Provides model-specific configurations (XGBoost, LightGBM, Neural Network)
   - Controls feature engineering and optimization strategies
   - Manages experiment tracking and hardware settings

### Key Design Patterns

1. **Pipeline Pattern**: Each module (data processing, model training, evaluation) operates as a self-contained pipeline with clear interfaces.

2. **Strategy Pattern**: Multiple imputation, scaling, and model strategies can be swapped via configuration without code changes.

3. **Factory Pattern**: Model initialization uses a factory approach to create different classifier instances based on configuration.

4. **Custom Scoring**: The framework implements a custom scoring function that weights sensitivity and specificity according to business requirements.

### Integration Points

- **MLflow Integration**: Automatic experiment tracking for all model runs, metrics, and artifacts
- **Multiple Model Support**: Seamlessly switch between different algorithms via configuration
- **Threshold Optimization**: Post-training threshold tuning to optimize for specific business metrics
- **Class Imbalance Handling**: Multiple strategies including class weights, SMOTE, and ADASYN

### Data Flow
```
Raw Data → Data Processing → Feature Engineering → Model Training → 
Threshold Optimization → Evaluation → Model Serialization
```

The main entry point (`main.py`) orchestrates these components but is currently a skeleton implementation. The actual pipeline logic resides in the individual modules which need to be integrated into the main script.
