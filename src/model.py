"""
Model Training Module

Implements various binary classification models with emphasis on 
optimizing sensitivity and specificity.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, recall_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Main class for model training and selection."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = self._initialize_models()
        self.best_model = None
        self.best_params = None
        self.best_threshold = 0.5
        
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize a collection of binary classification models.
        
        Returns:
            Dictionary of model names and instances
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                class_weight='balanced',
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                random_state=self.random_state,
                n_estimators=100,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'adaboost': AdaBoostClassifier(
                random_state=self.random_state,
                n_estimators=50
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=1
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=self.random_state,
                n_estimators=100,
                class_weight='balanced',
                verbose=-1
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'naive_bayes': GaussianNB(),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=1000
            )
        }
        
        return models
    
    def get_param_grids(self) -> Dict[str, Dict[str, List]]:
        """
        Get parameter grids for hyperparameter tuning.
        
        Returns:
            Dictionary of model names and their parameter grids
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'decision_tree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'extra_trees': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'adaboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 50, 100],
                'min_child_samples': [10, 20, 30]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'naive_bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        return param_grids
    
    def custom_sensitivity_specificity_scorer(self, y_true, y_pred, 
                                            sensitivity_weight: float = 0.7) -> float:
        """
        Custom scorer that balances sensitivity and specificity.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitivity_weight: Weight for sensitivity (1 - weight for specificity)
            
        Returns:
            Weighted score
        """
        # Calculate sensitivity (true positive rate)
        sensitivity = recall_score(y_true, y_pred, pos_label=1)
        
        # Calculate specificity (true negative rate)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        
        # Weighted combination
        score = (sensitivity_weight * sensitivity + 
                (1 - sensitivity_weight) * specificity)
        
        return score
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   tune_hyperparameters: bool = True, cv_folds: int = 5,
                   sensitivity_weight: float = 0.7) -> Tuple[Any, Dict, float]:
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            sensitivity_weight: Weight for sensitivity in custom scorer
            
        Returns:
            Trained model, best parameters, and cross-validation score
        """
        logger.info(f"Training {model_name}...")
        
        model = self.models[model_name]
        
        # Create custom scorer
        scorer = make_scorer(
            self.custom_sensitivity_specificity_scorer,
            sensitivity_weight=sensitivity_weight
        )
        
        if tune_hyperparameters and model_name in self.get_param_grids():
            # Perform grid search
            param_grid = self.get_param_grids()[model_name]
            grid_search = GridSearchCV(
                model,
                param_grid,
                scoring=scorer,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                 random_state=self.random_state),
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info(f"{model_name} - Best params: {best_params}")
            logger.info(f"{model_name} - Best CV score: {best_score:.4f}")
            
        else:
            # Train without hyperparameter tuning
            best_model = model
            best_model.fit(X_train, y_train)
            
            # Calculate cross-validation score
            cv_scores = cross_val_score(
                best_model, X_train, y_train,
                scoring=scorer,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                 random_state=self.random_state)
            )
            best_score = cv_scores.mean()
            best_params = {}
            
            logger.info(f"{model_name} - CV score: {best_score:.4f}")
        
        return best_model, best_params, best_score
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        model_names: Optional[List[str]] = None,
                        tune_hyperparameters: bool = True,
                        cv_folds: int = 5,
                        sensitivity_weight: float = 0.7) -> Dict[str, Dict]:
        """
        Train all specified models and compare their performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_names: List of model names to train (None for all)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            sensitivity_weight: Weight for sensitivity in custom scorer
            
        Returns:
            Dictionary with results for each model
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = {}
        
        for model_name in model_names:
            try:
                model, params, score = self.train_model(
                    model_name, X_train, y_train,
                    tune_hyperparameters, cv_folds,
                    sensitivity_weight
                )
                
                results[model_name] = {
                    'model': model,
                    'params': params,
                    'cv_score': score
                }
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Find best model
        best_model_name = max(results.keys(), 
                            key=lambda k: results[k]['cv_score'])
        self.best_model = results[best_model_name]['model']
        self.best_params = results[best_model_name]['params']
        
        logger.info(f"\nBest model: {best_model_name}")
        logger.info(f"Best CV score: {results[best_model_name]['cv_score']:.4f}")
        
        return results
    
    def optimize_threshold(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series,
                         sensitivity_weight: float = 0.7) -> float:
        """
        Find optimal probability threshold for binary classification.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            sensitivity_weight: Weight for sensitivity
            
        Returns:
            Optimal threshold
        """
        logger.info("Optimizing probability threshold...")
        
        # Get predicted probabilities
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            score = self.custom_sensitivity_specificity_scorer(
                y_val, y_pred, sensitivity_weight
            )
            scores.append(score)
        
        # Find best threshold
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        
        logger.info(f"Optimal threshold: {best_threshold:.3f}")
        logger.info(f"Score at optimal threshold: {scores[best_idx]:.4f}")
        
        self.best_threshold = best_threshold
        
        return best_threshold
    
    def calibrate_model(self, model: Any, X_train: pd.DataFrame, 
                       y_train: pd.Series, method: str = 'sigmoid') -> Any:
        """
        Calibrate model probabilities.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            method: Calibration method ('sigmoid' or 'isotonic')
            
        Returns:
            Calibrated model
        """
        logger.info(f"Calibrating model using {method} method...")
        
        calibrated = CalibratedClassifierCV(
            model,
            method=method,
            cv=3
        )
        calibrated.fit(X_train, y_train)
        
        return calibrated
    
    def save_model(self, model: Any, file_path: str):
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            file_path: Path to save the model
        """
        logger.info(f"Saving model to {file_path}")
        joblib.dump(model, file_path)
        
        # Also save threshold if available
        if self.best_threshold != 0.5:
            threshold_path = file_path.replace('.pkl', '_threshold.pkl')
            joblib.dump(self.best_threshold, threshold_path)
    
    def load_model(self, file_path: str) -> Any:
        """
        Load trained model from disk.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {file_path}")
        model = joblib.load(file_path)
        
        # Try to load threshold
        threshold_path = file_path.replace('.pkl', '_threshold.pkl')
        try:
            self.best_threshold = joblib.load(threshold_path)
        except:
            self.best_threshold = 0.5
        
        return model
    
    def predict_with_threshold(self, model: Any, X: pd.DataFrame, 
                             threshold: Optional[float] = None) -> np.ndarray:
        """
        Make predictions using custom threshold.
        
        Args:
            model: Trained model
            X: Features to predict
            threshold: Probability threshold (uses best_threshold if None)
            
        Returns:
            Binary predictions
        """
        if threshold is None:
            threshold = self.best_threshold
        
        y_proba = model.predict_proba(X)[:, 1]
        return (y_proba >= threshold).astype(int)


if __name__ == "__main__":
    # Test the model trainer
    from data_processing import create_sample_data, DataProcessor
    
    logger.info("Testing model trainer with sample data")
    
    # Create and process sample data
    df = create_sample_data(n_samples=1000, n_features=10, class_balance=0.3)
    
    processor = DataProcessor()
    processed_data = processor.preprocess_pipeline(
        df, 
        target_column='target',
        engineer_features=False,
        select_features=False
    )
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train a single model
    model, params, score = trainer.train_model(
        'random_forest',
        processed_data['X_train'],
        processed_data['y_train'],
        tune_hyperparameters=True,
        sensitivity_weight=0.7
    )
    
    print(f"\nTrained model with CV score: {score:.4f}")
    print(f"Best parameters: {params}")
