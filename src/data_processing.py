"""
Data Processing Module

Handles data loading, preprocessing, feature engineering, and data validation
for the binary classification project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import logging
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class DataProcessor:
    """Main class for data processing operations."""
    
    def __init__(self, scaler_type: str = 'standard', 
                 imputer_strategy: str = 'median',
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize DataProcessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            imputer_strategy: Strategy for imputation ('mean', 'median', 'most_frequent', 'knn')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.scaler_type = scaler_type
        self.imputer_strategy = imputer_strategy
        self.test_size = test_size
        self.random_state = random_state
        
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features_target(self, df: pd.DataFrame, 
                               target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Features DataFrame and target Series
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with imputed values
        """
        logger.info(f"Handling missing values using {self.imputer_strategy} strategy")
        
        # Check for missing values
        missing_info = X.isnull().sum()
        if missing_info.sum() == 0:
            logger.info("No missing values found")
            return X
        
        logger.info(f"Missing values found:\n{missing_info[missing_info > 0]}")
        
        # Initialize imputer
        if self.imputer_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            self.imputer = SimpleImputer(strategy=self.imputer_strategy)
        
        # Impute missing values
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_imputed
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using specified scaler.
        
        Args:
            X: Features DataFrame
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Scaled features DataFrame
        """
        logger.info(f"Scaling features using {self.scaler_type} scaler")
        
        # Initialize scaler if not already done
        if self.scaler is None:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # Scale features
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled
    
    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features")
        
        X_eng = X.copy()
        
        # Example feature engineering (customize based on your data)
        # 1. Polynomial features for numeric columns
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limit to first 5 to avoid explosion
            X_eng[f'{col}_squared'] = X_eng[col] ** 2
            X_eng[f'{col}_sqrt'] = np.sqrt(np.abs(X_eng[col]))
        
        # 2. Interaction features
        if len(numeric_cols) >= 2:
            for i in range(min(3, len(numeric_cols) - 1)):
                for j in range(i + 1, min(i + 4, len(numeric_cols))):
                    X_eng[f'{numeric_cols[i]}_x_{numeric_cols[j]}'] = (
                        X_eng[numeric_cols[i]] * X_eng[numeric_cols[j]]
                    )
        
        # 3. Log transformations for positive numeric features
        for col in numeric_cols:
            if (X_eng[col] > 0).all():
                X_eng[f'{col}_log'] = np.log1p(X_eng[col])
        
        logger.info(f"Features after engineering: {X_eng.shape[1]} columns")
        
        return X_eng
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       k: int = 20, method: str = 'mutual_info') -> pd.DataFrame:
        """
        Select top k features based on specified method.
        
        Args:
            X: Features DataFrame
            y: Target Series
            k: Number of features to select
            method: Selection method ('chi2', 'f_classif', 'mutual_info')
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting top {k} features using {method}")
        
        # Choose scoring function
        if method == 'chi2':
            score_func = chi2
            # Ensure all features are non-negative for chi2
            X = X - X.min() + 1e-10
        elif method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Select features
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        logger.info(f"Selected features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                  pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data with test_size={self.test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_column: str,
                           engineer_features: bool = True,
                           select_features: bool = True,
                           k_features: int = 20) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            engineer_features: Whether to engineer new features
            select_features: Whether to perform feature selection
            k_features: Number of features to select
            
        Returns:
            Dictionary with processed data splits
        """
        logger.info("Starting preprocessing pipeline")
        
        # Separate features and target
        X, y = self.prepare_features_target(df, target_column)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Engineer features if requested
        if engineer_features:
            X = self.engineer_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train = self.scale_features(X_train, fit=True)
        X_test = self.scale_features(X_test, fit=False)
        
        # Select features if requested
        if select_features:
            X_train = self.select_features(X_train, y_train, k=k_features)
            # Apply same feature selection to test set
            X_test = X_test[X_train.columns]
        
        logger.info("Preprocessing pipeline completed")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }


def create_sample_data(n_samples: int = 1000, n_features: int = 10,
                      class_balance: float = 0.3) -> pd.DataFrame:
    """
    Create sample binary classification data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        class_balance: Proportion of positive class
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with specified class balance
    y = np.random.choice([0, 1], size=n_samples, p=[1-class_balance, class_balance])
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some missing values
    mask = np.random.random((n_samples, n_features)) < 0.1
    df.iloc[:, :n_features][mask] = np.nan
    
    return df


if __name__ == "__main__":
    # Test the data processor
    logger.info("Testing data processor with sample data")
    
    # Create sample data
    df = create_sample_data(n_samples=1000, n_features=10)
    
    # Initialize processor
    processor = DataProcessor()
    
    # Run preprocessing pipeline
    processed_data = processor.preprocess_pipeline(
        df, 
        target_column='target',
        engineer_features=True,
        select_features=True,
        k_features=15
    )
    
    print(f"Processed data shapes:")
    print(f"X_train: {processed_data['X_train'].shape}")
    print(f"X_test: {processed_data['X_test'].shape}")
    print(f"y_train: {processed_data['y_train'].shape}")
    print(f"y_test: {processed_data['y_test'].shape}")
