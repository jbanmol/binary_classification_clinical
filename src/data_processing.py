"""
Data Processing Module

Handles data loading, preprocessing, feature engineering, and data validation
for the binary classification project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import logging
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path

# Optional representation learning
try:
    from src.representation import RepresentationLearner, RepresentationConfig  # type: ignore
except Exception:
    try:
        from representation import RepresentationLearner, RepresentationConfig  # type: ignore
    except Exception:
        RepresentationLearner = None  # type: ignore
        RepresentationConfig = None  # type: ignore

logger = logging.getLogger(__name__)


class DataProcessor:
    """Main class for data processing operations."""
    
    def __init__(self, scaler_type: str = 'standard', 
                 imputer_strategy: str = 'median',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 representation_config: Optional[Dict] = None,
                 categorical_config: Optional[Dict] = None):
        """
        Initialize DataProcessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            imputer_strategy: Strategy for imputation ('mean', 'median', 'most_frequent', 'knn')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            representation_config: Configuration dict for representation learning step
        """
        self.scaler_type = scaler_type
        self.imputer_strategy = imputer_strategy
        self.test_size = test_size
        self.random_state = random_state
        
        self.scaler = None
        self.imputer = None
        self.feature_names = None

        # Representation learning state
        self.representation_config = representation_config or {"enabled": False}
        self._rep_learner = None
        
        # Categorical encoding state
        self.categorical_config = categorical_config or {
            "enabled": True,
            "method": "onehot",  # options: onehot, ordinal
            "max_onehot_categories": 20,
            "id_unique_ratio": 0.8,  # drop columns with >=80% unique values (likely IDs)
            "exclude_columns": ["child_id", "filename", "session_id", "id", "uid"],
        }
        self.cat_imputer = None
        self.onehot_encoder = None
        self.ordinal_encoder = None
        self.categorical_columns = []
        self.numeric_columns = []
        
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
    
    def _identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Identify numeric, categorical, and id-like columns.
        """
        exclude = set(self.categorical_config.get("exclude_columns", []))
        id_ratio = float(self.categorical_config.get("id_unique_ratio", 0.8))

        numeric_cols = X.select_dtypes(include=[np.number, np.bool_]).columns.tolist()
        categorical_candidates = [c for c in X.columns if c not in numeric_cols]

        id_like = []
        cat_cols = []
        n = len(X)
        for c in categorical_candidates:
            if c in exclude:
                id_like.append(c)
                continue
            nunique = X[c].nunique(dropna=True)
            if n > 0 and (nunique / n) >= id_ratio:
                id_like.append(c)
            else:
                cat_cols.append(c)
        return numeric_cols, cat_cols, id_like

    def impute_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Impute missing values in categorical features with most_frequent.
        """
        if X.shape[1] == 0:
            return X
        if self.cat_imputer is None:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
        if fit:
            X_imp = pd.DataFrame(self.cat_imputer.fit_transform(X), columns=X.columns, index=X.index)
        else:
            X_imp = pd.DataFrame(self.cat_imputer.transform(X), columns=X.columns, index=X.index)
        return X_imp

    def encode_categorical(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical columns using OneHot for low-cardinality and Ordinal for higher-cardinality.
        """
        if X_train.shape[1] == 0:
            return (pd.DataFrame(index=X_train.index), pd.DataFrame(index=X_test.index))

        max_onehot = int(self.categorical_config.get("max_onehot_categories", 20))

        # Determine which columns to one-hot vs ordinal
        onehot_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=True) <= max_onehot]
        ordinal_cols = [c for c in X_train.columns if c not in onehot_cols]

        Xtr_parts = []
        Xte_parts = []

        if onehot_cols:
            self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            Xtr_ohe = self.onehot_encoder.fit_transform(X_train[onehot_cols])
            Xte_ohe = self.onehot_encoder.transform(X_test[onehot_cols])
            try:
                ohe_cols = self.onehot_encoder.get_feature_names_out(onehot_cols).tolist()
            except Exception:
                ohe_cols = []
                for i, c in enumerate(onehot_cols):
                    cats = self.onehot_encoder.categories_[i]
                    ohe_cols.extend([f"{c}__{cat}" for cat in cats])
            Xtr_parts.append(pd.DataFrame(Xtr_ohe, columns=ohe_cols, index=X_train.index))
            Xte_parts.append(pd.DataFrame(Xte_ohe, columns=ohe_cols, index=X_test.index))

        if ordinal_cols:
            self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            Xtr_ord = self.ordinal_encoder.fit_transform(X_train[ordinal_cols])
            Xte_ord = self.ordinal_encoder.transform(X_test[ordinal_cols])
            ord_cols = [f"ord_{c}" for c in ordinal_cols]
            Xtr_parts.append(pd.DataFrame(Xtr_ord, columns=ord_cols, index=X_train.index))
            Xte_parts.append(pd.DataFrame(Xte_ord, columns=ord_cols, index=X_test.index))

        X_train_enc = pd.concat(Xtr_parts, axis=1) if Xtr_parts else pd.DataFrame(index=X_train.index)
        X_test_enc = pd.concat(Xte_parts, axis=1) if Xte_parts else pd.DataFrame(index=X_test.index)
        return X_train_enc, X_test_enc

    def learn_representations(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Learn latent representations on training data and transform both train and test.

        Args:
            X_train: Scaled training features
            X_test: Scaled test features

        Returns:
            Tuple of augmented (X_train, X_test) with learned representation columns appended
        """
        cfg = self.representation_config or {"enabled": False}
        if not cfg.get("enabled", False):
            return X_train, X_test

        if RepresentationLearner is None:
            logger.warning("Representation module not available; skipping representation learning.")
            return X_train, X_test

        # Fit on numeric columns only (by this point, data should be numeric)
        rep_cfg = RepresentationConfig(
            method=cfg.get("method", "umap"),
            n_components=int(cfg.get("n_components", 16)),
            random_state=int(cfg.get("random_state", self.random_state)),
            ae_hidden_dims=cfg.get("ae_hidden_dims", [64, 32]),
            ae_dropout=float(cfg.get("ae_dropout", 0.0)),
            ae_lr=float(cfg.get("ae_lr", 1e-3)),
            ae_batch_size=int(cfg.get("ae_batch_size", 128)),
            ae_epochs=int(cfg.get("ae_epochs", 50)),
        )
        learner = RepresentationLearner(rep_cfg)

        logger.info(f"Learning representations using method='{rep_cfg.method}' with {rep_cfg.n_components} components")
        Z_train = learner.fit_transform(X_train)
        Z_test = learner.transform(X_test)

        # Persist learner for later transforms (e.g., predict mode)
        self._rep_learner = learner

        # Append to original dataframes
        X_train_aug = pd.concat([X_train, Z_train], axis=1)
        X_test_aug = pd.concat([X_test, Z_test], axis=1)
        logger.info(f"Appended representation features: +{Z_train.shape[1]} columns (now {X_train_aug.shape[1]})")
        return X_train_aug, X_test_aug
    
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
                           k_features: int = 20,
                           use_representation: Optional[bool] = None) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
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
        
        # Engineer features if requested (operates only on numeric columns internally)
        if engineer_features:
            X = self.engineer_features(X)
        
        # Split data to avoid leakage
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Identify feature types on training data
        num_cols, cat_cols, id_like = self._identify_feature_types(X_train)
        if id_like:
            logger.info(f"Excluding ID-like columns: {id_like}")
        
        # Drop id-like columns
        X_train = X_train.drop(columns=id_like, errors='ignore')
        X_test = X_test.drop(columns=id_like, errors='ignore')

        # Recompute types after drop
        num_cols, cat_cols, _ = self._identify_feature_types(X_train)
        self.numeric_columns = num_cols
        self.categorical_columns = cat_cols

        # Impute numeric
        if num_cols:
            X_train_num = self.handle_missing_values(X_train[num_cols])
            X_test_num = pd.DataFrame(
                self.imputer.transform(X_test[num_cols]) if self.imputer is not None else X_test[num_cols],
                columns=num_cols,
                index=X_test.index
            )
        else:
            X_train_num = pd.DataFrame(index=X_train.index)
            X_test_num = pd.DataFrame(index=X_test.index)

        # Impute + encode categorical
        if cat_cols and self.categorical_config.get("enabled", True):
            X_train_cat = self.impute_categorical(X_train[cat_cols], fit=True)
            X_test_cat = self.impute_categorical(X_test[cat_cols], fit=False)
            X_train_cat_enc, X_test_cat_enc = self.encode_categorical(X_train_cat, X_test_cat)
        else:
            X_train_cat_enc = pd.DataFrame(index=X_train.index)
            X_test_cat_enc = pd.DataFrame(index=X_test.index)

        # Combine
        X_train_combined = pd.concat([X_train_num, X_train_cat_enc], axis=1)
        X_test_combined = pd.concat([X_test_num, X_test_cat_enc], axis=1)

        # Scale features
        X_train_scaled = self.scale_features(X_train_combined, fit=True)
        X_test_scaled = self.scale_features(X_test_combined, fit=False)

        # Representation learning
        rep_enabled = self.representation_config.get("enabled", False) if use_representation is None else bool(use_representation)
        if rep_enabled:
            X_train_scaled, X_test_scaled = self.learn_representations(X_train_scaled, X_test_scaled)
        
        # Feature selection
        if select_features:
            X_train_scaled = self.select_features(X_train_scaled, y_train, k=k_features)
            X_test_scaled = X_test_scaled[X_train_scaled.columns]
        
        logger.info("Preprocessing pipeline completed")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
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
