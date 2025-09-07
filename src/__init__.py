"""
Binary Classification Project Source Package

This package contains modules for data processing, model training,
and evaluation of binary classification models.
"""

from .data_processing import DataProcessor, create_sample_data
from .model import ModelTrainer
from .evaluation import ModelEvaluator, find_optimal_threshold

__version__ = "0.1.0"
__all__ = [
    "DataProcessor",
    "create_sample_data",
    "ModelTrainer", 
    "ModelEvaluator",
    "find_optimal_threshold"
]
