"""
Evaluation Module

Comprehensive evaluation utilities for binary classification models
with focus on sensitivity, specificity, and other key metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss,
    balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
import logging
from typing import Dict, Tuple, Optional, Any, Union, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Main class for model evaluation and visualization."""
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.evaluation_results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None,
                         pos_label: int = 1) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            pos_label: Positive class label
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        
        # Sensitivity and Specificity
        metrics['sensitivity'] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics['specificity'] = recall_score(y_true, y_pred, pos_label=1-pos_label, zero_division=0)
        
        # Balanced metrics
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Matthews Correlation Coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Cohen's Kappa
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Metrics requiring probabilities
        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            metrics['auc_pr'] = average_precision_score(y_true, y_proba)
            metrics['log_loss'] = log_loss(y_true, y_proba)
        
        # Confusion matrix values
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = tp
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            
            # Additional derived metrics
            metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def print_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray] = None,
                               model_name: str = "Model") -> Dict[str, float]:
        """
        Print comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model for display
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*50}")
        print(f"Evaluation Report for {model_name}")
        print(f"{'='*50}\n")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_proba)
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=['Class 0', 'Class 1']))
        
        # Print key metrics
        print("\nKey Performance Metrics:")
        print(f"Accuracy:            {metrics['accuracy']:.4f}")
        print(f"Precision:           {metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"Specificity:         {metrics['specificity']:.4f}")
        print(f"F1 Score:            {metrics['f1_score']:.4f}")
        print(f"Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
        
        print(f"\nMCC:                 {metrics['mcc']:.4f}")
        print(f"Cohen's Kappa:       {metrics['kappa']:.4f}")
        
        if y_proba is not None:
            print(f"\nAUC-ROC:             {metrics['auc_roc']:.4f}")
            print(f"AUC-PR:              {metrics['auc_pr']:.4f}")
            print(f"Log Loss:            {metrics['log_loss']:.4f}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        if cm.shape == (2, 2):
            print(f"\nTrue Positives:  {metrics['true_positives']}")
            print(f"True Negatives:  {metrics['true_negatives']}")
            print(f"False Positives: {metrics['false_positives']}")
            print(f"False Negatives: {metrics['false_negatives']}")
        
        print(f"{'='*50}\n")
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Model",
                            normalize: bool = False,
                            figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for display
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title_suffix = " (Normalized)"
        else:
            fmt = 'd'
            title_suffix = ""
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'],
                   ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}{title_suffix}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      model_name: str = "Model",
                      figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model for display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   model_name: str = "Model",
                                   figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model for display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_proba: np.ndarray,
                              model_name: str = "Model",
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot analysis of metrics at different probability thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model for display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        thresholds = np.arange(0.05, 1.0, 0.05)
        
        metrics = {
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'f1_score': [],
            'balanced_accuracy': []
        }
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            metrics['sensitivity'].append(recall_score(y_true, y_pred, pos_label=1))
            metrics['specificity'].append(recall_score(y_true, y_pred, pos_label=0))
            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['f1_score'].append(f1_score(y_true, y_pred, zero_division=0))
            metrics['balanced_accuracy'].append(balanced_accuracy_score(y_true, y_pred))
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Sensitivity vs Specificity
        ax = axes[0, 0]
        ax.plot(thresholds, metrics['sensitivity'], label='Sensitivity', marker='o')
        ax.plot(thresholds, metrics['specificity'], label='Specificity', marker='s')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Sensitivity and Specificity vs Threshold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Precision vs Recall
        ax = axes[0, 1]
        ax.plot(thresholds, metrics['precision'], label='Precision', marker='o')
        ax.plot(thresholds, metrics['sensitivity'], label='Recall', marker='s')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall vs Threshold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 3: F1 Score
        ax = axes[1, 0]
        ax.plot(thresholds, metrics['f1_score'], color='green', marker='o')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score vs Threshold')
        ax.grid(alpha=0.3)
        
        # Plot 4: Balanced Accuracy
        ax = axes[1, 1]
        ax.plot(thresholds, metrics['balanced_accuracy'], color='purple', marker='o')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title('Balanced Accuracy vs Threshold')
        ax.grid(alpha=0.3)
        
        plt.suptitle(f'Threshold Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                             model_name: str = "Model",
                             n_bins: int = 10,
                             figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot calibration curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model for display
            n_bins: Number of bins for calibration
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
               label=model_name, markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Plot - {model_name}')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_names: List[str],
                              feature_importance: np.ndarray,
                              model_name: str = "Model",
                              top_n: int = 20,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: Names of features
            feature_importance: Importance values
            model_name: Name of the model for display
            top_n: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importance = feature_importance[indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, top_importance)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def generate_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_proba: Optional[np.ndarray] = None,
                                model_name: str = "Model",
                                save_dir: Optional[str] = None) -> List[plt.Figure]:
        """
        Generate all evaluation plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model for display
            save_dir: Directory to save plots (optional)
            
        Returns:
            List of generated figures
        """
        figures = []
        
        # Confusion Matrix
        fig = self.plot_confusion_matrix(y_true, y_pred, model_name)
        figures.append(fig)
        if save_dir:
            fig.savefig(f"{save_dir}/{model_name}_confusion_matrix.png", dpi=300)
        
        # Normalized Confusion Matrix
        fig = self.plot_confusion_matrix(y_true, y_pred, model_name, normalize=True)
        figures.append(fig)
        if save_dir:
            fig.savefig(f"{save_dir}/{model_name}_confusion_matrix_normalized.png", dpi=300)
        
        # Plots requiring probabilities
        if y_proba is not None:
            # ROC Curve
            fig = self.plot_roc_curve(y_true, y_proba, model_name)
            figures.append(fig)
            if save_dir:
                fig.savefig(f"{save_dir}/{model_name}_roc_curve.png", dpi=300)
            
            # Precision-Recall Curve
            fig = self.plot_precision_recall_curve(y_true, y_proba, model_name)
            figures.append(fig)
            if save_dir:
                fig.savefig(f"{save_dir}/{model_name}_pr_curve.png", dpi=300)
            
            # Threshold Analysis
            fig = self.plot_threshold_analysis(y_true, y_proba, model_name)
            figures.append(fig)
            if save_dir:
                fig.savefig(f"{save_dir}/{model_name}_threshold_analysis.png", dpi=300)
            
            # Calibration Curve
            fig = self.plot_calibration_curve(y_true, y_proba, model_name)
            figures.append(fig)
            if save_dir:
                fig.savefig(f"{save_dir}/{model_name}_calibration_curve.png", dpi=300)
        
        return figures
    
    def compare_models(self, results: Dict[str, Dict[str, Union[np.ndarray, str]]],
                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Compare multiple models visually.
        
        Args:
            results: Dictionary with model results
                     {model_name: {'y_true': array, 'y_pred': array, 'y_proba': array}}
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        metrics_df = pd.DataFrame()
        
        for model_name, data in results.items():
            metrics = self.calculate_metrics(
                data['y_true'], 
                data['y_pred'],
                data.get('y_proba')
            )
            metrics_df[model_name] = metrics
        
        # Select key metrics for comparison
        key_metrics = ['accuracy', 'precision', 'sensitivity', 'specificity', 
                      'f1_score', 'balanced_accuracy']
        if 'auc_roc' in metrics_df.index:
            key_metrics.append('auc_roc')
        
        comparison_df = metrics_df.loc[key_metrics].T
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        comparison_df.plot(kind='bar', ax=ax)
        ax.set_title('Model Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8)
        
        plt.tight_layout()
        return fig


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                          metric: str = 'balanced',
                          sensitivity_weight: float = 0.7) -> Tuple[float, float]:
    """
    Find optimal probability threshold for binary classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Optimization metric ('balanced', 'f1', 'youden', 'custom')
        sensitivity_weight: Weight for sensitivity when metric='custom'
        
    Returns:
        Optimal threshold and corresponding metric value
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'balanced':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'youden':
            sensitivity = recall_score(y_true, y_pred, pos_label=1)
            specificity = recall_score(y_true, y_pred, pos_label=0)
            score = sensitivity + specificity - 1  # Youden's J statistic
        elif metric == 'custom':
            sensitivity = recall_score(y_true, y_pred, pos_label=1)
            specificity = recall_score(y_true, y_pred, pos_label=0)
            score = sensitivity_weight * sensitivity + (1 - sensitivity_weight) * specificity
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return best_threshold, best_score


if __name__ == "__main__":
    # Test the evaluator
    from data_processing import create_sample_data, DataProcessor
    from model import ModelTrainer
    
    logger.info("Testing evaluator with sample data")
    
    # Create and process sample data
    df = create_sample_data(n_samples=1000, n_features=10, class_balance=0.3)
    
    processor = DataProcessor()
    processed_data = processor.preprocess_pipeline(df, target_column='target')
    
    # Train a model
    trainer = ModelTrainer()
    model, _, _ = trainer.train_model(
        'random_forest',
        processed_data['X_train'],
        processed_data['y_train'],
        tune_hyperparameters=False
    )
    
    # Make predictions
    y_pred = model.predict(processed_data['X_test'])
    y_proba = model.predict_proba(processed_data['X_test'])[:, 1]
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.print_evaluation_report(
        processed_data['y_test'],
        y_pred,
        y_proba,
        model_name="Random Forest"
    )
    
    # Find optimal threshold
    best_threshold, best_score = find_optimal_threshold(
        processed_data['y_test'],
        y_proba,
        metric='custom',
        sensitivity_weight=0.7
    )
    
    print(f"\nOptimal threshold: {best_threshold:.3f}")
    print(f"Score at optimal threshold: {best_score:.4f}")
