"""Safe enhancement pipeline orchestrating baseline vs demographic+device runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.demographics import SafeDemographicManager, demographic_manager
from src.domain_adaptation import DeviceRobustPreprocessor


@dataclass
class PipelineResult:
    strategy: str
    metrics: Dict[str, float]
    details: Dict[str, Any]


@dataclass
class SafeEnhancementPipeline:
    """Orchestrates baseline vs enhanced evaluation with safety fallback."""

    min_sensitivity: float = 0.86
    min_specificity: float = 0.74
    margin: float = 0.02
    demographic_manager: SafeDemographicManager = field(default_factory=SafeDemographicManager)
    device_preprocessor: DeviceRobustPreprocessor = field(default_factory=DeviceRobustPreprocessor)

    def fit_baseline(
        self,
        features_train: pd.DataFrame,
        labels_train: np.ndarray,
        predict_fn: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]],
        validation_data: Dict[str, Any],
        child_ids: Iterable[str],
    ) -> PipelineResult:
        y_pred, y_prob = predict_fn(validation_data["X_val"])  # type: ignore
        metrics = self._compute_metrics(validation_data["y_val"], y_pred, y_prob)
        return PipelineResult("baseline", metrics, {"child_ids": list(child_ids)})

    def fit_enhanced(
        self,
        features_train: pd.DataFrame,
        labels_train: np.ndarray,
        predict_fn: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]],
        validation_data: Dict[str, Any],
        demo_dir: Optional[str] = None,
        metadata_roots: Optional[Iterable[str]] = None,
    ) -> PipelineResult:
        demographics_loaded = False
        if demo_dir:
            demographics_loaded = self.demographic_manager.load(demo_dir, metadata_roots)
        training_ref = validation_data.get("training_reference")
        if training_ref is not None:
            try:
                self.device_preprocessor.fit_training_distribution(training_ref)
            except Exception:
                training_ref = None
        X_val = validation_data["X_val"].copy()
        if training_ref is not None:
            try:
                X_val = self.device_preprocessor.adapt(X_val)
            except Exception:
                pass
        child_ids = validation_data.get("child_ids", [])
        y_pred, y_prob = predict_fn(X_val)
        if demographics_loaded and child_ids:
            y_pred = self._apply_demographic_thresholds(y_prob, child_ids)
        metrics = self._compute_metrics(validation_data["y_val"], y_pred, y_prob)
        details = {
            "demographics_loaded": demographics_loaded,
            "device_adaptation": training_ref is not None,
            "child_ids": list(child_ids),
        }
        return PipelineResult("enhanced", metrics, details)

    def run(
        self,
        baseline_result: PipelineResult,
        enhanced_result: PipelineResult,
    ) -> PipelineResult:
        if self._safe(enhanced_result.metrics) and self._non_degradation(
            baseline_result.metrics, enhanced_result.metrics
        ):
            return enhanced_result
        return baseline_result

    def _safe(self, metrics: Dict[str, float]) -> bool:
        sens_floor = self.min_sensitivity * (1 - self.margin)
        spec_floor = self.min_specificity * (1 - self.margin)
        return (
            metrics.get("sensitivity", 0.0) >= sens_floor
            and metrics.get("specificity", 0.0) >= spec_floor
        )

    def _non_degradation(
        self, baseline: Dict[str, float], enhanced: Dict[str, float]
    ) -> bool:
        return (
            enhanced.get("sensitivity", 0.0) >= baseline.get("sensitivity", 0.0) * 0.99
            and enhanced.get("specificity", 0.0) >= baseline.get("specificity", 0.0) * 0.99
        )

    def _apply_demographic_thresholds(self, probs: np.ndarray, child_ids: Iterable[str]) -> np.ndarray:
        tau_default = 0.5
        labels = np.zeros_like(probs, dtype=int)
        for idx, cid in enumerate(child_ids):
            thr_demo = float(demographic_manager.get_clinical_threshold(cid))
            thr = max(thr_demo, tau_default)
            labels[idx] = 1 if probs[idx] >= thr else 0
        return labels

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        from sklearn.metrics import roc_auc_score, confusion_matrix

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else 0.0
        return {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "auc": auc,
        }

