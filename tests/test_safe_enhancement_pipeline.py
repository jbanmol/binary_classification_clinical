"""Tests for SafeEnhancementPipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines.safe_enhancement_pipeline import (
    SafeEnhancementPipeline,
    PipelineResult,
)


def dummy_predict(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    probs = np.linspace(0.2, 0.8, num=n)
    labels = (probs >= 0.5).astype(int)
    return labels, probs


def test_pipeline_prefers_enhanced_when_safe() -> None:
    pipeline = SafeEnhancementPipeline()
    X_train = pd.DataFrame({"a": np.random.randn(100)})
    y_train = np.random.randint(0, 2, size=100)
    X_val = pd.DataFrame({"a": np.random.randn(30)})
    y_val = np.random.randint(0, 2, size=30)

    baseline = PipelineResult(
        strategy="baseline",
        metrics={"sensitivity": 0.87, "specificity": 0.75, "auc": 0.9},
        details={},
    )
    enhanced = PipelineResult(
        strategy="enhanced",
        metrics={"sensitivity": 0.88, "specificity": 0.76, "auc": 0.91},
        details={"demographics_loaded": True, "device_adaptation": True},
    )

    result = pipeline.run(baseline, enhanced)
    assert result.strategy == "enhanced"


def test_pipeline_fallbacks_on_degradation() -> None:
    pipeline = SafeEnhancementPipeline()
    baseline = PipelineResult(
        strategy="baseline",
        metrics={"sensitivity": 0.88, "specificity": 0.78, "auc": 0.9},
        details={},
    )
    enhanced = PipelineResult(
        strategy="enhanced",
        metrics={"sensitivity": 0.80, "specificity": 0.72, "auc": 0.91},
        details={"demographics_loaded": True, "device_adaptation": True},
    )

    result = pipeline.run(baseline, enhanced)
    assert result.strategy == "baseline"


