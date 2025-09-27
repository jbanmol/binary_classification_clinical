"""Device robustness preprocessor implementing CORAL alignment with safeguards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .core import DomainShiftDetector, DomainAdapter


@dataclass
class AdaptationStats:
    means: pd.Series
    stds: pd.Series
    correlations: pd.DataFrame
    samples: int


@dataclass
class DeviceRobustPreprocessor:
    """Conservative domain adaptation for deployment-time robustness."""

    adaptation_strength: float = 0.4
    min_samples: int = 50
    detector: DomainShiftDetector = field(default_factory=DomainShiftDetector)
    adapter: DomainAdapter = field(default_factory=DomainAdapter)

    def __post_init__(self) -> None:
        if not 0.0 <= self.adaptation_strength <= 1.0:
            raise ValueError("adaptation_strength must be between 0 and 1")
        self.training_stats: Optional[AdaptationStats] = None
        self.training_sample: Optional[pd.DataFrame] = None

    def fit_training_distribution(self, training_data: pd.DataFrame) -> None:
        if training_data.empty:
            raise ValueError("training_data must not be empty")
        numeric = training_data.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("training_data must contain numeric features")
        self.training_stats = AdaptationStats(
            means=numeric.mean(),
            stds=numeric.std(ddof=0).replace(0, 1.0),
            correlations=numeric.corr().fillna(0.0),
            samples=int(len(numeric)),
        )
        sample_size = min(len(numeric), max(self.min_samples, 500))
        self.training_sample = numeric.sample(n=sample_size, random_state=42)

    def adapt(self, target_data: pd.DataFrame) -> pd.DataFrame:
        if self.training_stats is None:
            return target_data
        if self.training_sample is None or self.training_sample.empty:
            return target_data

        numeric_cols = target_data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            return target_data

        source_stats = self.training_stats
        if source_stats.samples < self.min_samples:
            return target_data

        target_numeric = target_data[numeric_cols].copy()
        source_df = self.training_sample[numeric_cols].copy()
        if source_df.empty:
            return target_data

        analysis = self.detector.comprehensive_shift_analysis(source_df, target_numeric)
        if analysis.get("overall_shift_severity") == "MINIMAL":
            return target_data

        adapted_numeric = self._coral_blend(source_df, target_numeric)

        result = target_data.copy()
        result[numeric_cols] = adapted_numeric
        return result

    def _coral_blend(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        common_cols = list(set(source_df.columns) & set(target_df.columns))
        if not common_cols:
            return target_df

        source = source_df[common_cols].fillna(0.0).to_numpy(dtype=float)
        target = target_df[common_cols].fillna(0.0).to_numpy(dtype=float)

        try:
            source_adapted, _ = self.adapter.coral_alignment(source, target)
        except Exception:
            return target_df

        if len(source_adapted) != len(target_df):
            idx = np.random.choice(len(source_adapted), size=len(target_df), replace=len(source_adapted) < len(target_df))
            aligned = source_adapted[idx]
        else:
            aligned = source_adapted
        adapted = pd.DataFrame(aligned, columns=common_cols, index=target_df.index)
        blended = (
            (1.0 - self.adaptation_strength) * target_df[common_cols]
            + self.adaptation_strength * adapted
        )
        adjusted = blended.copy()
        for col in common_cols:
            source_std = float(self.training_stats.stds.get(col, 1.0)) if self.training_stats else 1.0
            if source_std <= 0:
                source_std = 1.0
            diff = (adjusted[col] - target_df[col]).abs()
            mask = diff > (3.0 * source_std)
            adjusted.loc[mask, col] = target_df.loc[mask, col]
        result = target_df.copy()
        result[common_cols] = adjusted
        return result

    def to_dict(self) -> Dict[str, Any]:
        if self.training_stats is None:
            return {}
        return {
            "means": self.training_stats.means.to_dict(),
            "stds": self.training_stats.stds.to_dict(),
            "samples": self.training_stats.samples,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        if not data:
            self.training_stats = None
            self.training_sample = None
            return
        self.training_stats = AdaptationStats(
            means=pd.Series(data.get("means", {})),
            stds=pd.Series(data.get("stds", {})),
            correlations=pd.DataFrame(),
            samples=int(data.get("samples", 0)),
        )
        self.training_sample = None
