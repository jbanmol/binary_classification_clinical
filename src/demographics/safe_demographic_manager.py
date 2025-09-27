"""Safe demographic integration with performance safeguards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd

from .demographic_manager import DemographicManager, demographic_manager


@dataclass(frozen=True)
class SafetyThresholds:
    min_sensitivity: float
    min_specificity: float
    margin: float = 0.02

    @property
    def sensitivity_floor(self) -> float:
        return self.min_sensitivity * (1.0 - self.margin)

    @property
    def specificity_floor(self) -> float:
        return self.min_specificity * (1.0 - self.margin)


class SafeDemographicManager:
    """Wraps demographic-aware enhancements with safety validation and fallback."""

    def __init__(
        self,
        base_manager: DemographicManager = demographic_manager,
        thresholds: SafetyThresholds = SafetyThresholds(0.86, 0.74),
    ) -> None:
        self.base_manager = base_manager
        self.thresholds = thresholds

    def load(self, demo_dir: str, metadata_roots: Optional[Iterable[str]] = None) -> bool:
        roots = None if metadata_roots is None else [Path(r) for r in metadata_roots]
        return self.base_manager.load_from_dir(Path(demo_dir), metadata_roots=roots)

    def enhance_features(
        self,
        features: pd.DataFrame,
        child_ids: Iterable[str],
        demographics: Optional[Mapping[str, Dict[str, Any]]] = None,
        use_gender_masking: bool = True,
    ) -> pd.DataFrame:
        import warnings

        from .age_features import create_age_developmental_features
        from .gender_features import augment_with_gender_masking

        enhanced_rows: list[MutableMapping[str, Any]] = []

        for idx, child_id in enumerate(child_ids):
            base_row = features.iloc[idx].to_dict()
            demo = (
                demographics[str(child_id)]
                if demographics and str(child_id) in demographics
                else self.base_manager.get_child_demographics(child_id)
            )

            age = demo.get("age")
            if age is not None:
                age_months = float(age) * 12.0 if age < 10 else float(age)
            else:
                age_months = None

            row_augmented = create_age_developmental_features(age_months, base_row)
            if use_gender_masking:
                try:
                    row_augmented = augment_with_gender_masking(
                        demo.get("gender"),
                        row_augmented,
                        base_row,
                    )
                except Exception as exc:
                    warnings.warn(f"Gender masking augmentation failed for {child_id}: {exc}")

            row_augmented["child_id"] = child_id
            enhanced_rows.append(row_augmented)

        return pd.DataFrame(enhanced_rows)

    def run_with_fallback(
        self,
        baseline_metrics: Mapping[str, float],
        enhanced_metrics: Mapping[str, float],
    ) -> bool:
        if not self._meets_thresholds(enhanced_metrics):
            return False

        if not self._non_degradation(baseline_metrics, enhanced_metrics):
            return False

        return True

    def _meets_thresholds(self, metrics: Mapping[str, float]) -> bool:
        sens = float(metrics.get("sensitivity", 0.0))
        spec = float(metrics.get("specificity", 0.0))
        return sens >= self.thresholds.sensitivity_floor and spec >= self.thresholds.specificity_floor

    def _non_degradation(
        self, baseline: Mapping[str, float], enhanced: Mapping[str, float]
    ) -> bool:
        sens_base = float(baseline.get("sensitivity", 0.0))
        spec_base = float(baseline.get("specificity", 0.0))
        sens_new = float(enhanced.get("sensitivity", 0.0))
        spec_new = float(enhanced.get("specificity", 0.0))

        return sens_new >= sens_base * 0.99 and spec_new >= spec_base * 0.99


safe_demographic_manager = SafeDemographicManager()


