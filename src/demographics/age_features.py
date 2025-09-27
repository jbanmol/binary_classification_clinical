"""Age-aware developmental feature engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class AgeFeatureConfig:
    """Configuration for age-based developmental windows."""

    early_cutoff_months: float = 24.0
    language_window_end_months: float = 36.0
    executive_window_end_months: float = 60.0
    safety_floor_months: float = 0.0


DEFAULT_CONFIG = AgeFeatureConfig()


def create_age_developmental_features(
    age_months: Optional[float],
    base_features: Mapping[str, float],
    config: AgeFeatureConfig = DEFAULT_CONFIG,
) -> MutableMapping[str, float]:
    """Enhance features with age-dependent developmental expectations.

    Args:
        age_months: Child age in months. ``None`` if unavailable.
        base_features: Mapping of existing numerical features.
        config: Developmental window configuration.

    Returns:
        A mutable mapping containing the engineered features. The returned mapping
        is independent of the input mapping to avoid side effects.
    """

    enhanced: MutableMapping[str, float] = dict(base_features)

    if age_months is None or age_months < config.safety_floor_months:
        enhanced.update(
            {
                "critical_early": 0.0,
                "critical_language": 0.0,
                "critical_executive": 0.0,
                "post_critical": 0.0,
                "motor_development_ratio": 1.0,
                "completion_age_ratio": 1.0,
            }
        )
        return enhanced

    # Developmental critical-period flags
    enhanced["critical_early"] = float(age_months <= config.early_cutoff_months)
    enhanced["critical_language"] = float(
        config.early_cutoff_months < age_months <= config.language_window_end_months
    )
    enhanced["critical_executive"] = float(
        config.language_window_end_months < age_months <= config.executive_window_end_months
    )
    enhanced["post_critical"] = float(age_months > config.executive_window_end_months)

    # Age-normalized motor control expectations
    velocity_std = _safe_get(base_features, "velocity_std")
    velocity_mean = _safe_get(base_features, "velocity_mean")
    if velocity_std is not None and velocity_mean is not None:
        expected_motor_control = max(0.5, 1.2 - (age_months - 24.0) * 0.015)
        actual_control = velocity_std / (velocity_mean + 1e-8)
        enhanced["motor_development_ratio"] = float(actual_control / expected_motor_control)
    else:
        enhanced.setdefault("motor_development_ratio", 1.0)

    # Age-normalized completion expectations
    final_completion = _safe_get(base_features, "final_completion")
    if final_completion is not None:
        expected_completion = min(95.0, 60.0 + (age_months - 24.0) * 0.8)
        enhanced["completion_age_ratio"] = float(final_completion / (expected_completion + 1e-8))
    else:
        enhanced.setdefault("completion_age_ratio", 1.0)

    # Age percentile helpers
    enhanced["age_months"] = float(age_months)
    enhanced["age_years"] = float(age_months / 12.0)

    return enhanced


def _safe_get(features: Mapping[str, float], key: str) -> Optional[float]:
    value = features.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


