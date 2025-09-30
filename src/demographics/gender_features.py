"""Gender-aware feature augmentation for masking detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class GenderMaskingConfig:
    """Configuration parameters for gender masking heuristics."""

    female_safety_multiplier: float = 1.05
    masking_velocity_threshold: float = 0.85
    masking_completion_delta: float = 0.9


DEFAULT_CONFIG = GenderMaskingConfig()


def augment_with_gender_masking(
    gender: Optional[str],
    age_features: Mapping[str, float],
    base_features: Mapping[str, float],
    config: GenderMaskingConfig = DEFAULT_CONFIG,
) -> MutableMapping[str, float]:
    """Augment feature set with gender-specific masking indicators.

    Args:
        gender: Normalized gender string ("male"/"female"/None).
        age_features: Mapping of age-engineered features.
        base_features: Original base features used for heuristics.
        config: Masking configuration.

    Returns:
        Mutable mapping containing augmented features.
    """

    enhanced: MutableMapping[str, float] = dict(age_features)

    gender_norm = gender.lower() if isinstance(gender, str) else "unknown"
    enhanced["gender_female"] = float(gender_norm == "female")
    enhanced["gender_male"] = float(gender_norm == "male")

    if gender_norm != "female":
        enhanced.setdefault("masking_indicator", 0.0)
        enhanced.setdefault("masking_safety_scale", 1.0)
        return enhanced

    velocity_ratio = _safe_ratio(base_features, "velocity_std", "velocity_mean")
    completion_ratio = _safe_ratio(age_features, "completion_age_ratio", None)

    masking_signals = []
    if velocity_ratio is not None:
        masking_signals.append(float(velocity_ratio <= config.masking_velocity_threshold))
    if completion_ratio is not None:
        masking_signals.append(float(completion_ratio >= config.masking_completion_delta))

    if masking_signals:
        indicator = max(masking_signals)
        enhanced["masking_indicator"] = indicator
        enhanced["masking_safety_scale"] = 1.0 + indicator * (config.female_safety_multiplier - 1.0)
    else:
        enhanced.setdefault("masking_indicator", 0.0)
        enhanced.setdefault("masking_safety_scale", 1.0)

    return enhanced


def _safe_ratio(features: Mapping[str, float], numerator_key: str, denominator_key: Optional[str]) -> Optional[float]:
    try:
        numerator = float(features[numerator_key])
    except Exception:
        return None

    if denominator_key is None:
        return numerator

    try:
        denominator = float(features[denominator_key])
    except Exception:
        return None

    return numerator / (denominator + 1e-8) if denominator > 1e-8 else numerator


