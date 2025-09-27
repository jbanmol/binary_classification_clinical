"""Unit tests for demographic integration utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src.demographics import (
    SafeDemographicManager,
    augment_with_gender_masking,
    create_age_developmental_features,
    demographic_manager,
)


def test_create_age_developmental_features_basic() -> None:
    base_features = {"velocity_std": 1.2, "velocity_mean": 1.0, "final_completion": 70.0}
    result = create_age_developmental_features(30.0, base_features)
    assert result["critical_language"] == 1.0
    assert result["critical_early"] == 0.0
    assert "motor_development_ratio" in result
    assert "completion_age_ratio" in result


def test_create_age_developmental_features_missing_age() -> None:
    base_features: Dict[str, float] = {}
    result = create_age_developmental_features(None, base_features)
    assert result["motor_development_ratio"] == 1.0
    assert result["completion_age_ratio"] == 1.0


def test_gender_masking_augments_female() -> None:
    age_features = {"completion_age_ratio": 1.0}
    base_features = {"velocity_std": 0.6, "velocity_mean": 1.0}
    augmented = augment_with_gender_masking("female", age_features, base_features)
    assert augmented["gender_female"] == 1.0
    assert augmented["masking_indicator"] >= 0.0
    assert augmented["masking_safety_scale"] >= 1.0


def test_gender_masking_non_female_defaults() -> None:
    age_features = {}
    base_features = {}
    augmented = augment_with_gender_masking("male", age_features, base_features)
    assert augmented["masking_indicator"] == 0.0
    assert augmented["masking_safety_scale"] == 1.0


def test_demographic_manager_metadata_override(tmp_path: Path) -> None:
    csv_dir = tmp_path / "age_data"
    csv_dir.mkdir()
    csv_file = csv_dir / "demo.csv"
    csv_file.write_text("UID,Child Age,Child Gender\nchild1,4,Male\n")

    raw_dir = tmp_path / "raw"
    child_dir = raw_dir / "OrgA" / "child1"
    child_dir.mkdir(parents=True)
    metadata = {
        "subject_uid": "child1",
        "subject_age": 5,
        "subject_gender": "Female",
        "organization_name": "OrgA",
    }
    (child_dir / "child1_metadata.json").write_text(json.dumps(metadata))

    mgr = SafeDemographicManager()
    assert mgr.load(str(csv_dir), metadata_roots=[str(raw_dir)])
    demo = demographic_manager.get_child_demographics("child1")
    assert demo["gender"] == "female"
    assert demo["age"] == 5


def test_safe_manager_fallback_logic() -> None:
    safe_mgr = SafeDemographicManager()
    baseline = {"sensitivity": 0.88, "specificity": 0.78}
    enhanced_good = {"sensitivity": 0.89, "specificity": 0.78}
    assert safe_mgr.run_with_fallback(baseline, enhanced_good)

    enhanced_bad = {"sensitivity": 0.80, "specificity": 0.70}
    assert not safe_mgr.run_with_fallback(baseline, enhanced_bad)


