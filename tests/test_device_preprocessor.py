"""Tests for DeviceRobustPreprocessor."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.domain_adaptation import DeviceRobustPreprocessor


def make_training_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "feature_a": rng.normal(0, 1, size=200),
        "feature_b": rng.normal(5, 2, size=200),
        "categorical": ["x"] * 200,
    }
    return pd.DataFrame(data)


def make_target_df(shift: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {
        "feature_a": rng.normal(shift, 1, size=100),
        "feature_b": rng.normal(5 + shift, 2, size=100),
        "categorical": ["y"] * 100,
    }
    return pd.DataFrame(data)


def test_adapt_blends_toward_training_distribution() -> None:
    trainer = DeviceRobustPreprocessor(adaptation_strength=0.4)
    train_df = make_training_df()
    trainer.fit_training_distribution(train_df)
    target_df = make_target_df()

    adapted = trainer.adapt(target_df)

    assert adapted["feature_a"].mean() < target_df["feature_a"].mean()
    assert adapted["feature_a"].mean() > train_df["feature_a"].mean() - 1.0


def test_adapt_bypass_when_no_shift() -> None:
    trainer = DeviceRobustPreprocessor(adaptation_strength=0.4)
    train_df = make_training_df()
    trainer.fit_training_distribution(train_df)
    target_df = make_training_df()

    adapted = trainer.adapt(target_df)

    pd.testing.assert_frame_equal(target_df, adapted)


def test_to_dict_roundtrip() -> None:
    trainer = DeviceRobustPreprocessor()
    train_df = make_training_df()
    trainer.fit_training_distribution(train_df)

    payload = trainer.to_dict()
    new_trainer = DeviceRobustPreprocessor()
    new_trainer.from_dict(payload)

    assert new_trainer.training_stats is not None
    assert set(new_trainer.training_stats.means.index) == {"feature_a", "feature_b"}
