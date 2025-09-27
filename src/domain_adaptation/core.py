"""Core domain adaptation utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.covariance import EmpiricalCovariance


class DomainShiftDetector:
    """Detect and quantify domain shift between datasets."""

    def __init__(self) -> None:
        self.shift_thresholds = {
            "ks_statistic": 0.2,
            "wasserstein": 0.5,
            "mean_shift": 1.5,
        }

    def detect_feature_shifts(
        self, source_data: pd.DataFrame, target_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        shift_results: Dict[str, Dict[str, float]] = {}
        common_features = list(set(source_data.columns) & set(target_data.columns))
        for feature in common_features:
            s = source_data[feature].dropna()
            t = target_data[feature].dropna()
            if len(s) < 10 or len(t) < 10:
                continue
            ks_stat, ks_p = ks_2samp(s, t)
            w_dist = float(wasserstein_distance(s, t))
            pooled_std = float(np.sqrt((s.var() + t.var()) / 2.0))
            mean_shift = float(abs(s.mean() - t.mean()) / (pooled_std + 1e-8))
            shift_results[feature] = {
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "wasserstein_distance": w_dist,
                "mean_shift_std": mean_shift,
            }
        return shift_results

    def detect_covariance_shift(
        self, source_data: np.ndarray, target_data: np.ndarray
    ) -> Dict[str, float]:
        source_cov = EmpiricalCovariance().fit(source_data).covariance_
        target_cov = EmpiricalCovariance().fit(target_data).covariance_
        cov_diff_norm = float(np.linalg.norm(source_cov - target_cov, ord="fro"))
        source_norm = float(np.linalg.norm(source_cov, ord="fro"))
        rel = cov_diff_norm / (source_norm + 1e-8)
        return {
            "covariance_difference_norm": cov_diff_norm,
            "relative_covariance_shift": float(rel),
        }

    def comprehensive_shift_analysis(
        self, source_data: pd.DataFrame, target_data: pd.DataFrame
    ) -> Dict[str, Any]:
        feat = self.detect_feature_shifts(source_data, target_data)
        numeric = list(
            set(source_data.select_dtypes(include=[np.number]).columns)
            & set(target_data.select_dtypes(include=[np.number]).columns)
        )
        cov = {"covariance_difference_norm": 0.0, "relative_covariance_shift": 0.0}
        if len(numeric) > 1:
            cov = self.detect_covariance_shift(
                source_data[numeric].fillna(0).values,
                target_data[numeric].fillna(0).values,
            )

        ks_over = sum(1 for v in feat.values() if v["ks_statistic"] > 0.4)
        w_over = sum(1 for v in feat.values() if v["wasserstein_distance"] > 1.0)
        m_over = sum(1 for v in feat.values() if v["mean_shift_std"] > 3.0)
        severe = (ks_over + w_over + m_over) >= 2
        moderate = (ks_over + w_over + m_over) == 1 or any(
            v["ks_statistic"] > 0.2
            or v["wasserstein_distance"] > 0.5
            or v["mean_shift_std"] > 1.5
            for v in feat.values()
        )
        overall = "MINIMAL"
        if severe:
            overall = "SEVERE"
        elif moderate:
            overall = "MODERATE"

        return {
            "feature_shifts": feat,
            "covariance_shift": cov,
            "overall_shift_severity": overall,
        }


class DomainAdapter:
    """Adapt models for domain shift."""

    def coral_alignment(
        self, source_features: np.ndarray, target_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        Cs = np.cov(source_features.T) + np.eye(source_features.shape[1]) * 1e-6
        Ct = np.cov(target_features.T) + np.eye(target_features.shape[1]) * 1e-6
        es, Us = np.linalg.eigh(Cs)
        et, Ut = np.linalg.eigh(Ct)
        es = np.maximum(es, 1e-6)
        et = np.maximum(et, 1e-6)
        Ws = Us @ np.diag(es ** -0.5) @ Us.T
        Wt = Ut @ np.diag(et ** 0.5) @ Ut.T
        A = Ws @ Wt
        source_adapted = source_features @ A.T
        return source_adapted, target_features


domain_detector = DomainShiftDetector()
domain_adapter = DomainAdapter()
