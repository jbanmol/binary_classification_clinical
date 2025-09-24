#!/usr/bin/env python3
"""
Demographic Manager for Clinical ASD Detection
Loads demographics from data/age_data and provides age/gender-aware thresholds
without changing the existing model architecture or bundle format.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

import numpy as np
import pandas as pd


class DemographicManager:
    """Manages demographic data and population-specific clinical thresholds."""

    def __init__(self) -> None:
        self.demographic_data: Dict[str, Dict] = {}

        # Age group definitions (years)
        self.age_thresholds: Dict[str, tuple] = {
            'toddler': (0.0, 2.5),
            'preschool': (2.5, 5.0),
            'school_age': (5.0, 100.0),  # open-ended upper bound
        }

        # Population-specific probability thresholds (initial, tunable)
        self.clinical_thresholds: Dict[str, Dict[str, float] | float] = {
            'toddler': {
                'male': 0.68,
                'female': 0.65,
            },
            'preschool': {
                'male': 0.60,
                'female': 0.57,
            },
            'school_age': {
                'male': 0.53,
                'female': 0.50,
            },
            'default': 0.531,
        }

        # Clinical safety targets (do not compromise)
        self.clinical_targets: Dict[str, float] = {
            'min_sensitivity': 0.86,
            'min_specificity': 0.72,
        }

    # ---------- Public API ----------

    def load_from_dir(self, dir_path: Path) -> bool:
        """Load demographic data from a directory containing CSV/XLSX files.

        Supports flexible column names:
        - UID -> child_id (required)
        - Age, Child Age -> age (optional)
        - Gender, Child Gender -> gender (optional)
        - Child DOB, DOB -> dob (optional)
        """
        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            return False

        frames: List[pd.DataFrame] = []
        for p in sorted(dir_path.iterdir()):
            if p.suffix.lower() in ('.csv', '.xlsx', '.xls'):
                try:
                    df = pd.read_csv(p) if p.suffix.lower() == '.csv' else pd.read_excel(p)
                except Exception:
                    continue
                if df is None or df.empty:
                    continue
                frames.append(self._normalize_demographic_frame(df, dataset_hint=p.stem))

        frames = [f for f in frames if not f.empty]
        if not frames:
            return False

        all_demo = pd.concat(frames, ignore_index=True)
        self._build_lookup(all_demo)
        return True

    def get_child_demographics(self, child_id: str) -> Dict:
        return self.demographic_data.get(str(child_id), {
            'age': None,
            'gender': None,
            'dataset': 'unknown',
            'age_group': 'unknown',
        })

    def get_clinical_threshold(self, child_id: str) -> float:
        info = self.get_child_demographics(child_id)
        age_group = info.get('age_group', 'unknown')
        gender = info.get('gender', 'unknown')

        # Specific group threshold
        group_cfg = self.clinical_thresholds.get(age_group)
        if isinstance(group_cfg, dict):
            thr = group_cfg.get(gender)
            if isinstance(thr, (float, int)):
                return float(thr)

        # Fallback default
        return float(self.clinical_thresholds.get('default', 0.531))

    def save_for_deployment(self, filepath: str = "deployment_demographics.pkl") -> None:
        import pickle
        payload = {
            'demographic_data': self.demographic_data,
            'clinical_thresholds': self.clinical_thresholds,
            'clinical_targets': self.clinical_targets,
            'age_thresholds': self.age_thresholds,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)

    def load_for_deployment(self, filepath: str = "deployment_demographics.pkl") -> bool:
        import pickle
        try:
            with open(filepath, 'rb') as f:
                payload = pickle.load(f)
            self.demographic_data = payload.get('demographic_data', {})
            self.clinical_thresholds = payload.get('clinical_thresholds', self.clinical_thresholds)
            self.clinical_targets = payload.get('clinical_targets', self.clinical_targets)
            self.age_thresholds = payload.get('age_thresholds', self.age_thresholds)
            return True
        except Exception:
            return False

    def validate_clinical_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       age_groups: List[str]) -> Dict[str, Dict[str, float]]:
        from sklearn.metrics import confusion_matrix
        results: Dict[str, Dict[str, float]] = {}
        unique_groups = sorted(set(age_groups))
        for group in unique_groups:
            mask = (np.array(age_groups) == group)
            if int(mask.sum()) < 5:
                continue
            tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            results[group] = {
                'sensitivity': float(sens),
                'specificity': float(spec),
                'n_samples': int(mask.sum()),
            }
        return results

    # ---------- Internal helpers ----------

    def _normalize_demographic_frame(self, df: pd.DataFrame, dataset_hint: str = 'unknown') -> pd.DataFrame:
        cols = {c.lower().strip(): c for c in df.columns}

        def pick(*names: str) -> Optional[str]:
            for n in names:
                key = n.lower().strip()
                if key in cols:
                    return cols[key]
            return None

        uid_col = pick('UID', 'uid', 'child_id', 'Unity_id', 'unity_id')
        if uid_col is None:
            return pd.DataFrame(columns=['child_id', 'age', 'gender', 'dataset'])

        age_col = pick('Age', 'Child Age', 'age', 'child age')
        gen_col = pick('Gender', 'Child Gender', 'gender', 'child gender')
        dob_col = pick('Child DOB', 'DOB', 'child dob', 'dob')

        out = pd.DataFrame()
        out['child_id'] = df[uid_col].astype(str).map(self._normalize_child_id)

        if age_col is not None:
            out['age'] = pd.to_numeric(df[age_col], errors='coerce')
        else:
            out['age'] = np.nan

        if gen_col is not None:
            g = df[gen_col].astype(str).str.lower().str.strip()
            g = g.replace({'m': 'male', 'boy': 'male', 'male': 'male',
                           'f': 'female', 'girl': 'female', 'female': 'female'})
            out['gender'] = g.where(g.isin(['male', 'female']), other='unknown')
        else:
            out['gender'] = 'unknown'

        if out['age'].isna().any() and dob_col is not None:
            try:
                dob_parsed = pd.to_datetime(df[dob_col], errors='coerce')
                now = datetime.now()
                age_years = (now - dob_parsed).dt.days / 365.25
                out.loc[out['age'].isna() & dob_parsed.notna(), 'age'] = age_years
            except Exception:
                pass

        out['dataset'] = dataset_hint
        out['age_group'] = out['age'].apply(self._age_to_group)

        out = out.dropna(subset=['child_id'])
        out = out.drop_duplicates(subset=['child_id'], keep='first')
        return out[['child_id', 'age', 'gender', 'dataset', 'age_group']]

    def _build_lookup(self, df: pd.DataFrame) -> None:
        data: Dict[str, Dict] = {}
        for _, row in df.iterrows():
            data[str(row['child_id'])] = {
                'age': (None if pd.isna(row['age']) else float(row['age'])),
                'gender': (None if pd.isna(row['gender']) else str(row['gender'])),
                'dataset': str(row.get('dataset', 'unknown')),
                'age_group': str(row.get('age_group', 'unknown')),
            }
        self.demographic_data = data

    def _age_to_group(self, age: float) -> str:
        try:
            a = float(age)
        except Exception:
            return 'unknown'
        for group, (lo, hi) in self.age_thresholds.items():
            if lo <= a < hi:
                return group
        return 'school_age'

    def _normalize_child_id(self, cid: str) -> str:
        if cid is None:
            return ''
        s = str(cid).strip()
        # Optionally, customize normalization (e.g., remove spaces/underscores)
        return s


# Global singleton instance
demographic_manager = DemographicManager()


