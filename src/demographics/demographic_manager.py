#!/usr/bin/env python3
"""
Demographic Manager for Clinical ASD Detection
Loads demographics from data/age_data and provides age/gender-aware thresholds
without changing the existing model architecture or bundle format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemographicRecord:
    """Normalized demographic attributes for a single child."""

    child_id: str
    age: Optional[float]
    gender: Optional[str]
    dataset: str
    age_group: str
    source: str


class DemographicManager:
    """Manages demographic data and population-specific clinical thresholds."""

    def __init__(self) -> None:
        self.demographic_data: Dict[str, DemographicRecord] = {}

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

    def load_from_dir(self, dir_path: Path, metadata_roots: Optional[Iterable[Path]] = None) -> bool:
        """Load demographic data from CSV/XLS(X) files and metadata JSONs.

        Args:
            dir_path: Directory containing demographic CSV/XLS files.
            metadata_roots: Optional collection of raw-data directories to scan for
                per-child `*_metadata.json` files.

        Returns:
            True if any demographic records were loaded, False otherwise.
        """

        dir_path = Path(dir_path)
        frames: List[pd.DataFrame] = []

        if dir_path.exists() and dir_path.is_dir():
            for p in sorted(dir_path.iterdir()):
                if p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
                    try:
                        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
                    except Exception:
                        continue
                    if df is None or df.empty:
                        continue
                    frames.append(
                        self._normalize_demographic_frame(
                            df,
                            dataset_hint=p.stem,
                            source=str(p.name),
                            priority=1,
                        )
                    )

        metadata_records = list(self._load_metadata_records(metadata_roots))
        frames.extend(metadata_records)

        frames = [f for f in frames if not f.empty]
        if not frames:
            return False

        all_demo = pd.concat(frames, ignore_index=True)
        self._build_lookup(all_demo)
        return True

    def get_child_demographics(self, child_id: str) -> Dict:
        record = self.demographic_data.get(str(child_id))
        if record is None:
            return {
                "age": None,
                "gender": None,
                "dataset": "unknown",
                "age_group": "unknown",
                "source": "unknown",
            }
        return {
            "age": record.age,
            "gender": record.gender,
            "dataset": record.dataset,
            "age_group": record.age_group,
            "source": record.source,
        }

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

    def _normalize_demographic_frame(
        self,
        df: pd.DataFrame,
        dataset_hint: str = "unknown",
        source: str = "unknown",
        priority: int = 1,
    ) -> pd.DataFrame:
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
        out['source'] = source
        out['priority'] = int(priority)

        out = out.dropna(subset=['child_id'])
        out = out.drop_duplicates(subset=['child_id', 'source'], keep='first')
        return out[['child_id', 'age', 'gender', 'dataset', 'age_group', 'source', 'priority']]

    def _build_lookup(self, df: pd.DataFrame) -> None:
        if 'priority' not in df.columns:
            df['priority'] = 1

        df_sorted = (
            df.sort_values(by=['child_id', 'priority'], ascending=[True, False])
            .drop_duplicates(subset=['child_id'], keep='first')
        )

        data: Dict[str, DemographicRecord] = {}
        for _, row in df_sorted.iterrows():
            child_id = str(row['child_id'])
            age_val: Optional[float]
            try:
                age_val = float(row['age']) if not pd.isna(row['age']) else None
            except Exception:
                age_val = None

            gender_val: Optional[str]
            if pd.isna(row['gender']):
                gender_val = None
            else:
                gender_clean = str(row['gender']).strip().lower()
                gender_val = gender_clean if gender_clean else None
            data[child_id] = DemographicRecord(
                child_id=child_id,
                age=age_val,
                gender=gender_val,
                dataset=str(row.get('dataset', 'unknown')),
                age_group=str(row.get('age_group', 'unknown')),
                source=str(row.get('source', 'unknown')),
            )
        self.demographic_data = data

    def _load_metadata_records(self, metadata_roots: Optional[Iterable[Path]]) -> Iterable[pd.DataFrame]:
        roots: List[Path]
        if metadata_roots is None:
            roots = [Path('data/raw')]
        else:
            roots = [Path(r) for r in metadata_roots]

        frames: List[pd.DataFrame] = []
        for root in roots:
            root = root.expanduser().resolve()
            if not root.exists():
                continue

            rows: List[Dict[str, Any]] = []
            for meta_path in root.rglob('*metadata.json'):
                try:
                    with meta_path.open('r') as f:
                        meta = json.load(f)
                except Exception:
                    continue

                child_raw = meta.get('subject_uid') or meta.get('child_id') or meta.get('uid')
                if not child_raw:
                    child_raw = meta_path.parent.name
                child_id = self._normalize_child_id(str(child_raw))
                if not child_id:
                    continue

                age_value = self._coerce_age(meta.get('subject_age') or meta.get('age'))
                if age_value is None:
                    age_value = self._age_from_dob(meta.get('subject_dob') or meta.get('dob'))

                gender_value = self._normalize_gender(meta.get('subject_gender') or meta.get('gender'))

                dataset_hint = str(meta.get('organization_name') or meta.get('dataset') or self._infer_dataset(root, meta_path))

                rows.append(
                    {
                        'child_id': child_id,
                        'age': age_value,
                        'gender': gender_value,
                        'dataset': dataset_hint,
                        'age_group': self._age_to_group(age_value) if age_value is not None else 'unknown',
                        'source': f'metadata:{meta_path.name}',
                        'priority': 2,
                    }
                )

            if rows:
                frames.append(pd.DataFrame(rows))

        return frames

    def _coerce_age(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str):
            if not value.strip():
                return None
        try:
            return float(value)
        except Exception:
            return None

    def _age_from_dob(self, dob: Any) -> Optional[float]:
        if dob in (None, ''):
            return None
        try:
            dob_series = pd.to_datetime([dob], errors='coerce')
        except Exception:
            return None
        if dob_series.isna().all():
            return None
        now = datetime.now()
        dob_val = dob_series.iloc[0]
        if pd.isna(dob_val):
            return None
        return float((now - dob_val).days / 365.25)

    def _normalize_gender(self, value: Any) -> str:
        if value is None:
            return 'unknown'
        label = str(value).strip().lower()
        if label in {'m', 'male', 'boy'}:
            return 'male'
        if label in {'f', 'female', 'girl'}:
            return 'female'
        return 'unknown'

    def _infer_dataset(self, root: Path, meta_path: Path) -> str:
        try:
            rel = meta_path.relative_to(root)
            parts = rel.parts
            if len(parts) >= 2:
                return parts[0]
            if parts:
                return parts[0]
        except Exception:
            pass
        return root.name

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


