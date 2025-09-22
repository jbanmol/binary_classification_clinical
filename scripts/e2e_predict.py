#!/usr/bin/env python3
"""
Minimal interactive CLI: select raw data folder -> run end-to-end prediction

What it does (single command):
- Lists raw data folders under ./data
- Lets you pick one (or type a path)
- Builds child-level features using the existing pipeline (no model changes)
- Uses the best trained bundle with predict_cli to generate predictions
- Saves <raw_folder_name>_results.csv in the project root (or current working directory)

Dependencies: none beyond the project requirements.
"""
import os
import sys
import json
from pathlib import Path
from typing import List

import pandas as pd

# Ensure project root is on sys.path so we can import local modules
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# Import existing functionality (do not change the model!)
######################## Stability: thread limits (no algo change) ########################
# On some macOS setups, OpenMP-backed libs (BLAS/UMAP) can segfault. To improve robustness
# without changing any modeling/preprocessing logic, limit thread counts if not already set.
# Also suppress OpenMP deprecation warnings on macOS.
try:
    from src.openmp_fix import apply_openmp_fix
    apply_openmp_fix()
except ImportError:
    # Fallback to original thread limiting if openmp_fix is not available
    for _k, _v in (
        ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ):
        os.environ.setdefault(_k, _v)
###########################################################################################
from scripts.bagged_end_to_end import build_child_dataset  # feature construction from raw (labeled)
from scripts.predict_cli import load_bundle, predict  # best model bundle scoring
from rag_system.research_engine import ColoringDataProcessor  # for unlabeled fallback


def build_child_dataset_unlabeled(raw_folder: Path):
    """Fallback builder: create child-level dataset from raw JSONs without labels.
    Mirrors the aggregation used during training, but does not filter on labels.
    Returns (X_df, child_ids).
    """
    # Walk raw_folder for Coloring_*.json
    json_files = list(raw_folder.rglob('Coloring_*.json'))
    if not json_files:
        raise RuntimeError(f"No Coloring_*.json files found under {raw_folder}")

    proc = ColoringDataProcessor()  # will try to load labels; OK if missing
    rows = []
    for fp in json_files:
        ses = proc.parse_session_file(fp)
        if not ses:
            continue
        feats = proc.extract_behavioral_features(ses)
        if feats:
            rows.append(feats)

    if not rows:
        raise RuntimeError("Parsed zero sessions from raw JSONs")

    import numpy as np
    import pandas as pd

    df = pd.DataFrame(rows)
    # Basic expected numeric columns (subset will be present depending on data)
    NUMERIC_FEATURES_CANON = [
        'velocity_mean', 'velocity_std', 'velocity_max', 'velocity_cv',
        'tremor_indicator', 'acc_magnitude_mean', 'acc_magnitude_std',
        'palm_touch_ratio', 'unique_fingers', 'max_finger_id',
        'session_duration', 'stroke_count', 'total_touch_points',
        'unique_zones', 'unique_colors', 'final_completion',
        'completion_progress_rate', 'avg_time_between_points',
        'canceled_touches'
    ]
    available = [c for c in NUMERIC_FEATURES_CANON if c in df.columns]
    if not available:
        raise RuntimeError("No expected numeric features present in parsed data")

    # Aggregate to child-level
    agg = df.groupby('child_id')[available].mean().reset_index()
    sess_count = df.groupby('child_id').size().rename('session_count').reset_index()
    agg = agg.merge(sess_count, on='child_id', how='left')

    # Domain features (same as training-time builder)
    eps = 1e-8
    if 'unique_zones' in agg.columns and 'total_touch_points' in agg.columns:
        agg['touches_per_zone'] = agg['total_touch_points'] / (agg['unique_zones'] + eps)
    if 'unique_zones' in agg.columns and 'stroke_count' in agg.columns:
        agg['strokes_per_zone'] = agg['stroke_count'] / (agg['unique_zones'] + eps)
    if 'unique_zones' in agg.columns and 'session_duration' in agg.columns:
        agg['zones_per_minute'] = agg['unique_zones'] / (agg['session_duration'] / 60.0 + eps)
    if 'velocity_mean' in agg.columns and 'velocity_std' in agg.columns:
        agg['vel_std_over_mean'] = agg['velocity_std'] / (agg['velocity_mean'] + eps)
    if 'acc_magnitude_mean' in agg.columns and 'acc_magnitude_std' in agg.columns:
        agg['acc_std_over_mean'] = agg['acc_magnitude_std'] / (agg['acc_magnitude_mean'] + eps)
    if 'avg_time_between_points' in agg.columns and 'session_duration' in agg.columns:
        agg['avg_ibp_norm'] = agg['avg_time_between_points'] / (agg['session_duration'] + eps)
        agg['interpoint_rate'] = (agg['session_duration'] + eps) / (agg['avg_time_between_points'] + eps)
    if 'total_touch_points' in agg.columns and 'session_duration' in agg.columns:
        agg['touch_rate'] = agg['total_touch_points'] / (agg['session_duration'] + eps)
    if 'stroke_count' in agg.columns and 'session_duration' in agg.columns:
        agg['stroke_rate'] = agg['stroke_count'] / (agg['session_duration'] + eps)

    # Quantile bin flags
    def _add_bin_flags(df_in: pd.DataFrame, col: str, q: int = 4) -> pd.DataFrame:
        if col not in df_in.columns:
            return df_in
        s = df_in[col]
        try:
            binned = pd.qcut(s.rank(method='first'), q=q, labels=False, duplicates='drop')
        except Exception:
            return df_in
        dummies = pd.get_dummies(binned, prefix=f"bin_{col}")
        return pd.concat([df_in, dummies], axis=1)

    for ratio_col in ['touch_rate', 'strokes_per_zone', 'vel_std_over_mean', 'acc_std_over_mean', 'zones_per_minute', 'interpoint_rate']:
        agg = _add_bin_flags(agg, ratio_col, q=4)

    X = agg.drop(columns=['child_id'], errors='ignore').copy()
    child_ids = agg['child_id'].astype(str).tolist()
    return X, child_ids


def list_raw_folders(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    # Show only directories that contain at least one Coloring_*.json somewhere beneath
    candidates = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir():
            # Heuristic: accept directory; deeper validation will happen in build_child_dataset
            candidates.append(p)
    return candidates


def pick_raw_folder_interactive() -> Path:
    data_dir = PROJ_ROOT / 'data'
    options = list_raw_folders(data_dir)

    print("\nSelect a raw data folder to score:")
    if not options:
        print(f"No subfolders found in {data_dir}. You can type a path manually.")
    else:
        for i, p in enumerate(options, 1):
            print(f"  {i}) {p}")
    print("  0) Enter a path manually")

    while True:
        choice = input("Enter number (or 0 to type a path): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if idx == 0:
                raw_path_str = input("Enter full or relative path to raw data folder: ").strip()
                raw_path = Path(raw_path_str).expanduser().resolve()
                if raw_path.exists() and raw_path.is_dir():
                    return raw_path
                print("Path not found or not a directory. Try again.\n")
                continue
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid selection. Try again.\n")


def main():
    # Optional: allow non-interactive usage via --raw PATH
    raw_arg = None
    for i, a in enumerate(sys.argv):
        if a == '--raw' and i + 1 < len(sys.argv):
            raw_arg = Path(sys.argv[i + 1]).expanduser().resolve()
        if a in ('-h', '--help'):
            print("Usage: python scripts/e2e_predict.py [--raw /path/to/raw_data]")
            return

    if raw_arg is not None:
        raw_folder = raw_arg
        if not raw_folder.exists() or not raw_folder.is_dir():
            print(f"Provided --raw path is invalid: {raw_folder}")
            sys.exit(1)
    else:
        raw_folder = pick_raw_folder_interactive()

    # Point the ingestion to the selected raw folder without editing code:
    # research_engine reads RAW_DATA_PATH env var from rag_system.config
    os.environ['RAW_DATA_PATH'] = str(raw_folder)

    # Pre-run: count raw session files
    json_files = list(raw_folder.rglob('Coloring_*.json'))
    print(f"\nFound {len(json_files)} Coloring_*.json files under: {raw_folder}")

    # Build child-level features from raw JSONs (uses existing pipeline)
    print("Building child-level features from raw data...")
    try:
        X_df, y_arr, child_ids = build_child_dataset()
    except Exception as e:
        print(f"Labeled feature build failed (falling back to unlabeled mode): {e}")
        try:
            X_df, child_ids = build_child_dataset_unlabeled(raw_folder)
        except Exception as e2:
            print(f"Unlabeled feature build failed: {e2}")
            sys.exit(1)

    # Post-build: report unique children aggregated
    num_children = len(child_ids) if isinstance(child_ids, list) else 0
    print(f"Aggregated {num_children} unique children.")

    # Load the best model bundle (do not modify the model)
    bundle_path = PROJ_ROOT / 'models' / 'final_np_iqrmid_u16n50_k2' / 'bundle.json'
    if not bundle_path.exists():
        print(f"Best model bundle not found at {bundle_path}. Aborting.")
        sys.exit(1)

    bundle, bundle_root = load_bundle(bundle_path)
    feature_cols = bundle.get('feature_columns', [])
    if not feature_cols:
        print("Bundle missing 'feature_columns'. Aborting.")
        sys.exit(1)

    # Align columns to what the bundle expects
    X_aligned = X_df.reindex(columns=feature_cols, fill_value=0).copy()

    # Persist features for inspection and reproducibility
    results_dir = PROJ_ROOT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_feat_csv = results_dir / f"{raw_folder.name}_features_raw.csv"
    aligned_feat_csv = results_dir / f"{raw_folder.name}_features_aligned.csv"
    try:
        df_raw_save = X_df.copy()
        df_raw_save.insert(0, 'child_id', child_ids)
        df_raw_save.to_csv(raw_feat_csv, index=False)
    except Exception:
        pass
    df_aligned_save = X_aligned.copy()
    df_aligned_save.insert(0, 'child_id', child_ids)
    df_aligned_save.to_csv(aligned_feat_csv, index=False)
    print(f"Saved raw features: {raw_feat_csv}")
    print(f"Saved aligned features: {aligned_feat_csv}")

    # Run predictions using the existing predict() from predict_cli
    print("Running predictions with the best model bundle...")

    # predict() expects a CSV path; use the aligned features we just saved
    bundle, bundle_root = load_bundle(bundle_path)
    out_df = predict(bundle, bundle_root, aligned_feat_csv)

    # Keep production-ready columns when available
    cols = list(out_df.columns)
    id_col = 'child_id' if 'child_id' in cols else (cols[0] if cols else None)
    core_cols = [c for c in ['prob_asd', 'pred_label'] if c in cols]
    if id_col and core_cols:
        out_df = out_df[[id_col] + core_cols]

    # Name output file
    out_name = f"{raw_folder.name}_results.csv"
    out_path = PROJ_ROOT / out_name
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved predictions to {out_path}")


if __name__ == '__main__':
    main()


