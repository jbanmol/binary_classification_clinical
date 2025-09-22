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
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List

# Apply OpenMP fix automatically for macOS users
if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import subprocess
import shutil

# Ensure project root is on sys.path so we can import local modules
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# Import existing functionality (do not change the model!)
######################## Stability: thread limits (no algo change) ########################
# On some macOS setups, OpenMP-backed libs (BLAS/UMAP) can segfault. To improve robustness
# without changing any modeling/preprocessing logic, limit thread counts if not already set.
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
from scripts.predict_cli import load_bundle  # best model bundle scoring
from rag_system.research_engine import ColoringDataProcessor  # for unlabeled fallback


def build_child_dataset_unlabeled(raw_folder: Path):
    """Fallback builder: create child-level dataset from raw JSONs without labels.
    Mirrors the aggregation used during training, but does not filter on labels.
    Returns (X_df_aligned_like, child_ids, raw_features_df_before_bins).
    """
    # Walk raw_folder for Coloring_*.json
    json_files = list(raw_folder.rglob('Coloring_*.json'))
    if not json_files:
        raise RuntimeError(f"No Coloring_*.json files found under {raw_folder}")

    print(f"Found {len(json_files)} Coloring_*.json files")
    
    # Use ColoringDataProcessor to extract features
    processor = ColoringDataProcessor()
    all_features = []
    child_ids = []
    
    for json_file in json_files:
        try:
            # Extract child_id from filename (assuming format: Coloring_<child_id>_*.json)
            filename = json_file.name
            child_id = filename.split('_')[1] if '_' in filename else json_file.stem
            
            # Parse the session file
            session_data = processor.parse_session_file(json_file)
            if session_data is not None:
                # Extract behavioral features
                features = processor.extract_behavioral_features(session_data)
                if features is not None:
                    all_features.append(features)
                    child_ids.append(child_id)
                    print(f"  ✅ Processed {filename} -> child_id: {child_id}")
                else:
                    print(f"  ⚠️  Skipped {filename} (no valid features)")
            else:
                print(f"  ⚠️  Skipped {filename} (no valid session data)")
        except Exception as e:
            print(f"  ❌ Error processing {json_file}: {e}")
    
    if not all_features:
        raise RuntimeError("No valid features extracted from any files")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Define the canonical numeric features (same as training)
    NUMERIC_FEATURES_CANON = [
        'velocity_mean', 'velocity_std', 'velocity_max', 'velocity_cv',
        'tremor_indicator', 'acc_magnitude_mean', 'acc_magnitude_std',
        'palm_touch_ratio', 'unique_fingers', 'max_finger_id',
        'session_duration', 'stroke_count', 'total_touch_points',
        'unique_zones', 'unique_colors', 'final_completion',
        'completion_progress_rate', 'avg_time_between_points',
        'canceled_touches'
    ]
    
    # Features intersection
    available = [c for c in NUMERIC_FEATURES_CANON if c in df.columns]
    if not available:
        raise RuntimeError("No expected numeric features present in data")

    print(f"Using {len(available)} available features: {available}")

    # Child aggregation (same as training)
    agg = df.groupby('child_id')[available].mean().reset_index()
    sess_count = df.groupby('child_id').size().rename('session_count').reset_index()
    agg = agg.merge(sess_count, on='child_id', how='left')

    # Engineer domain features (same as training)
    eps = 1e-8
    
    # Per-zone dynamics
    if 'unique_zones' in agg.columns and 'total_touch_points' in agg.columns:
        agg['touches_per_zone'] = agg['total_touch_points'] / (agg['unique_zones'] + eps)
    if 'unique_zones' in agg.columns and 'stroke_count' in agg.columns:
        agg['strokes_per_zone'] = agg['stroke_count'] / (agg['unique_zones'] + eps)
    if 'unique_zones' in agg.columns and 'session_duration' in agg.columns:
        agg['zones_per_minute'] = agg['unique_zones'] / (agg['session_duration'] / 60.0 + eps)
    
    # Velocity and acceleration ratios
    if 'velocity_std' in agg.columns and 'velocity_mean' in agg.columns:
        agg['vel_std_over_mean'] = agg['velocity_std'] / (agg['velocity_mean'] + eps)
    if 'acc_magnitude_std' in agg.columns and 'acc_magnitude_mean' in agg.columns:
        agg['acc_std_over_mean'] = agg['acc_magnitude_std'] / (agg['acc_magnitude_mean'] + eps)
    
    # Inter-point behavior
    if 'avg_time_between_points' in agg.columns and 'session_duration' in agg.columns:
        agg['avg_ibp_norm'] = agg['avg_time_between_points'] / (agg['session_duration'] + eps)
        agg['interpoint_rate'] = 1.0 / (agg['avg_time_between_points'] + eps)
    
    # Touch and stroke rates
    if 'total_touch_points' in agg.columns and 'session_duration' in agg.columns:
        agg['touch_rate'] = agg['total_touch_points'] / (agg['session_duration'] / 60.0 + eps)
    if 'stroke_count' in agg.columns and 'session_duration' in agg.columns:
        agg['stroke_rate'] = agg['stroke_count'] / (agg['session_duration'] / 60.0 + eps)
    
    # Save a copy before quantile bins as "raw" features
    raw_features_df = agg.copy()
    
    # Quantile-based binning for key ratios
    for col in ['touch_rate', 'strokes_per_zone', 'vel_std_over_mean', 'acc_std_over_mean', 'zones_per_minute', 'interpoint_rate']:
        if col in agg.columns:
            q25, q50, q75 = agg[col].quantile([0.25, 0.5, 0.75])
            agg[f'bin_{col}_0'] = (agg[col] <= q25).astype(int)
            agg[f'bin_{col}_1'] = ((agg[col] > q25) & (agg[col] <= q50)).astype(int)
            agg[f'bin_{col}_2'] = ((agg[col] > q50) & (agg[col] <= q75)).astype(int)
            agg[f'bin_{col}_3'] = (agg[col] > q75).astype(int)
    
    # Extract child_ids for return
    child_ids = agg['child_id'].tolist()
    
    # Remove child_id from features
    X_df = agg.drop('child_id', axis=1)
    
    print(f"Built dataset with {len(X_df)} samples and {len(X_df.columns)} features")
    return X_df, child_ids, raw_features_df


def find_raw_data_folders() -> List[Path]:
    """Find all raw data folders under ./data/raw/"""
    data_raw = Path("data/raw")
    if not data_raw.exists():
        return []
    
    folders = [f for f in data_raw.iterdir() if f.is_dir()]
    return sorted(folders)


def interactive_folder_selection() -> Path:
    """Interactive folder selection with fallback to manual input"""
    folders = find_raw_data_folders()
    
    if not folders:
        print("No raw data folders found under ./data/raw/")
        print("Please provide the path to your raw data folder:")
        while True:
            path_str = input("Path: ").strip()
            if not path_str:
                continue
            path = Path(path_str)
            if path.exists() and path.is_dir():
                return path
            print(f"❌ Path does not exist or is not a directory: {path}")
    else:
        print("Available raw data folders:")
        for i, folder in enumerate(folders, 1):
            print(f"  {i}. {folder.name}")
        print(f"  {len(folders)+1}. Enter custom path")

    while True:
        try:
            choice = input(f"Select folder (1-{len(folders)+1}): ").strip()
            if not choice:
                continue
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(folders):
                return folders[choice_num - 1]
            elif choice_num == len(folders) + 1:
                # Custom path
                path_str = input("Enter custom path: ").strip()
                if path_str:
                    path = Path(path_str)
                    if path.exists() and path.is_dir():
                        return path
                    print(f"❌ Path does not exist or is not a directory: {path}")
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(folders)+1}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="End-to-end prediction from raw data")
    parser.add_argument("--raw", type=str, help="Path to raw data folder")
    parser.add_argument("--out", type=str, help="Output CSV file path")
    parser.add_argument("--bundle", type=str, default="models/final_np_iqrmid_u16n50_k2/bundle.json", 
                       help="Path to model bundle")
    
    args = parser.parse_args()
    
    # Determine raw data folder
    if args.raw:
        raw_folder = Path(args.raw)
        if not raw_folder.exists():
            print(f"❌ Raw data folder does not exist: {raw_folder}")
            sys.exit(1)
    else:
        raw_folder = interactive_folder_selection()
    
    print(f"📁 Using raw data folder: {raw_folder}")
    
    # Determine output file
    if args.out:
        output_file = Path(args.out)
    else:
        experiments_dir = PROJ_ROOT / 'data_experiments'
        experiments_dir.mkdir(parents=True, exist_ok=True)
        output_file = experiments_dir / f"{raw_folder.name}_results.csv"
    
    print(f"💾 Output will be saved to: {output_file}")
    
    # Load model bundle
    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        print(f"❌ Model bundle not found: {bundle_path}")
        sys.exit(1)

    print(f"🤖 Loading model bundle: {bundle_path}")
    bundle, bundle_root = load_bundle(bundle_path)

    # Precompute feature file paths
    results_dir = PROJ_ROOT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = results_dir / f"{raw_folder.name}_features_raw.csv"
    aligned_csv_path = results_dir / f"{raw_folder.name}_features_aligned.csv"
    
    try:
        # Try to build dataset with labels first (for validation)
        print("🔍 Attempting to build dataset with labels...")
        try:
            X, y, child_ids = build_child_dataset()
            print(f"✅ Built labeled dataset: {len(X)} samples")
            has_labels = True
        except Exception as e:
            print(f"⚠️  Could not build labeled dataset: {e}")
            print("🔄 Falling back to unlabeled dataset...")
            has_labels = False
        
        aligned_for_inference: Path
        
        if aligned_csv_path.exists():
            # Use existing aligned features exactly as-is to preserve prior outputs
            print(f"📄 Using existing aligned features: {aligned_csv_path}")
            aligned_for_inference = aligned_csv_path
            # Do not regenerate raw/aligned to avoid drift
            y = None if not has_labels else y
        else:
            # Build unlabeled dataset and write raw + aligned features
            X, child_ids = build_child_dataset_unlabeled(raw_folder)[:2]
            _, _, raw_features_df = build_child_dataset_unlabeled(raw_folder)
            # Write raw features
            # Ensure child_id column present in raw
            if 'child_id' not in raw_features_df.columns:
                raw_features_df.insert(0, 'child_id', child_ids)
            raw_features_df.to_csv(raw_csv_path, index=False)
            
            # Align features to bundle schema
            aligned_df = raw_features_df.copy()
            # Drop child_id for alignment
            aligned_no_id = aligned_df.drop(columns=['child_id'])
            expected = bundle['feature_columns']
            # Add missing with zeros
            for col in expected:
                if col not in aligned_no_id.columns:
                    aligned_no_id[col] = 0
            # Drop extras
            extra = [c for c in aligned_no_id.columns if c not in expected]
            if extra:
                aligned_no_id = aligned_no_id.drop(columns=extra)
            # Reorder
            aligned_no_id = aligned_no_id[expected]
            # Reattach child_id
            aligned_final = pd.concat([aligned_df[['child_id']], aligned_no_id], axis=1)
            aligned_final.to_csv(aligned_csv_path, index=False)
            aligned_for_inference = aligned_csv_path
            y = None
        
        # Make predictions via isolated subprocess using aligned features
        print("🔮 Making predictions...")
        temp_predictions_csv = Path("temp_predictions.csv")
        try:
            # Prepare environment for subprocess
            env = os.environ.copy()
            env.setdefault('OMP_NUM_THREADS', '1')
            env.setdefault('OMP_MAX_ACTIVE_LEVELS', '1')
            env.setdefault('OPENBLAS_NUM_THREADS', '1')
            env.setdefault('MKL_NUM_THREADS', '1')
            env.setdefault('VECLIB_MAXIMUM_THREADS', '1')
            env.setdefault('NUMEXPR_NUM_THREADS', '1')

            python_exe = sys.executable
            cmd = [
                python_exe,
                str(PROJ_ROOT / 'scripts' / 'predict_cli.py'),
                '--bundle', str(bundle_path),
                '--in', str(aligned_for_inference),
                '--out', str(temp_predictions_csv)
            ]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stdout)
                print(result.stderr)
                raise RuntimeError(f"predict_cli failed with code {result.returncode}")

            # Load predictions and write final CSV in prior format
            pred_full = pd.read_csv(temp_predictions_csv)
            cols = [c for c in ['child_id', 'prob_asd', 'pred_label'] if c in pred_full.columns]
            if len(cols) < 3:
                raise RuntimeError("Prediction output missing required columns 'child_id', 'prob_asd', 'pred_label'")
            final_df = pred_full[['child_id', 'prob_asd', 'pred_label']]
        finally:
            if temp_predictions_csv.exists():
                try:
                    temp_predictions_csv.unlink()
                except Exception:
                    pass
        
        # Save results (exact prior format)
        final_df.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"  - Samples processed: {len(final_df)}")
        print(f"  - Predictions made: {len(final_df)}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()