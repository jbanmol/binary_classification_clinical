#!/usr/bin/env python3
"""
Predict CLI for ASD/TD model bundle.
Usage:
  ./venv/bin/python scripts/predict_cli.py --bundle models/final_np_iqrmid_u16n50_k2/bundle.json --in data/child_level_features.csv --out predictions.csv
"""
import argparse, json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Stability: limit threads on macOS to avoid OpenMP segfaults (no algorithm change)
# Also suppress OpenMP deprecation warnings on macOS.
try:
    from src.openmp_fix import apply_openmp_fix
    apply_openmp_fix()
except ImportError:
    # Fallback to original thread limiting if openmp_fix is not available
    for _k, _v in (
        ("OMP_NUM_THREADS", "1"),
        ("OMP_MAX_ACTIVE_LEVELS", "1"),  # This fixes the deprecation warning
        ("OPENBLAS_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ):
        os.environ.setdefault(_k, _v)


def load_bundle(bundle_path: Path):
    b = json.load(open(bundle_path))
    root = bundle_path.parent
    return b, root


def _load_training_reference(bundle_root: Path):
    try:
        ref_path = bundle_root / 'training_reference.csv'
        if ref_path.exists():
            return pd.read_csv(ref_path)
    except Exception:
        pass
    return None


def predict(bundle, root, in_csv: Path, demographics_path: Path | None = None) -> pd.DataFrame:
    cols = bundle['feature_columns']
    df = pd.read_csv(in_csv)
    X = df[cols].copy()

    # Optional domain adaptation using exported training reference
    train_ref = _load_training_reference(root)
    if train_ref is not None:
        try:
            # Import lazily to avoid mandatory dependency
            from src.domain_adaptation import domain_detector, domain_adapter  # type: ignore
            analysis = domain_detector.comprehensive_shift_analysis(train_ref, X)
            if analysis.get('overall_shift_severity') in ('MODERATE', 'SEVERE'):
                common = list(set(train_ref.columns) & set(X.columns))
                if len(common) > 5:
                    T_vals = train_ref[common].fillna(0).values
                    X_vals = X[common].fillna(0).values
                    X_adapted, _ = domain_adapter.coral_alignment(T_vals, X_vals)
                    X = X.copy()
                    X[common] = X_adapted
        except Exception:
            pass

    # Preprocess
    scaler = joblib.load(root/'preprocess'/'scaler_all.joblib')
    X_s = scaler.transform(X)
    try:
        umap = joblib.load(root/'preprocess'/'umap_all.joblib')
        U = umap.transform(X_s)
        X_s = np.concatenate([X_s, U], axis=1)
    except Exception:
        pass

    models_used = bundle['models_used']
    probs = []
    for name in models_used:
        mdl = joblib.load(root/'models'/f'{name}.joblib')
        if hasattr(mdl, 'predict_proba'):
            p = mdl.predict_proba(X_s)[:,1]
        elif hasattr(mdl, 'decision_function'):
            s = mdl.decision_function(X_s)
            p = (s - s.min())/(s.max()-s.min()+1e-8)
        else:
            p = np.full(len(X_s), 0.5)
        probs.append(p)

    def mean_of(names):
        arr = [probs[models_used.index(n)] for n in names if n in models_used]
        return np.mean(np.vstack(arr), axis=0)

    E_sens = mean_of(bundle['ensembles']['e_sens_names'])
    E_spec = mean_of(bundle['ensembles']['e_spec_names'])
    comb = bundle['combiner']
    if comb['type'] == 'alpha':
        alpha = float(comb['best_alpha'])
        P_ens = alpha*E_sens + (1-alpha)*E_spec
    else:
        Z = np.column_stack([E_sens, E_spec, np.abs(E_sens - E_spec)])
        meta = joblib.load(root/'models'/'meta_combiner.joblib')
        P_ens = meta.predict_proba(Z)[:,1]

    if comb.get('temperature_applied') and 'temperature_T' in comb:
        T = float(comb['temperature_T'])
        P_ens = 1/(1+np.exp(-np.log(P_ens/(1-P_ens+1e-12))/T))

    # Default labels via bundle threshold
    tau = float(bundle['threshold'])
    labels = (P_ens >= tau).astype(int)

    # Optional demographic-aware overrides per child
    if demographics_path is not None and 'child_id' in df.columns:
        try:
            from src.demographics.demographic_manager import demographic_manager  # type: ignore
            base = demographics_path
            base_dir = base if base.is_dir() else base.parent
            if demographic_manager.load_from_dir(base_dir):
                labels_d = np.zeros_like(labels)
                for i, cid in enumerate(df['child_id'].astype(str).tolist()):
                    # Never go below bundle default threshold to avoid increasing FPR
                    thr_demo = float(demographic_manager.get_clinical_threshold(cid))
                    thr_i = max(thr_demo, tau)
                    labels_d[i] = 1 if P_ens[i] >= thr_i else 0
                labels = labels_d
        except Exception:
            pass

    out = df.copy()
    out['prob_asd'] = P_ens
    out['pred_label'] = labels
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True, help='Path to bundle.json')
    ap.add_argument('--in', dest='in_csv', required=True, help='Input CSV with child-level features')
    ap.add_argument('--out', required=True, help='Where to write predictions CSV')
    ap.add_argument('--demographics', default=None, help='Optional path to demographics dir or file')
    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    bundle, root = load_bundle(bundle_path)
    demo_path = Path(args.demographics) if args.demographics else None
    if demo_path is None and os.environ.get('DEMOGRAPHICS_PATH'):
        demo_path = Path(os.environ['DEMOGRAPHICS_PATH'])
    out_df = predict(bundle, root, Path(args.in_csv), demo_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")

if __name__ == '__main__':
    main()
