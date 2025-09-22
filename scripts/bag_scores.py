#!/usr/bin/env python3
"""
Bag and evaluate holdout probabilities across seeds for a fixed holdout split.

Usage:
  ./venv/bin/python scripts/bag_scores.py --pattern 'results/waveA_s*_ho777_*_u16n50_k2.json' --out results/bagged_waveA_ho777_u16n50_k2.json
"""
import argparse
import glob
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

def load_preds(pattern: str):
    files = sorted(glob.glob(pattern))
    runs = []
    for p in files:
        with open(p) as f:
            o = json.load(f)
        if not all(k in o for k in ['holdout_ids','holdout_y','holdout_proba']):
            continue
        runs.append((o['holdout_ids'], np.array(o['holdout_y']), np.array(o['holdout_proba']), p))
    return runs


def bag_and_eval(runs, target_sens: float, target_spec: float):
    # Verify same holdout ordering
    ids0 = runs[0][0]
    ys = runs[0][1]
    probs = [r[2] for r in runs]
    P = np.mean(np.vstack(probs), axis=0)
    # Choose threshold with spec-first (or Youden fallback) on holdout (diagnostic)
    fpr, tpr, thr = roc_curve(ys, P)
    spec = 1 - fpr
    idx = np.where((tpr >= target_sens) & (spec >= target_spec))[0]
    if len(idx) > 0:
        gains = (tpr[idx]-target_sens) + (spec[idx]-target_spec)
        best = int(idx[np.argmax(gains)])
    else:
        idx = np.where(spec >= target_spec)[0]
        if len(idx) > 0:
            best = int(idx[np.argmax(tpr[idx])])
        else:
            youden = tpr - fpr
            best = int(np.argmax(youden))
    tau = float(thr[best])
    y_pred = (P >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, y_pred).ravel()
    sens = float(tp/(tp+fn)) if (tp+fn)>0 else 0.0
    specf = float(tn/(tn+fp)) if (tn+fp)>0 else 0.0
    auc = float(roc_auc_score(ys, P))
    return {
        'threshold': tau,
        'sensitivity': sens,
        'specificity': specf,
        'auc': auc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--target-sens', type=float, default=0.86)
    ap.add_argument('--target-spec', type=float, default=0.70)
    args = ap.parse_args()

    runs = load_preds(args.pattern)
    if len(runs) < 2:
        raise SystemExit('Not enough runs matched or missing preds; ensure --save-preds was used.')

    # Basic check on IDs alignment
    ids0 = runs[0][0]
    for ids, y, p, path in runs[1:]:
        if ids != ids0:
            raise SystemExit(f'Holdout IDs not aligned in {path}')

    metrics = bag_and_eval(runs, args.target_sens, args.target_spec)
    out_obj = {
        'pattern': args.pattern,
        'files': [r[3] for r in runs],
        'bagged_metrics': metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out_obj, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()

