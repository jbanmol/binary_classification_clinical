#!/usr/bin/env python3
"""
Experiment Runner
- Executes Wave A (thresholding + calibration robustness) and Wave B (ensemble/representation tuning)
  using YAML-defined grids.
- Produces JSON result files in results/ with informative names.

Usage:
  ./venv/bin/python scripts/experiment_runner.py --config configs/experiments.yaml --wave A
  ./venv/bin/python scripts/experiment_runner.py --config configs/experiments.yaml --wave B

You can also run a custom YAML with arbitrary grids.
"""
import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / 'venv' / 'bin' / 'python'
PIPELINE = REPO_ROOT / 'scripts' / 'clinical_fair_pipeline.py'
RESULTS_DIR = REPO_ROOT / 'results'


def run_command(cmd: List[str]) -> int:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("$", " ".join(cmd))
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    return proc.returncode


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def build_cmd(base: Dict[str, Any]) -> List[str]:
    args = []
    # map boolean flags and parameters to CLI
    bool_flags = {
        'use_polynomial': '--use-polynomial',
        'use_umap_cosine': '--use-umap-cosine',
        'report_holdout_specfirst': '--report-holdout-specfirst',
        'use_quantile_threshold': '--use-quantile-threshold',
    }
    param_flags = {
        'target_sens': '--target-sens',
        'target_spec': '--target-spec',
        'umap_components': '--umap-components',
        'umap_neighbors': '--umap-neighbors',
        'calibration': '--calibration',
        'final_calibration': '--final-calibration',
        'models': '--models',
        'top_k_models': '--top-k-models',
        'threshold_policy': '--threshold-policy',
        'threshold_transfer': '--threshold-transfer',
        'quantile_guard_ks': '--quantile-guard-ks',
        'seed': '--seed',
        'holdout_seed': '--holdout-seed',
        'out_name': '--out-name',
    }
    # booleans
    for k, flag in bool_flags.items():
        if base.get(k):
            args.append(flag)
    # params
    for k, flag in param_flags.items():
        if k in base and base[k] is not None:
            args.extend([flag, str(base[k])])
    # Always save preds for bagging
    args.append('--save-preds')
    return [str(PYTHON_BIN), str(PIPELINE)] + args


def run_wave(config: Dict[str, Any], wave: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    common = config.get('common', {})
    seeds = config.get('seeds', [42])
    grid = config.get('grid', {})

    variants = expand_grid(grid)
    print(f"Running wave {wave} with {len(variants)} variants x {len(seeds)} seeds")

    for seed in seeds:
        for variant in variants:
            params = dict(common)
            params.update(variant)
            params['seed'] = seed
            # Build informative out_name if not set
            if 'out_name' not in params or not params['out_name']:
                name_bits = [
                    f"wave{wave}",
                    f"s{seed}",
                    f"ho{params.get('holdout_seed','-')}",
                    f"tp{params.get('threshold_policy','bt')}",
                    f"tt{params.get('threshold_transfer','med')}",
                    f"fc{params.get('final_calibration','none')}",
                    f"u{params.get('umap_components','-')}n{params.get('umap_neighbors','-')}",
                    f"k{params.get('top_k_models','-')}",
                ]
                params['out_name'] = "_".join(str(x) for x in name_bits if x) + ".json"
            cmd = build_cmd(params)
            code = run_command(cmd)
            if code != 0:
                print(f"Variant failed with exit code {code}: {params}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='Path to YAML config defining waves')
    ap.add_argument('--wave', type=str, choices=['A', 'B', 'CUSTOM'], required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f)

    if args.wave in cfg:
        run_wave(cfg[args.wave], args.wave)
    elif args.wave == 'CUSTOM' and 'CUSTOM' in cfg:
        run_wave(cfg['CUSTOM'], 'CUSTOM')
    else:
        raise SystemExit(f"Wave {args.wave} not found in config {args.config}")


if __name__ == '__main__':
    main()

