# Experiment README

This directory contains YAML configs for the experiment runner. Use the runner to execute predefined grids:

Examples
- Wave A (thresholding + calibration robustness):
  ./venv/bin/python scripts/experiment_runner.py --config configs/experiments.yaml --wave A

- Wave B (ensemble/representation tuning):
  ./venv/bin/python scripts/experiment_runner.py --config configs/experiments.yaml --wave B

- Custom grid (if defined under CUSTOM):
  ./venv/bin/python scripts/experiment_runner.py --config configs/experiments.yaml --wave CUSTOM

Notes
- Ensure dependencies installed: pip install -r requirements.txt
- Results are written to results/*.json with informative names derived from variant settings.
- The pipeline supports new flags:
  --threshold-policy {both_targets,spec_first,youden,np}
  --threshold-transfer {median,iqr_mid,quantile_map}
  --quantile-guard-ks FLOAT
  --final-calibration {none,isotonic,sigmoid,temperature}
  --seed INT

