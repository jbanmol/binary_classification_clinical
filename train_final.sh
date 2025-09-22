#!/usr/bin/env bash
set -euo pipefail

# 1) Ensure venv and deps
if [ ! -d venv ]; then
  python3 -m venv venv
fi
./venv/bin/python -m pip install --upgrade pip >/dev/null
./venv/bin/pip install -r requirements.txt

# 2) Run final pipeline on fixed holdout, saving preds for bagging
SEEDS=(17 29 43)
HOLDOUT_SEED=777
OUTS=()
for S in "${SEEDS[@]}"; do
  OUT="results/final_s${S}_ho${HOLDOUT_SEED}_np_iqrmid_temp_u16n50_k2.json"
  ./venv/bin/python scripts/clinical_fair_pipeline.py \
    --target-sens 0.86 --target-spec 0.70 \
    --use-umap-cosine --umap-components 16 --umap-neighbors 50 \
    --calibration isotonic --final-calibration temperature \
    --models lightgbm,xgboost,brf,extratrees --top-k-models 2 \
    --threshold-policy np --threshold-transfer iqr_mid \
    --seed "$S" --holdout-seed "$HOLDOUT_SEED" --save-preds \
    --out-name "$(basename "$OUT")"
  OUTS+=("$OUT")
done

# 3) Bag seeds and write final bagged metrics
./venv/bin/python scripts/bag_scores.py \
  --pattern "results/final_s*_ho${HOLDOUT_SEED}_np_iqrmid_temp_u16n50_k2.json" \
  --out results/final_bagged_ho${HOLDOUT_SEED}_np_iqrmid_temp_u16n50_k2.json

# 4) Export bundle using seed 17 (reference bundle)
./venv/bin/python scripts/clinical_fair_pipeline.py \
  --target-sens 0.86 --target-spec 0.70 \
  --use-umap-cosine --umap-components 16 --umap-neighbors 50 \
  --calibration isotonic --final-calibration temperature \
  --models lightgbm,xgboost,brf,extratrees --top-k-models 2 \
  --threshold-policy np --threshold-transfer iqr_mid \
  --seed 17 --holdout-seed "$HOLDOUT_SEED" --save-preds \
  --export-dir models/final_np_iqrmid_u16n50_k2 \
  --out-name export_s17_ho${HOLDOUT_SEED}_np_iqrmid_temp_u16n50_k2.json

# 5) Print summary
echo "\n=== Final bagged metrics ==="
cat results/final_bagged_ho${HOLDOUT_SEED}_np_iqrmid_temp_u16n50_k2.json

