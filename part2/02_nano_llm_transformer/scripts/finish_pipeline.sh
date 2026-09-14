#!/usr/bin/env bash
# Runs after AutoResearch: scale up the champion, verify stability, train, SFT, evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=${STEPS:-4000}

echo "=== 1/5 build champion config ==="
$PY scripts/build_champion_config.py --steps "$STEPS"

echo; echo "=== 2/5 stability probe (150 steps at full width) ==="
# The search accepted its learning rate at d256; optimal LR is not width-invariant
# outside muP, so confirm the scaled model is stable before committing an hour.
$PY -u -m slm.train --config configs/champion.yaml --name champion-probe \
    --set train.steps=150 train.eval_every=75 train.eval_iters=10 \
          train.sample_every=0 train.save_checkpoints=false 2>&1 | tail -8

echo; echo "=== 3/5 full pretraining ($STEPS steps) ==="
$PY -u -m slm.train --config configs/champion.yaml 2>&1 | tail -30

CKPT=$($PY -c "from slm.infer import latest_checkpoint; print(latest_checkpoint(phase='pretrain'))")
echo; echo "=== 4/5 instruction tuning from $CKPT ==="
$PY -u -m slm.sft --base-ckpt "$CKPT" --steps 1500 2>&1 | tail -20

echo; echo "=== 5/5 evaluation ==="
$PY -u -m slm.evaluate --phase pretrain --instruct data/raw/instruct_valid.txt 2>&1 | tail -14
$PY -u -m slm.evaluate --phase sft --instruct data/raw/instruct_valid.txt 2>&1 | tail -14
echo; echo "=== pipeline complete ==="
