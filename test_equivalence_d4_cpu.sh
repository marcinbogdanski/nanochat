#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="" python -m scripts.base_train \
  --depth=4 --total-batch-size=2048 --device-batch-size=1 \
  --eval-every=10 --eval-tokens=2048 --core-metric-every=0 --sample-every=0 --save-every=10 --num-iterations=4 \
  --window-pattern=SSSL
