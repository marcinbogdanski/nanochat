#!/usr/bin/env bash
set -euo pipefail

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
  -- --depth=4 --device_batch_size=8 \
  --eval_every=100 --core_metric_every=100 --sample_every=100 --save_every=100 \
  --run=d4k
