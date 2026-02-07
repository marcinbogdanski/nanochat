#!/usr/bin/env bash
set -euo pipefail

TOTAL_BATCH_SIZE=$((524288/128))

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    --depth=10 \
    --total_batch_size="${TOTAL_BATCH_SIZE}" \
    --device_batch_size=1 \
    --max_seq_len=1024 \
    --num_iterations=10
