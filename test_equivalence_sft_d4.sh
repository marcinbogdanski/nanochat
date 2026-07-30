#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft \
  -- --model-tag=d4 \
  --eval-every=10 --eval-tokens=524288 --chatcore-every=-1 --num-iterations=64

