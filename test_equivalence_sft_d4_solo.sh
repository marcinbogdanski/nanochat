#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.chat_sft \
  --model-tag=d4 \
  --eval-every=10 --eval-tokens=524288 --chatcore-every=-1 --num-iterations=64

