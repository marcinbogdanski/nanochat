#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.base_train --depth=4 --total-batch-size=262144 --device-batch-size=8 --eval-tokens=524288

# Prints:
# Tokens / micro-batch / rank: 8 x 2048 = 16,384
# Tokens / micro-batch: 16,384
# Total batch size 262,144 => gradient accumulation steps: 16
# model.smear_lambda = nan | model.backout_lambda = nan
# Step 00000 | Validation bpb: nan
# step 00000/00528 (0.00%) | loss: nan | lrm: 0.03 | dt: 3482.30ms | tok/sec: 75,279 | bf16_mfu: 8.51 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
# step 00001/00528 (0.19%) | loss: nan | lrm: 0.05 | dt: 1259.08ms | tok/sec: 208,202 | bf16_mfu: 23.52 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
# step 00002/00528 (0.38%) | loss: nan | lrm: 0.07 | dt: 1258.98ms | tok/sec: 208,219 | bf16_mfu: 23.52 | epoch: 1 pq: 0 rg: 2 | total time: 0.00m
# step 00003/00528 (0.57%) | loss: nan | lrm: 0.10 | dt: 1256.64ms | tok/sec: 208,606 | bf16_mfu: 23.57 | epoch: 1 pq: 0 rg: 2 | total time: 0.00m
# step 00004/00528 (0.76%) | loss: nan | lrm: 0.12 | dt: 1257.19ms | tok/sec: 208,514 | bf16_mfu: 23.56 | epoch: 1 pq: 0 rg: 3 | total time: 0.00m
