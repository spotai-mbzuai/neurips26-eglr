#!/bin/bash
set -euo pipefail
# L x T grid sweep for Qwen2.5-3B-Instruct: minerva_math.
# tau_entropy hardcoded from the L=0 baseline (auto-calibrated, cap=2.5).
# Runs L=1..10 across T=0.1..1.0 (sampling).
# Usage:  bash main/scripts/sweep_LT_3b_minerva.sh <GPU_ID>     (default GPU 0)
# Run from main/ folder.

GPU=${1:-0}
MODEL="Qwen/Qwen2.5-3B-Instruct"
MODEL_SHORT="Qwen2.5-3B-Instruct"
CONFIG=main/configs/qwen3b_all_MATH.yaml
DATASET="minerva_math"
TAU=1.2658

TOPK=50
TOPP=1.0
DOSAMPLE=true
KVCACHE=modified
MAX_NEW_TOKENS=3072

PROMPT='You are a mathematical expert. Solve the following problem step-by-step with clear reasoning. Ensure the final answer is simplified and placed inside \boxed{}.'

mkdir -p outputs/.logs/EGLR/stdout

echo "==================== ${MODEL_SHORT} × ${DATASET}  (tau=${TAU}) ===================="

for T in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for L in 1 2 3 4 5 6 7 8 9 10; do
        LOG=outputs/.logs/EGLR/stdout/${MODEL_SHORT}_${DATASET}_T${T}_L${L}.log
        echo "[${DATASET}] T=${T} L=${L} → ${LOG}"
        CUDA_VISIBLE_DEVICES=${GPU} python main/main_eglr.py \
            --config ${CONFIG} \
            --model ${MODEL} \
            --dataset ${DATASET} \
            --samples -1 \
            --system_prompt "${PROMPT}" \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --L ${L} \
            --tau_entropy ${TAU} \
            --temperature ${T} \
            --top_k ${TOPK} \
            --top_p ${TOPP} \
            --do_sample ${DOSAMPLE} \
            --kv_cache ${KVCACHE} \
            > ${LOG} 2>&1 || echo "[WARN] ${DATASET} T=${T} L=${L} failed — see ${LOG}"
    done
done

echo ""
echo "[${DATASET}] Done — 10 T × 10 L."
