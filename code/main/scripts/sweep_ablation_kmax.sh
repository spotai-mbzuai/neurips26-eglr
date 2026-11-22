#!/bin/bash
set -euo pipefail
# Ablation: max refinement iterations K_max on Qwen2.5-3B-Instruct x MATH-500
# at L=4 (greedy decoding).
# Fixed: L=4, alpha=0.2 (yaml default), tau_entropy=1.2 (yaml default), greedy.
# Sweeps K_max in {1, 2, 4, 5}; K_max=3 is the default (already covered by the
# existing L=4 baseline run).
# Each run lands in outputs/.checkpoints/EGLR/.../Qwen2.5-3B-Instruct/math_500/ with K_max
# embedded in the run_tag, so the cells coexist with the default L=4 run.
# Usage:  bash main/scripts/sweep_ablation_kmax.sh <GPU_ID>     (default GPU 0)
# Run from main/ folder.

GPU=${1:-0}
MODEL="Qwen/Qwen2.5-3B-Instruct"
MODEL_SHORT="Qwen2.5-3B-Instruct"
CONFIG=main/configs/all_greedy.yaml
DATASET="math_500"
L=4
ALPHA=0.2
TEMP=0.0
TOPK=1
DOSAMPLE=false
MAX_NEW_TOKENS=3072

PROMPT='You are a mathematical expert. Solve the following problem step-by-step with clear reasoning. Ensure the final answer is simplified and placed inside \boxed{}.'

mkdir -p outputs/.logs/EGLR/stdout

echo "==================== ${MODEL_SHORT} | ablation: K_max (L=${L}, alpha=${ALPHA}) ===================="
for KMAX in 1 2 4 5; do
    LOG=outputs/.logs/EGLR/stdout/${MODEL_SHORT}_${DATASET}_L${L}_K${KMAX}.log
    echo "[ablation:K_max] K_max=${KMAX} -> ${LOG}"
    CUDA_VISIBLE_DEVICES=${GPU} python main/main_eglr.py \
        --config ${CONFIG} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --samples -1 \
        --system_prompt "${PROMPT}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --L ${L} \
        --K_max ${KMAX} \
        --alpha ${ALPHA} \
        --temperature ${TEMP} \
        --top_k ${TOPK} \
        --do_sample ${DOSAMPLE} \
        > ${LOG} 2>&1 || echo "[WARN] K_max=${KMAX} failed -- see ${LOG}"
done

echo ""
echo "[ablation:K_max] Done -- 4 runs."
