#!/bin/bash
set -euo pipefail
# L x T grid sweep for Qwen2.5-3B-Instruct: amc23 + aime24 + aime25.
# Per dataset: tau_entropy is hardcoded from the L=0 baseline run that was
# already auto-calibrated. We then run L=1..10 across T=0.1..1.0 (sampling).
# Usage:  bash main/scripts/sweep_LT_3b_set1.sh <GPU_ID>     (default GPU 0)
# Run from main/ folder.

GPU=${1:-0}
MODEL="Qwen/Qwen2.5-3B-Instruct"
MODEL_SHORT="Qwen2.5-3B-Instruct"
CONFIG=main/configs/qwen3b_all_MATH.yaml
DATASETS=("amc23" "aime24" "aime25")

# Per-dataset tau (95th percentile of L=0 per-token entropies, cap=2.5).
declare -A DATASET_TAU=(
    [amc23]=0.998305
    [aime24]=0.933815
    [aime25]=1.11353
)

# Sampling knobs (kept identical across all T to isolate the T sweep).
TOPK=50
TOPP=1.0
DOSAMPLE=true
KVCACHE=modified
MAX_NEW_TOKENS=3072

# Per-dataset system prompts (AIME has strict 0-999 integer answer constraint).
PROMPT_DEFAULT='You are a mathematical expert. Solve the following problem step-by-step with clear reasoning. Ensure the final answer is simplified and placed inside \boxed{}.'
PROMPT_AIME='You are a mathematical expert. Solve the following AIME problem step-by-step. The answer is an integer between 0 and 999. Place the final answer inside \boxed{}.'

mkdir -p outputs/.logs/EGLR/stdout

for DATASET in "${DATASETS[@]}"; do
    TAU=${DATASET_TAU[${DATASET}]}

    case "${DATASET}" in
        aime24|aime25) PROMPT="${PROMPT_AIME}" ;;
        *)             PROMPT="${PROMPT_DEFAULT}" ;;
    esac

    echo ""
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
done

echo ""
echo "[set1] Done — ${#DATASETS[@]} datasets × 10 T × 10 L."
