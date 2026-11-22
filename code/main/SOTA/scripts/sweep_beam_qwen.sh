#!/bin/bash
set -euo pipefail
# Beam baseline sweep for Qwen2.5-{3B, 7B}-Instruct, sequential.
# For each model, runs beam=4 then beam=11 across all math datasets.
# Pass a comma-separated GPU list (e.g. "0,4") so HF can shard the model
# across both GPUs via device_map="auto" — handles OOM for the larger model.
# Calls main/SOTA/beam/main_eglr.py (the beam-enabled fork).
# Usage:  bash main/SOTA/scripts/sweep_beam_qwen.sh <GPU_LIST>     (default "0")
#   examples:
#     bash main/SOTA/scripts/sweep_beam_qwen.sh 0          # single GPU
#     bash main/SOTA/scripts/sweep_beam_qwen.sh "0,4"      # 2-GPU sharded (recommended)
# Run from code/.

GPU=${1:-"0"}
CONFIG=main/SOTA/configs/all_greedy.yaml
DATASETS=("math_500" "amc23" "aime24" "aime25" "minerva_math" "gsm8k")
BEAMS=(4 11)
BATCH_SIZE=16
MAX_NEW_TOKENS=3072

MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

# Per-dataset system prompts (AIME has strict 0-999 integer answer constraint).
PROMPT_DEFAULT='You are a mathematical expert. Solve the following problem step-by-step with clear reasoning. Ensure the final answer is simplified and placed inside \boxed{}.'
PROMPT_AIME='You are a mathematical expert. Solve the following AIME problem step-by-step. The answer is an integer between 0 and 999. Place the final answer inside \boxed{}.'

EXTRA_DIR=outputs/.logs/EGLR/SOTA/stdout
LOG_DIR=outputs/.logs/EGLR/SOTA
mkdir -p ${EXTRA_DIR} ${LOG_DIR}

for NUM_BEAMS in "${BEAMS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        MODEL_SHORT=$(echo "${MODEL}" | awk -F'/' '{print $NF}')
        for DATASET in "${DATASETS[@]}"; do
            echo ""
            echo "==================== ${MODEL_SHORT} | beam=${NUM_BEAMS} (bs=${BATCH_SIZE}, GPU=${GPU}) | ${DATASET} ===================="

            case "${DATASET}" in
                aime24|aime25) PROMPT="${PROMPT_AIME}" ;;
                *)             PROMPT="${PROMPT_DEFAULT}" ;;
            esac

            LOG=${EXTRA_DIR}/${MODEL_SHORT}_${DATASET}_beam${NUM_BEAMS}.log
            CUDA_VISIBLE_DEVICES=${GPU} python main/SOTA/beam/main_eglr.py \
                --config ${CONFIG} \
                --model ${MODEL} \
                --dataset ${DATASET} \
                --system_prompt "${PROMPT}" \
                --num_beams ${NUM_BEAMS} \
                --test_batch_size ${BATCH_SIZE} \
                --max_new_tokens ${MAX_NEW_TOKENS} \
                --device_map auto \
                --log-dir ${LOG_DIR} \
                --L 0 \
                > ${LOG} 2>&1 || echo "[WARN] ${MODEL_SHORT} × ${DATASET} beam=${NUM_BEAMS} failed — see ${LOG}"
        done
    done
done

echo ""
echo "[Qwen beam sweep] Done — ${#MODELS[@]} models × ${#BEAMS[@]} beam widths × ${#DATASETS[@]} datasets."
