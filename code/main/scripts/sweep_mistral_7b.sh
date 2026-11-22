#!/bin/bash
set -euo pipefail
# Greedy L=0..10 sweep across all math datasets for Mistral-7B-Instruct-v0.3.
# Auto-calibrates tau_entropy as the 95th-percentile per-token entropy from
# each L=0 run, so each (dataset) gets its own data-driven tau.
# Usage:  bash main/scripts/sweep_mistral_7b.sh <GPU_ID>     (default GPU 0)
# Run from main/ folder.

GPU=${1:-0}
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT="Mistral-7B-Instruct-v0.3"
CONFIG=main/configs/all_greedy.yaml
# math_500, amc23, aime24, aime25, minerva_math already completed for Mistral.
# gsm8k was killed mid-run after L=0 baseline finished — resume with the known tau.
DATASETS=("gsm8k")
TARGET_REFINE_RATE=0.05
MAX_ENTROPY=2.5      # cap before computing 95th percentile (drops outlier high-entropy tokens)
MAX_NEW_TOKENS=2048  # gsm8k answers are short; 2048 is plenty and cuts wall time
RESULTS_BASE=outputs/.checkpoints/EGLR/main_eglr/eglr_outputs

# Per-dataset system prompts (AIME has strict 0-999 integer answer constraint).
PROMPT_DEFAULT='You are a mathematical expert. Solve the following problem step-by-step with clear reasoning. Ensure the final answer is simplified and placed inside \boxed{}.'
PROMPT_AIME='You are a mathematical expert. Solve the following AIME problem step-by-step. The answer is an integer between 0 and 999. Place the final answer inside \boxed{}.'

mkdir -p outputs/.logs/EGLR/stdout

for DATASET in "${DATASETS[@]}"; do
    RESULTS_DIR=${RESULTS_BASE}/${MODEL_SHORT}/${DATASET}
    mkdir -p ${RESULTS_DIR}
    echo ""
    echo "==================== ${MODEL_SHORT} × ${DATASET} ===================="

    case "${DATASET}" in
        aime24|aime25) PROMPT="${PROMPT_AIME}" ;;
        *)             PROMPT="${PROMPT_DEFAULT}" ;;
    esac

    # Resume logic: gsm8k L=0 was already completed before the kill; reuse tau.
    if [ "${DATASET}" = "gsm8k" ]; then
        TAU=1.232400
        echo "[${DATASET}] RESUME — reusing tau=${TAU} from prior L=0 run"
    else
        # Step 1: L=0 baseline (no refinement; harvest entropy distribution).
        LOG=outputs/.logs/EGLR/stdout/${MODEL_SHORT}_${DATASET}_L0.log
        echo "[${DATASET}] Running L=0 (baseline) → ${LOG}"
        CUDA_VISIBLE_DEVICES=${GPU} python main/main_eglr.py \
            --config ${CONFIG} \
            --model ${MODEL} \
            --dataset ${DATASET} \
            --system_prompt "${PROMPT}" \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --L 0 \
            > ${LOG} 2>&1

        # Step 2: Auto-calibrate tau from the L=0 JSONL.
        L0_JSONL=$(ls -t ${RESULTS_DIR}/*_L0_*.jsonl | head -1)
        TAU=$(python main/scripts/compute_tau.py "${L0_JSONL}" ${TARGET_REFINE_RATE} ${MAX_ENTROPY})
        echo "[${DATASET}] Calibrated tau_entropy = ${TAU}  (target rate = ${TARGET_REFINE_RATE})"
    fi

    # Step 3: Run L=1..10 with the chosen tau.
    for L in 1 2 3 4 5 6 7 8 9 10; do
        LOG=outputs/.logs/EGLR/stdout/${MODEL_SHORT}_${DATASET}_L${L}.log
        echo "[${DATASET}] Running L=${L} (tau=${TAU}) → ${LOG}"
        CUDA_VISIBLE_DEVICES=${GPU} python main/main_eglr.py \
            --config ${CONFIG} \
            --model ${MODEL} \
            --dataset ${DATASET} \
            --system_prompt "${PROMPT}" \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --L ${L} \
            --tau_entropy ${TAU} \
            > ${LOG} 2>&1
    done
done

echo ""
echo "[${MODEL_SHORT}] Done — all ${#DATASETS[@]} datasets, L=0..10."
