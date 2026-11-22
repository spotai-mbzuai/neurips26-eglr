#!/bin/bash
mkdir -p outputs/.logs/EGLR/stdout   # run from code/
set -euo pipefail
# Run L=0 (no refinement, baseline) at temperatures 0.0..1.0 with top_k=1.
# Note: top_k=1 forces argmax, so all temperatures produce identical greedy
# output — this run is mainly for filename consistency / completeness.
# Usage:  bash main/scripts/L0.sh <GPU_ID>     (default GPU 0)
# Run from main/ folder.

GPU=${1:-0}
TOPK=1
TOPP=1.0
KVCACHE=modified
RESULTS_DIR=outputs/.checkpoints/EGLR/main_eglr/eglr_outputs/Qwen2.5-3B-Instruct/math_500
CONFIG=main/configs/qwen3b_L0_MATH.yaml
L=0

for TEMP in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    if [ "${TEMP}" = "0.0" ]; then
        DOSAMPLE=false
    else
        DOSAMPLE=true
    fi
    LOG=outputs/.logs/EGLR/stdout/L${L}_T${TEMP}.log
    echo "[L0] Running T=${TEMP} on GPU ${GPU} → ${LOG}"
    CUDA_VISIBLE_DEVICES=${GPU} python main/main_eglr.py \
        --config ${CONFIG} \
        --L ${L} \
        --temperature ${TEMP} \
        --top_k ${TOPK} \
        --top_p ${TOPP} \
        --do_sample ${DOSAMPLE} \
        --kv_cache ${KVCACHE} \
        > ${LOG} 2>&1
done

echo "[L0] All temperatures finished. Computing per-T accuracy and union..."
python main/scripts/compute_union_L0.py ${TOPK} ${RESULTS_DIR}
echo "[L0] Done."
