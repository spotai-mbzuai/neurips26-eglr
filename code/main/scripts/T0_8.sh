#!/bin/bash
mkdir -p outputs/.logs/EGLR/stdout   # run from code/
set -euo pipefail
# Run L=1..10 at temperature=0.8 sequentially on a single GPU.
# Usage:  bash main/scripts/T0_8.sh <GPU_ID>     (default GPU 0)
# Run from main/ folder.

GPU=${1:-0}
TEMP=0.8
TOPK=1
TOPP=1.0
DOSAMPLE=true
KVCACHE=modified
RESULTS_DIR=outputs/.checkpoints/EGLR/main_eglr/eglr_outputs/Qwen2.5-3B-Instruct/math_500
CONFIG=main/configs/qwen3b_all_MATH.yaml

for L in 1 2 3 4 5 6 7 8 9 10; do
    LOG=outputs/.logs/EGLR/stdout/L${L}_T${TEMP}.log
    echo "[T${TEMP}] Running L=${L} on GPU ${GPU} → ${LOG}"
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

echo "[T${TEMP}] All L's finished. Computing union accuracy..."
python main/scripts/compute_union.py ${TEMP} ${TOPK} ${RESULTS_DIR}
echo "[T${TEMP}] Done."
