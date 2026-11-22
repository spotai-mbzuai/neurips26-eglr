#!/usr/bin/env bash
# =============================================================================
# Phase 3 full benchmark sweep — Llama-3.1-8B-Instruct
#
# Runs the full Phase 3 benchmark (aime24, aime25, amc23, minerva_math,
# math_500, gsm8k) at every L in {0, 1, ..., 10}, with tau_entropy
# auto-calibrated from each dataset's L=0 entropy distribution.
#
# The model is loaded ONCE (run_sweep.py holds it in memory across all 66
# runs). Refinement rate is checked after each L; tau is recomputed and the
# run retried (up to 3 attempts) if it falls outside [5%, 10%].
#
# Sequence per script:
#   aime24   : L=0, 1, 2, ..., 10
#   aime25   : L=0, 1, 2, ..., 10
#   amc23    : L=0, 1, 2, ..., 10
#   minerva  : L=0, 1, 2, ..., 10
#   math_500 : L=0, 1, 2, ..., 10
#   gsm8k    : L=0, 1, 2, ..., 10          ← last by convention
#
# Resume: by default, any (dataset, L) combination that already has an
# accepted summary (in-band refinement rate) on disk is skipped. Pass
# --no-resume to force a fresh run.
#
# Extra flags — anything after the script name is passed through to
# run_sweep.py, e.g.:
#   bash main/scripts/phase3_llama_3_1_8b.sh --max-attempts 5
#   bash main/scripts/phase3_llama_3_1_8b.sh --no-resume
#   bash main/scripts/phase3_llama_3_1_8b.sh --target-rate 0.08
# =============================================================================
set -euo pipefail

CONFIG="main/configs/phase3_llama_3_1_8b.yaml"
DATASETS="main/configs/phase3_datasets.yaml"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: $CONFIG not found. Run this script from the project root." >&2
    exit 1
fi
if [[ ! -f "$DATASETS" ]]; then
    echo "ERROR: $DATASETS not found." >&2
    exit 1
fi

echo
echo "============================================================"
echo "Phase 3 sweep  |  Llama-3.1-8B-Instruct"
echo "Config:     $CONFIG"
echo "Datasets:   $DATASETS"
echo "============================================================"

CUDA_VISIBLE_DEVICES=5 python main/run_sweep.py \
    --config   "$CONFIG" \
    --datasets "$DATASETS" \
    --L-list   "0,1,2,3,4,5,6,7,8,9,10" \
    "$@"

echo
echo "============================================================"
echo "Phase 3 DONE for Llama-3.1-8B-Instruct"
echo "============================================================"
