"""Compute paper Table 2 (per-L sweep) numbers from JSONL outputs.

For each (dataset, model) cell, prints L=0..10 greedy accuracy + the
cross-L oracle (union of correct over L in 1..10) as LaTeX rows.

Reads only greedy (T=0.0) JSONLs. Both `topk0` and `topk1` count as greedy
(file naming differs between the older L*T grid run and the newer sweeps).

Usage (from main/):
    python output_extractor/compute_table2.py
    python output_extractor/compute_table2.py --datasets math_500,gsm8k
    python output_extractor/compute_table2.py --models Qwen2.5-3B-Instruct

Output: full Table 2 body (per-dataset sections, 8 model rows each)
        printed as LaTeX, ready to paste into the \\begin{tabular} block.
"""
import argparse
import glob
import json
import os
from collections import OrderedDict

DEFAULT_MODELS = [
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-Math-1.5B-Instruct",
    "Qwen2.5-Math-7B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
]
DEFAULT_DATASETS = OrderedDict([
    ("MATH-500", "math_500"),
    ("GSM8K", "gsm8k"),
    ("MinervaMath", "minerva_math"),
    ("AMC23", "amc23"),
    ("AIME24", "aime24"),
    ("AIME25", "aime25"),
])
RESULTS_BASE = "outputs/.checkpoints/EGLR/main_eglr/eglr_outputs"


def latest_greedy_file(model, dataset, L):
    pat = os.path.join(
        RESULTS_BASE, model, dataset,
        f"*_L{L}_*_T0.0_topk[01]_*_{dataset}.jsonl",
    )
    cands = sorted(glob.glob(pat))
    return cands[-1] if cands else None


def load_correct_set(path):
    correct = set()
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            if r.get("correct"):
                correct.add(r["idx"])
    return correct, n


def model_row(model, dataset_dir):
    """Returns 12 strings: L=0..10 accuracies + Oracle, all percents."""
    by_L = {}
    n_total = None
    for L in range(11):
        f = latest_greedy_file(model, dataset_dir, L)
        if not f:
            continue
        correct, n = load_correct_set(f)
        by_L[L] = correct
        if n_total is None:
            n_total = n
    if n_total is None:
        return ["--"] * 12

    cells = []
    union = set()
    for L in range(11):
        if L in by_L:
            cells.append(f"{100.0 * len(by_L[L]) / n_total:.1f}")
            if 1 <= L <= 10:
                union |= by_L[L]
        else:
            cells.append("--")
    oracle_str = (
        f"\\textbf{{{100.0 * len(union) / n_total:.1f}}}"
        if any(L in by_L for L in range(1, 11))
        else "--"
    )
    cells.append(oracle_str)
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model directory names.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS.values()),
                        help="Comma-separated dataset directory names (e.g. math_500,gsm8k).")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    requested_dirs = [d.strip() for d in args.datasets.split(",") if d.strip()]
    # Map requested dirs back to display names (preserving paper order)
    display_for = {v: k for k, v in DEFAULT_DATASETS.items()}
    sections = [(display_for.get(d, d), d) for d in requested_dirs]

    out = []
    for label, ddir in sections:
        out.append(f"\\midrule")
        out.append(f"\\multicolumn{{13}}{{c}}{{\\textbf{{{label}}}}} \\\\")
        out.append(f"\\midrule")
        for m in models:
            cells = model_row(m, ddir)
            row = f"{m:<26} & " + " & ".join(f"{v:>13}" for v in cells) + " \\\\"
            out.append(row)
    print("\n".join(out))


if __name__ == "__main__":
    main()
