"""Compute paper Table 1 numbers from per-(model, dataset) JSONL outputs.

For every (model, dataset) cell, prints four numbers as LaTeX rows:
    Greedy        : accuracy at L=0
    EGLR          : best-single-L accuracy, max over L in {1, ..., 10}
    EGLR-Oracle   : "any L in {1, ..., 10} got it correct" (diversity ceiling)
    EGLR-SC       : majority vote over L in {0, ..., 10}

Reads only greedy (T=0.0) JSONLs. Both `topk0` and `topk1` count as greedy
(file naming differs between the older L*T grid run and the newer sweeps).

Usage (from main/):
    python output_extractor/compute_table1.py
    python output_extractor/compute_table1.py --models Qwen2.5-3B-Instruct,Llama-3.1-8B-Instruct
    python output_extractor/compute_table1.py --no-eglrsc          # skip EGLR-SC row

Output: LaTeX rows printed to stdout, ready to paste into the Table 1
        \\begin{tabular} ... \\end{tabular} block.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

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
DATASETS = ["gsm8k", "math_500", "minerva_math", "amc23", "aime24", "aime25"]
RESULTS_BASE = "outputs/.checkpoints/EGLR/main_eglr/eglr_outputs"


def load_jsonl(path):
    correct = set()
    preds = {}
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            idx = r["idx"]
            if r.get("correct"):
                correct.add(idx)
            preds[idx] = r.get("prediction")
    return correct, preds, n


def latest_greedy_file(model, dataset, L):
    pat = os.path.join(
        RESULTS_BASE, model, dataset,
        f"*_L{L}_*_T0.0_topk[01]_*_{dataset}.jsonl",
    )
    cands = sorted(glob.glob(pat))
    return cands[-1] if cands else None


def cell_numbers(model, dataset, want_eglrsc=True):
    """Return (greedy_acc, eglr_acc, oracle_acc, eglrsc_acc) as percents.

    Any of these can be None if data is missing.
    """
    by_L = {}
    n_total = None
    for L in range(11):
        f = latest_greedy_file(model, dataset, L)
        if not f:
            continue
        correct, preds, n = load_jsonl(f)
        by_L[L] = (correct, preds)
        if n_total is None:
            n_total = n

    if n_total is None or 0 not in by_L:
        return (None, None, None, None)

    n = n_total
    greedy = 100.0 * len(by_L[0][0]) / n

    # EGLR = max accuracy over L in 1..10
    L_accs = [
        100.0 * len(by_L[L][0]) / n
        for L in range(1, 11) if L in by_L
    ]
    eglr = max(L_accs) if L_accs else None

    # Oracle = union of correct sets over L in 1..10
    union = set()
    for L in range(1, 11):
        if L in by_L:
            union |= by_L[L][0]
    oracle = 100.0 * len(union) / n if L_accs else None

    eglrsc = None
    if want_eglrsc:
        # Majority vote per problem across L in 1..10 (L=0 = Greedy baseline
        # we're comparing against, so excluded from the EGLR-SC pool).
        Ls_used = [L for L in sorted(by_L.keys()) if L >= 1]
        if Ls_used:
            wins = 0
            problem_ids = sorted(set().union(*[set(by_L[L][1].keys()) for L in Ls_used]))
            for idx in problem_ids:
                votes = Counter()
                for L in Ls_used:
                    p = by_L[L][1].get(idx)
                    if p is not None:
                        votes[p] += 1
                if not votes:
                    continue
                top_pred, _ = votes.most_common(1)[0]
                # The most-voted answer is correct iff some L emitted top_pred
                # AND was marked correct (same prediction → same gold check).
                for L in Ls_used:
                    if by_L[L][1].get(idx) == top_pred and idx in by_L[L][0]:
                        wins += 1
                        break
            eglrsc = 100.0 * wins / n

    return (greedy, eglr, oracle, eglrsc)


def fmt(v):
    return f"{v:.1f}" if v is not None else "--"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model directory names.")
    parser.add_argument("--no-eglrsc", action="store_true",
                        help="Skip the EGLR-SC row.")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    want_sc = not args.no_eglrsc

    rows = []
    for m in models:
        cells = {d: cell_numbers(m, d, want_eglrsc=want_sc) for d in DATASETS}
        # Order columns as in the paper: GSM8K, MATH-500, MinervaMath, AMC23, AIME24, AIME25
        col_order = DATASETS
        n_method_rows = 4 if want_sc else 3
        method_names = ["Greedy", "EGLR", "EGLR-Oracle"] + (["EGLR-SC"] if want_sc else [])
        for i, method in enumerate(method_names):
            if i == 0:
                row_head = f"\\multirow{{{n_method_rows}}}{{*}}{{{m}}}"
            else:
                row_head = " " * len(m)
            vals = []
            for d in col_order:
                triple = cells[d]
                v = triple[i] if triple else None
                vals.append(fmt(v))
            row = f"{row_head:<44} & {method:<11} & " + " & ".join(f"{v:>5}" for v in vals) + " \\\\"
            rows.append(row)
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows = rows[:-1]  # drop trailing midrule
    print("\n".join(rows))


if __name__ == "__main__":
    main()
