"""Compute per-T accuracy and union for L=0 across temperatures 0.0..1.0.

Writes to results_L0_topk{TOPK}.json (appending).

Usage:
    python scripts/compute_union_L0.py [<TOPK>]   # default 1
"""
import json
import glob
import sys
import os
from datetime import datetime


def main():
    topk = sys.argv[1] if len(sys.argv) >= 2 else "1"
    results_dir = sys.argv[2] if len(sys.argv) >= 3 else "."
    out_dir = "outputs/.checkpoints/EGLR/main_eglr/eglr_outputs"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"results_L0_topk{topk}.json")

    per_T = {}
    correct_idx_by_T = {}
    temperatures = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5",
                    "0.6", "0.7", "0.8", "0.9", "1.0"]

    for T in temperatures:
        pattern = f"{out_dir}/**/*_L0_*_T{T}_topk{topk}_*.jsonl"
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            print(f"  [warn] no JSONL found for L=0 T={T}")
            continue
        fpath = matches[-1]
        correct = 0
        total = 0
        correct_idx = set()
        with open(fpath) as fp:
            for line in fp:
                r = json.loads(line)
                total += 1
                if r.get("correct"):
                    correct += 1
                    correct_idx.add(r["idx"])
        per_T[f"T{T}"] = {
            "correct": correct,
            "total": total,
            "file": os.path.basename(fpath),
        }
        correct_idx_by_T[T] = correct_idx

    union = set()
    for s in correct_idx_by_T.values():
        union |= s

    record = {
        "timestamp": datetime.now().isoformat(),
        "L": 0,
        "top_k": int(topk),
        "per_temperature": per_T,
        "union_correct": len(union),
        "union_total": 500,
    }

    existing = []
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    existing.append(record)
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n=== L=0 summary (topk={topk}) ===")
    for T_label, info in per_T.items():
        print(f"  {T_label}: {info['correct']}/{info['total']}")
    print(f"  UNION: {len(union)}/500")
    print(f"Appended to {results_path}\n")


if __name__ == "__main__":
    main()
