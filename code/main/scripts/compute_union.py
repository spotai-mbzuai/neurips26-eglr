"""Compute per-L accuracy and union across L=1..10 for a given temperature.

Called by each T*.sh after its L sweep finishes. Writes (appending) a JSON
record to `results_T${TEMP}.json` in main/.

Usage:
    python scripts/compute_union.py <TEMP>
e.g.:
    python scripts/compute_union.py 0.5
"""
import json
import glob
import sys
import os
from datetime import datetime


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_union.py <TEMP> [<TOPK>] [<RESULTS_DIR>]")
        sys.exit(1)
    T = sys.argv[1]
    topk = sys.argv[2] if len(sys.argv) >= 3 else "50"
    results_dir = sys.argv[3] if len(sys.argv) >= 4 else "."

    out_dir = "outputs/.checkpoints/EGLR/main_eglr/eglr_outputs"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"results_T{T}_topk{topk}.json")

    per_L = {}
    correct_idx_by_L = {}

    for L in range(1, 11):
        pattern = f"{out_dir}/**/*_L{L}_*_T{T}_topk{topk}_*.jsonl"
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            print(f"  [warn] no JSONL found for L={L} T={T}")
            continue
        fpath = matches[-1]  # most recent

        correct = 0
        total = 0
        correct_idx = set()
        with open(fpath, "r") as fp:
            for line in fp:
                r = json.loads(line)
                total += 1
                if r.get("correct"):
                    correct += 1
                    correct_idx.add(r["idx"])
        per_L[f"L{L}"] = {
            "correct": correct,
            "total": total,
            "file": os.path.basename(fpath),
        }
        correct_idx_by_L[L] = correct_idx

    union = set()
    for s in correct_idx_by_L.values():
        union |= s

    record = {
        "timestamp": datetime.now().isoformat(),
        "temperature": T,
        "top_k": int(topk),
        "per_layer": per_L,
        "union_correct": len(union),
        "union_total": 500,
    }

    # Append: load existing list (or start fresh) and add this record
    existing = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    existing.append(record)
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n=== T={T} summary ===")
    for L_label, info in per_L.items():
        print(f"  {L_label}: {info['correct']}/{info['total']}")
    print(f"  UNION: {len(union)}/500")
    print(f"Appended to {results_path}\n")


if __name__ == "__main__":
    main()
