"""Cross-temperature union: for each L, how many problems were solved at SOME T?

Reads the results_T*.json files in main/, then loads the underlying
JSONL files to compute the per-problem `correct` set per (L, T).
For each L, takes the union of correct-idx sets across all T's.

Usage (from main/):
    python scripts/cross_temp_union.py [results_T0.5.json results_T0.6.json ...]

If no args, defaults to all results_T*.json files in CWD.
"""
import json
import sys
import os
import glob


def main():
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("results_T*.json"))
    if not args:
        print("No results files found.")
        sys.exit(1)

    out_dir = "outputs/.checkpoints/EGLR/main_eglr/eglr_outputs"

    # Collect: per_L_T[(L, T)] = set of correct idxs
    per_L_T = {}
    temperatures = []
    for path in args:
        with open(path) as f:
            records = json.load(f)
        rec = records[-1] if isinstance(records, list) else records
        T = rec.get("temperature")
        temperatures.append(T)
        for L_label, info in rec.get("per_layer", {}).items():
            L = int(L_label[1:])  # 'L3' -> 3
            # Search recursively because files may be under <model>/<dataset>/ subdirs
            basename = os.path.basename(info["file"])
            jsonl_path = os.path.join(out_dir, basename)
            if not os.path.exists(jsonl_path):
                hits = glob.glob(os.path.join(out_dir, "**", basename), recursive=True)
                if hits:
                    jsonl_path = hits[0]
                else:
                    print(f"  [warn] missing JSONL for L={L} T={T}: {basename}")
                    continue
            correct = set()
            with open(jsonl_path) as fp:
                for line in fp:
                    r = json.loads(line)
                    if r.get("correct"):
                        correct.add(r["idx"])
            per_L_T[(L, T)] = correct

    L_values = sorted({L for (L, _) in per_L_T.keys()})
    T_values = temperatures

    # Build table
    print(f"\nCross-temperature accuracy table (correct out of 500):\n")
    header = ["L"] + [f"T={T}" for T in T_values] + ["UNION (any T)"]
    print("  ".join(f"{h:>12}" for h in header))
    print("  ".join(["-" * 12] * len(header)))

    union_per_L = {}
    for L in L_values:
        row = [f"L{L}"]
        union_set = set()
        for T in T_values:
            s = per_L_T.get((L, T), set())
            row.append(str(len(s)))
            union_set |= s
        row.append(f"{len(union_set)}")
        union_per_L[L] = union_set
        print("  ".join(f"{c:>12}" for c in row))

    # Grand union across all (L, T) pairs
    grand = set()
    for s in per_L_T.values():
        grand |= s
    print()
    print(f"GRAND UNION across all L and all T: {len(grand)}/500")

    # Per-T sum: for each T, union across L's (just the column-wise union)
    print()
    print("Per-T union across L (any L solves it at this T):")
    for T in T_values:
        union_T = set()
        for L in L_values:
            union_T |= per_L_T.get((L, T), set())
        print(f"  T={T}:  {len(union_T)}/500")


if __name__ == "__main__":
    main()
