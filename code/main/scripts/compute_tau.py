"""Auto-calibrate tau_entropy from an L=0 JSONL run.

Reads per-token entropies across ALL records in the JSONL, OPTIONALLY caps
them at `max_entropy` (drops outlier high-entropy tokens that would otherwise
inflate the quantile — important for over-confident models like Qwen2.5-Math
where 95th percentile of the full distribution sits at 11+), and prints the
(1 - target_rate)-th quantile of the filtered set to stdout.

Usage:
    python scripts/compute_tau.py <jsonl_path> [<target_rate>] [<max_entropy>]

  <jsonl_path>    Path to an L=0 JSONL (or a glob; latest match is used).
  <target_rate>   Default 0.05 (i.e., 5%% refinement rate, 95th percentile).
  <max_entropy>   Default 2.5 — drop tokens with entropy above this cap
                  before computing the quantile. Pass `inf` to disable.

Designed to be captured by a shell script:
    TAU=$(python scripts/compute_tau.py /path/to/L0.jsonl 0.05 2.5)
"""
import glob
import json
import sys

import numpy as np


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(
            "Usage: compute_tau.py <jsonl_path> [<target_rate>] [<max_entropy>]\n"
        )
        sys.exit(1)
    jsonl_path  = sys.argv[1]
    target_rate = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.05
    max_entropy = float(sys.argv[3]) if len(sys.argv) >= 4 else 2.5

    # Allow a glob pattern; take the most recent match
    if any(ch in jsonl_path for ch in "*?["):
        matches = sorted(glob.glob(jsonl_path))
        if not matches:
            sys.stderr.write(f"No files match: {jsonl_path}\n")
            sys.exit(1)
        jsonl_path = matches[-1]

    entropies = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            entropies.extend(r.get("per_token_entropies", []))

    if not entropies:
        sys.stderr.write(f"No per-token entropies in {jsonl_path}\n")
        sys.exit(1)

    n_total = len(entropies)
    filtered = [e for e in entropies if e <= max_entropy]
    n_kept = len(filtered)

    if not filtered:
        sys.stderr.write(
            f"All {n_total} tokens have entropy > max_entropy={max_entropy}; "
            f"falling back to unfiltered quantile.\n"
        )
        filtered = entropies
        n_kept = n_total

    tau = float(np.quantile(filtered, 1.0 - target_rate))
    sys.stderr.write(
        f"[compute_tau] {jsonl_path}: {n_total} tokens, "
        f"{n_kept} after cap (entropy <= {max_entropy}), "
        f"tau = quantile({1 - target_rate:.3f}) = {tau:.4f}\n"
    )
    # ONLY the numeric value goes to stdout (for shell capture)
    print(f"{tau:.6f}")


if __name__ == "__main__":
    main()
