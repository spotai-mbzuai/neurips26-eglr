"""
Auto-tuning utilities for EGLR's tau_entropy.

Design
──────
The refinement rate is, by definition,

    rate ≈ P(entropy_0 > tau)                                  (1)

where entropy_0 is the next-token entropy BEFORE refinement fires (the
value EGLR checks against tau). Eq. (1) inverts cleanly: the tau that
yields a target rate `r` is the (1 - r)-th quantile of the observed
entropy distribution.

    tau = quantile(entropies, 1 - r)                           (2)

The target band from the plan is [5%, 10%]; we aim for the midpoint (7.5%)
so we have slack in both directions. Since entropy_0 is a pre-refinement
quantity, its distribution shifts only slightly when L changes — so a
tau picked from the L=0 distribution works well as a starting point for
all L ∈ [1, 10]. If the observed rate drifts outside the band, we
recompute tau from the just-finished run's entropy distribution and
retry. This is a fixed-point iteration and converges in 1-2 attempts in
practice.

Safety clamps (MIN_TAU, MAX_TAU) keep the search inside sane bounds in
pathological cases (e.g. all-zero entropies on a frozen model, or a
uniform-almost-random output that would push tau into the stratosphere).
"""


# =============================================================================
# REQUIRE RUNNING FROM code/ (meta-repo neurips26-eglr)
# =============================================================================
import sys
from pathlib import Path as _Path
_git_root = next((p for p in _Path(__file__).resolve().parents if (p / ".git").exists()), None)
_expected_cwd = _git_root / "code" if _git_root is not None else None
if _expected_cwd is None or _Path.cwd().resolve() != _expected_cwd:
    sys.stderr.write(
        f"\n[ERROR] This script must be run from code/:          {_expected_cwd}\n"
        f"        Current working directory:                    {_Path.cwd().resolve()}\n"
        f"        Example:\n"
        f"            cd {_expected_cwd}\n"
        f"            python {_Path(__file__).resolve().relative_to(_expected_cwd) if _expected_cwd else _Path(__file__).name} --config <yaml>\n\n"
    )
    sys.exit(1)

import json
import os
from typing import Iterable, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Plan-mandated target band.
# ─────────────────────────────────────────────────────────────────────────────
TARGET_RATE_LOW  = 0.03
TARGET_RATE_HIGH = 0.15
TARGET_RATE_MID  = 0.09

# Safety clamps — tau below MIN_TAU refines almost every token (runaway
# compute); above MAX_TAU refines none (degenerates to baseline).
MIN_TAU = 0.10
MAX_TAU = 5.0

# Max retries per (dataset, L) before we give up and accept whatever rate
# the last attempt produced.
DEFAULT_MAX_ATTEMPTS = 3

QUANTILE_GRID = [0.50, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]
QUANTILE_KEYS = ["p50", "p75", "p80", "p85", "p90", "p92_5", "p95", "p97_5", "p99"]


# ─────────────────────────────────────────────────────────────────────────────
# Core picker
# ─────────────────────────────────────────────────────────────────────────────

def compute_tau_for_rate(entropies: Iterable[float],
                         target_rate: float = TARGET_RATE_MID,
                         fallback: float = 1.0) -> float:
    """Pick tau so that ~target_rate fraction of tokens trigger refinement.

    Per eq. (2), tau = the (1 - target_rate)-th quantile of `entropies`.
    Clamped to [MIN_TAU, MAX_TAU].
    """
    arr = np.asarray(list(entropies), dtype=np.float64)
    if arr.size == 0:
        return float(np.clip(fallback, MIN_TAU, MAX_TAU))
    q = 1.0 - float(target_rate)
    q = max(0.0, min(1.0, q))
    tau = float(np.quantile(arr, q))
    return float(np.clip(tau, MIN_TAU, MAX_TAU))


def in_target_band(rate: float,
                   low: float = TARGET_RATE_LOW,
                   high: float = TARGET_RATE_HIGH) -> bool:
    """True iff the observed refinement rate is inside the plan's band."""
    return low <= float(rate) <= high


def compute_quantiles(entropies: Iterable[float]) -> dict:
    """Return a labeled dict of entropy quantiles for diagnostics and for
    storage in run summaries. Matches QUANTILE_GRID / QUANTILE_KEYS."""
    arr = np.asarray(list(entropies), dtype=np.float64)
    if arr.size == 0:
        return {k: None for k in QUANTILE_KEYS}
    qs = np.quantile(arr, QUANTILE_GRID)
    return {k: float(v) for k, v in zip(QUANTILE_KEYS, qs)}


def tau_from_quantiles(quantiles: dict,
                       target_rate: float = TARGET_RATE_MID) -> Optional[float]:
    """Best-effort: extract tau from a quantile dict without raw entropies.

    Useful when we've already written a run summary and want the same
    tau without re-reading the flat entropy list.
    """
    if not quantiles:
        return None
    key = {
        0.05:  "p95",
        0.075: "p92_5",
        0.10:  "p90",
    }.get(round(target_rate, 3))
    if key and quantiles.get(key) is not None:
        return float(np.clip(quantiles[key], MIN_TAU, MAX_TAU))
    # Fall back to linear interpolation in the quantile grid.
    target_q = 1.0 - target_rate
    xs = QUANTILE_GRID
    ys = [quantiles.get(k) for k in QUANTILE_KEYS]
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        return None
    xs_, ys_ = zip(*pairs)
    tau = float(np.interp(target_q, xs_, ys_))
    return float(np.clip(tau, MIN_TAU, MAX_TAU))


# ─────────────────────────────────────────────────────────────────────────────
# Retry policy
# ─────────────────────────────────────────────────────────────────────────────

def next_tau_after_miss(run_entropies: Iterable[float],
                        observed_rate: float,
                        prev_tau: float,
                        target_rate: float = TARGET_RATE_MID) -> float:
    """Given an out-of-band run, pick the next tau.

    Preferred path: recompute tau as the target quantile of `run_entropies`.
    This is self-correcting — if the distribution has shifted slightly
    relative to L=0, the new tau naturally tracks it.

    Fallback (if run_entropies is empty for some reason): additive bump
    proportional to how far off the rate was.
    """
    arr = np.asarray(list(run_entropies), dtype=np.float64)
    if arr.size > 0:
        return compute_tau_for_rate(arr, target_rate)

    # Proportional fallback — rate too high ⇒ bump tau up; too low ⇒ down.
    step = 0.1 * max(1.0, abs(observed_rate - target_rate) / target_rate)
    delta = +step if observed_rate > target_rate else -step
    return float(np.clip(prev_tau + delta, MIN_TAU, MAX_TAU))


# ─────────────────────────────────────────────────────────────────────────────
# Calibration logging — one JSONL per (model, dataset)
# ─────────────────────────────────────────────────────────────────────────────

def calibration_log_path(calibration_dir: str,
                         model_name: str,
                         dataset_name: str) -> str:
    """Return the path to a per-(model, dataset) calibration log file."""
    safe_model = model_name.replace("/", "_")
    fname = f"{safe_model}__{dataset_name}.jsonl"
    return os.path.join(calibration_dir, fname)


def append_calibration_record(path: str, record: dict) -> None:
    """Append one record to the calibration log, creating the file if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def summarize_calibration(path: str) -> list:
    """Read the calibration log and return the list of records."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records