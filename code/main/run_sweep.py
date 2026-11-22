#!/usr/bin/env python
"""
Hands-free Phase 3 sweep orchestrator.

Flow (per model, per shell script)
──────────────────────────────────
  load model + tokenizer  (ONCE)
  for each dataset in the configured list (GSM8K last):
      ── L = 0 ───────────────────────────────────────────────────
      model.L = 0; run eval (this is the eval-base pass)
      collect per-token entropies; set tau = 92.5th percentile
        (=> initial aim: ~7.5% refinement rate)
      ── L = 1, 2, …, 10 ─────────────────────────────────────────
      for each L:
          for attempt in 1..MAX_ATTEMPTS:
              run eval with current tau
              if refinement_rate ∈ [5%, 10%]: accept, break
              else: recompute tau = quantile(this_run_entropies, 92.5%)
          if still out-of-band after MAX_ATTEMPTS: log WARN, accept last
      seed next dataset's starting tau from this dataset's final tau
       (minor optimization — tends to be close enough)

Why this works
──────────────
rate ≈ P(entropy_0 > tau), so tau = (1 - rate)-th quantile of the pre-
refinement entropy distribution. Because entropy_0 is computed BEFORE
the refinement loop, the trigger distribution barely shifts as L varies
— so tau picked from L=0 transfers to all higher Ls within the target
band in most cases, and converges in 1 extra attempt when it doesn't.

Efficiency
──────────
The frozen base model is loaded ONCE per shell script. Each subsequent
run mutates `model.L`, `model.tau_entropy`, and the config's dataset /
max_new_tokens fields in place. For 7 models × 6 datasets × 11 Ls this
saves ~60 model loads per shell script, or several GPU-hours per model.

Usage
─────
    python run_sweep.py --config configs/phase3_qwen2_5_3b.yaml \\
                        --datasets configs/phase3_datasets.yaml
"""

# Import config FIRST — its _setup_hf_cache() side-effect sets HF env vars
# before transformers is imported anywhere.

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

from config import Config, load_config, CACHE_DIR   # noqa: F401

import argparse
import json
import os
import sys
import time
from datetime import datetime

import yaml
import wandb

from utils      import get_logger, set_seed, attach_file_handler, RUN_TIMESTAMP, SCRIPT_NAME, build_run_tag
from data       import initialize_tokenizer
from model_eglr import load_eglr_model
from eval_eglr  import evaluate_eglr


def _flatten_dict(data: dict, prefix: str = "") -> dict:
    """Flatten nested config dicts so each leaf is a separate wandb field."""
    flat = {}
    for k, v in data.items():
        fk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, fk))
        else:
            flat[fk] = v
    return flat


def _wandb_init_for_attempt(cfg_dict, config, model, run_meta, use_wandb,
                            wandb_project, wandb_entity):
    """Start a wandb run for a single (dataset, L, attempt). Returns run_tag."""
    run_tag = build_run_tag(config, model)
    if not use_wandb:
        return run_tag

    wandb_config = _flatten_dict(cfg_dict)
    wandb_config["run_timestamp"] = RUN_TIMESTAMP
    wandb_config["script"]        = SCRIPT_NAME
    wandb_config["run_tag"]       = run_tag
    for k, v in run_meta.items():
        wandb_config[f"run_meta.{k}"] = v

    wandb_dir = getattr(config.parameters, "wandb_path", None)
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        project = wandb_project,
        entity  = wandb_entity,
        name    = run_tag,
        config  = wandb_config,
        dir     = wandb_dir,
        reinit  = True,
    )
    return run_tag


def _wandb_log_results(result, use_wandb):
    if not use_wandb or wandb.run is None:
        return
    entropy_stats = result.get("entropy_stats", {}) or {}
    quantiles     = entropy_stats.get("quantiles", {}) or {}
    depth_hist    = result.get("depth_histogram", {}) or {}

    metrics = {
        "results/accuracy":          result.get("accuracy"),
        "results/avg_depth":         result.get("avg_depth"),
        "results/refinement_rate":   result.get("refinement_rate"),
        "results/n_samples":         result.get("n_samples"),
        "results/elapsed_sec":       result.get("elapsed_sec"),
        "entropy/avg_all_tokens":    entropy_stats.get("avg_entropy_all_tokens"),
        "entropy/avg_refined_tokens": entropy_stats.get("avg_entropy_refined_tokens"),
        "entropy/n_tokens_seen":     entropy_stats.get("n_tokens_seen"),
        "entropy/n_tokens_refined":  entropy_stats.get("n_tokens_refined"),
    }
    for q_name, q_val in quantiles.items():
        metrics[f"entropy/quantile_{q_name}"] = q_val
    for depth, count in depth_hist.items():
        metrics[f"depth_histogram/d{depth}"] = count

    wandb.log({k: v for k, v in metrics.items() if v is not None})
    if result.get("summary_path") and os.path.exists(result["summary_path"]):
        wandb.save(result["summary_path"], policy="now")
    wandb.finish()
from calibrate  import (
    TARGET_RATE_LOW, TARGET_RATE_HIGH, TARGET_RATE_MID, DEFAULT_MAX_ATTEMPTS,
    compute_tau_for_rate, in_target_band, next_tau_after_miss,
    calibration_log_path, append_calibration_record,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Hands-free Phase 3 sweep with auto-calibrated tau_entropy."
    )
    p.add_argument("--config", type=str, required=True,
                   help="Per-model base config (sets model, eglr defaults, generation, prompts)")
    p.add_argument("--datasets", type=str, default="main/configs/phase3_datasets.yaml",
                   help="YAML listing Phase 3 datasets in order (GSM8K should be last)")
    p.add_argument("--L-list", type=str, default="0,1,2,3,4,5,6,7,8,9,10",
                   help="Comma-separated list of L values, in order")
    p.add_argument("--seed", type=int, default=1333)
    p.add_argument("--target-rate", type=float, default=TARGET_RATE_MID,
                   help="Target refinement rate (default midpoint of the 5-10%% band)")
    p.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                   help="Max attempts per (dataset, L) before giving up")
    p.add_argument("--log-dir", type=str, default="outputs/.logs/EGLR")
    p.add_argument("--calibration-dir", type=str, default="outputs/.calibration/EGLR")
    p.add_argument("--no-resume", action="store_true",
                   help="Do NOT skip (dataset, L) combinations that already have an accepted summary")
    p.add_argument("--calibrate", action="store_true",
                   help="Auto-tune tau_entropy from entropy distribution (old behavior). "
                        "Default: use the YAML's tau_entropy as-is with a single attempt per (dataset, L).")
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false", default=True,
                   help="Disable wandb logging")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="Overrides parameters.wandb_project from YAML")
    p.add_argument("--wandb-entity",  type=str, default=None,
                   help="Overrides WANDB_ENTITY env var if given")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Resume: detect already-finished (dataset, L) combinations
# ─────────────────────────────────────────────────────────────────────────────

def find_completed_runs(summary_dir: str, model_short: str, dataset_name: str) -> dict:
    """Scan the summaries dir for runs of (model, dataset). Return {L: summary_path}
    for in-band runs only; out-of-band ones are treated as unfinished."""
    completed = {}
    if not os.path.isdir(summary_dir):
        return completed
    for fname in os.listdir(summary_dir):
        if not fname.endswith("_summary.json"):
            continue
        if model_short not in fname:
            continue
        if f"_{dataset_name}" not in fname:
            continue
        path = os.path.join(summary_dir, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            L = int(data.get("eglr_hparams", {}).get("L", -1))
            rate = float(data.get("results", {}).get("avg_refinement_rate", -1.0))
            if L == 0:
                # L=0 is always "accepted" — no refinement gate.
                completed.setdefault(L, path)
            elif in_target_band(rate):
                # Keep the latest accepted run per L.
                if L not in completed or os.path.getmtime(path) > os.path.getmtime(completed[L]):
                    completed[L] = path
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return completed


# ─────────────────────────────────────────────────────────────────────────────
# Core sweep driver
# ─────────────────────────────────────────────────────────────────────────────

def run_one_attempt(model, tokenizer, config, L: int, tau: float,
                    attempt: int, dataset_name: str, model_short: str,
                    seed: int, config_path: str, log_path: str,
                    cfg_dict: dict = None, use_wandb: bool = True,
                    wandb_project: str = None, wandb_entity: str = None) -> dict:
    """Mutate model + config for this (L, tau), run evaluate_eglr, return its result."""
    model.L = int(L)
    # tau_entropy is only meaningful when L > 0; still set it for completeness.
    if tau is not None:
        model.tau_entropy = float(tau)

    run_meta = {
        "phase":       "phase3",
        "L":           L,
        "attempt":     attempt,
        "tau_entropy": float(model.tau_entropy),
        "seed":        seed,
        "config_path": os.path.abspath(config_path),
        "log_path":    log_path,
        "argv":        " ".join(sys.argv),
        "dataset":     dataset_name,
        "model_short": model_short,
    }

    _wandb_init_for_attempt(
        cfg_dict or {}, config, model, run_meta,
        use_wandb, wandb_project, wandb_entity,
    )

    try:
        result = evaluate_eglr(
            model, tokenizer, config,
            split_name    = "test",
            out_store     = True,
            output_format = "float",
            run_meta      = run_meta,
        )
    finally:
        _wandb_log_results(locals().get("result", {}), use_wandb)
    return result


def sweep_dataset(model, tokenizer, config, dataset_spec: dict, l_list: list,
                  calibration_path: str, target_rate: float, max_attempts: int,
                  resume_completed: dict, seed: int, config_path: str,
                  log_path: str, starting_tau: float = None,
                  cfg_dict: dict = None, use_wandb: bool = True,
                  wandb_project: str = None, wandb_entity: str = None,
                  calibrate: bool = False):
    """Run all L values on a single dataset. Returns the final tau used,
    so the next dataset can inherit it as a warm start."""

    # Override the config fields this dataset wants — data loaders + eval loop
    # will pick these up automatically via dot-access.
    config.dataset.name    = dataset_spec["name"]
    config.dataset.samples = int(dataset_spec["samples"])
    config.generation.max_new_tokens = int(dataset_spec["max_new_tokens"])

    dataset_name = config.dataset.name
    model_short  = config.model.name.split("/")[-1]

    logger.info("")
    logger.info("#" * 72)
    logger.info(f"# Dataset: {dataset_name}   samples={config.dataset.samples}   "
                f"max_new_tokens={config.generation.max_new_tokens}")
    logger.info("#" * 72)

    # ── L = 0 ────────────────────────────────────────────────────────────
    if 0 in l_list:
        if 0 in resume_completed:
            logger.info(f"[{dataset_name}] L=0  resume: {resume_completed[0]}")
            # Read entropies from the completed summary's quantiles → derive tau.
            with open(resume_completed[0]) as f:
                sdata = json.load(f)
            quantiles = sdata.get("results", {}).get("entropy_stats", {}).get("quantiles", {})
            # quantile p92.5 is exactly the 7.5% target tau.
            # (If target differs, tau_from_quantiles re-interpolates.)
            from calibrate import tau_from_quantiles
            current_tau = tau_from_quantiles(quantiles, target_rate) or (starting_tau or model.tau_entropy)
            l0_acc = sdata.get("results", {}).get("accuracy", None)
            logger.info(f"[{dataset_name}] L=0 accuracy (resumed): {l0_acc}")
        else:
            logger.info(f"[{dataset_name}] L=0  running eval-base")
            result = run_one_attempt(
                model, tokenizer, config,
                L=0, tau=(starting_tau or model.tau_entropy),
                attempt=0,
                dataset_name=dataset_name, model_short=model_short,
                seed=seed, config_path=config_path, log_path=log_path,
                cfg_dict=cfg_dict, use_wandb=use_wandb,
                wandb_project=wandb_project, wandb_entity=wandb_entity,
            )
            if calibrate:
                current_tau = compute_tau_for_rate(result["all_entropies"], target_rate)
                logger.info(f"[{dataset_name}] L=0 accuracy: {result['accuracy']:.4f}  "
                            f"→ picked tau_entropy={current_tau:.4f} from {(1-target_rate)*100:.1f}th quantile")
            else:
                # Fixed-tau mode: keep YAML's tau_entropy as-is.
                current_tau = starting_tau or model.tau_entropy
                logger.info(f"[{dataset_name}] L=0 accuracy: {result['accuracy']:.4f}  "
                            f"(fixed-tau mode — keeping YAML tau_entropy={current_tau:.4f})")
            append_calibration_record(calibration_path, {
                "phase":          "phase3",
                "model":          config.model.name,
                "dataset":        dataset_name,
                "L":              0,
                "attempt":        0,
                "tau_used":       None,
                "refinement_rate": result["refinement_rate"],
                "accuracy":       result["accuracy"],
                "in_band":        True,   # L=0 vacuously in-band
                "quantiles":      result["entropy_stats"]["quantiles"],
                "tau_for_next_L": current_tau,
                "summary_path":   result["summary_path"],
                "timestamp":      datetime.utcnow().isoformat(),
            })
    else:
        current_tau = starting_tau or model.tau_entropy

    # ── L = 1, 2, …, 10 ──────────────────────────────────────────────────
    for L in [l for l in l_list if l > 0]:
        if L in resume_completed:
            logger.info(f"[{dataset_name}] L={L}  resume: {resume_completed[L]}")
            continue

        logger.info("")
        logger.info(f"[{dataset_name}] L={L}  starting tau_entropy={current_tau:.4f}")

        last_rate, last_accuracy, last_summary_path = None, None, None
        tau_for_this_L = current_tau
        accepted = False
        effective_max_attempts = max_attempts if calibrate else 1

        for attempt in range(1, effective_max_attempts + 1):
            result = run_one_attempt(
                model, tokenizer, config,
                L=L, tau=tau_for_this_L, attempt=attempt,
                dataset_name=dataset_name, model_short=model_short,
                seed=seed, config_path=config_path, log_path=log_path,
                cfg_dict=cfg_dict, use_wandb=use_wandb,
                wandb_project=wandb_project, wandb_entity=wandb_entity,
            )
            rate = float(result["refinement_rate"])
            last_rate         = rate
            last_accuracy     = result["accuracy"]
            last_summary_path = result["summary_path"]
            in_band = in_target_band(rate)

            append_calibration_record(calibration_path, {
                "phase":          "phase3",
                "model":          config.model.name,
                "dataset":        dataset_name,
                "L":              L,
                "attempt":        attempt,
                "tau_used":       tau_for_this_L,
                "refinement_rate": rate,
                "accuracy":       result["accuracy"],
                "in_band":        in_band,
                "quantiles":      result["entropy_stats"]["quantiles"],
                "summary_path":   result["summary_path"],
                "timestamp":      datetime.utcnow().isoformat(),
            })

            if not calibrate:
                # Fixed-tau mode: single attempt, always accept.
                logger.info(f"[{dataset_name}] L={L}  "
                            f"rate={rate:.4f}  acc={result['accuracy']:.4f}  "
                            f"(fixed tau_entropy={tau_for_this_L:.4f})")
                accepted = True
                break

            if in_band:
                logger.info(f"[{dataset_name}] L={L}  attempt {attempt}: "
                            f"rate={rate:.4f} ∈ [{TARGET_RATE_LOW}, {TARGET_RATE_HIGH}] ✓  "
                            f"acc={result['accuracy']:.4f}")
                accepted = True
                current_tau = tau_for_this_L  # carry into next L
                break

            # Out of band — recompute tau from this run's distribution and retry.
            new_tau = next_tau_after_miss(
                run_entropies=result["all_entropies"],
                observed_rate=rate,
                prev_tau=tau_for_this_L,
                target_rate=target_rate,
            )
            direction = "too high" if rate > TARGET_RATE_HIGH else "too low"
            logger.warning(
                f"[{dataset_name}] L={L}  attempt {attempt}: rate={rate:.4f} {direction}  "
                f"tau {tau_for_this_L:.4f} → {new_tau:.4f} (retrying)"
            )
            tau_for_this_L = new_tau

        if calibrate and not accepted:
            logger.error(
                f"[{dataset_name}] L={L}  FAILED to land in [{TARGET_RATE_LOW}, "
                f"{TARGET_RATE_HIGH}] after {max_attempts} attempts. "
                f"Last rate={last_rate:.4f}, accuracy={last_accuracy:.4f}. Keeping last result."
            )
            current_tau = tau_for_this_L

    return current_tau


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    # ── File logging ────────────────────────────────────────────────
    log_path = attach_file_handler(os.path.join(args.log_dir, f"{SCRIPT_NAME}_{RUN_TIMESTAMP}.log"))
    logger.info(f"Log file: {log_path}")

    # ── Base config ─────────────────────────────────────────────────
    cfg_dict = load_config(args.config)
    config   = Config(cfg_dict)
    model_short = config.model.name.split("/")[-1]
    logger.info(f"Model: {config.model.name}")

    # ── Dataset list ────────────────────────────────────────────────
    with open(args.datasets, "r") as f:
        datasets_yaml = yaml.safe_load(f)
    datasets: list = datasets_yaml["datasets"]

    # Enforce "GSM8K last" invariant regardless of YAML order.
    non_gsm8k = [d for d in datasets if not d["name"].startswith("gsm8k")]
    gsm8k     = [d for d in datasets if d["name"].startswith("gsm8k")]
    datasets  = non_gsm8k + gsm8k

    logger.info("Dataset order (gsm8k forced to end):")
    for i, d in enumerate(datasets, start=1):
        logger.info(f"  {i}. {d['name']}  samples={d['samples']}  mnt={d['max_new_tokens']}")

    # ── L list ──────────────────────────────────────────────────────
    l_list = [int(x.strip()) for x in args.L_list.split(",") if x.strip()]
    logger.info(f"L values (in order): {l_list}")

    # ── Model + tokenizer (ONCE) ────────────────────────────────────
    logger.info("Loading frozen base model for EGLR (this happens once)...")
    model = load_eglr_model(config)
    model.eval()
    tokenizer, _pad_is_eos = initialize_tokenizer(config, for_generation=True)

    # ── Sweep ───────────────────────────────────────────────────────
    t0 = time.time()
    starting_tau = getattr(config.eglr, "tau_entropy", None)  # YAML fallback
    _model_short_sweep = config.model.name.split("/")[-1]
    summary_dir  = os.path.join(config.parameters.checkpoint_path, "main_eglr", "eglr_summaries", _model_short_sweep)
    # ^ Note: evaluate_eglr uses SCRIPT_NAME for the subdir. When run_sweep.py
    # is the entry point, SCRIPT_NAME == "run_sweep", not "main_eglr". So we
    # honor whatever SCRIPT_NAME is currently set to for resume detection.
    # Per-dataset subdir is appended inside the loop below.
    summary_dir = os.path.join(config.parameters.checkpoint_path, SCRIPT_NAME, "eglr_summaries", _model_short_sweep)

    for ds_spec in datasets:
        dataset_name = ds_spec["name"]
        calibration_path = calibration_log_path(
            args.calibration_dir, config.model.name, dataset_name
        )

        # Resume detection — skip (dataset, L) combos that already converged.
        if args.no_resume:
            resume_completed: dict = {}
        else:
            resume_completed = find_completed_runs(
                os.path.join(summary_dir, dataset_name), model_short, dataset_name
            )
            if resume_completed:
                logger.info(f"[{dataset_name}] resume found {len(resume_completed)} accepted runs: "
                            f"L={sorted(resume_completed.keys())}")

        starting_tau = sweep_dataset(
            model, tokenizer, config,
            dataset_spec     = ds_spec,
            l_list           = l_list,
            calibration_path = calibration_path,
            target_rate      = args.target_rate,
            max_attempts     = args.max_attempts,
            resume_completed = resume_completed,
            seed             = args.seed,
            config_path      = args.config,
            log_path         = log_path,
            starting_tau     = starting_tau,
            cfg_dict         = cfg_dict,
            use_wandb        = args.use_wandb,
            wandb_project    = args.wandb_project,
            wandb_entity     = args.wandb_entity,
            calibrate        = args.calibrate,
        )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"SWEEP DONE — total elapsed: {elapsed/60.0:.1f} min "
                f"({len(datasets)} datasets × {len(l_list)} L values, +retries)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()