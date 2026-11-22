#!/usr/bin/env python
"""
Entry point for Entropy-Gated Latent Recursion (EGLR) evaluation.

Usage
-----
    # Full eval
    python main_eglr.py --config configs/math_qwen_3b_eglr.yaml

    # Quick smoke test (50 problems — weekend manifold-sanity run)
    python main_eglr.py --config configs/math_qwen_3b_eglr.yaml --samples 50

    # Override EGLR hyperparameters from the command line
    python main_eglr.py --config configs/math_qwen_3b_eglr.yaml \
        --L 2 --alpha 0.15 --K_max 3 --tau_entropy 1.0 --norm_match true

Artifacts per run (all timestamped)
-----------------------------------
  logs/main_eglr_<timestamp>.log
      Full run log (everything written to console is mirrored here).

  <output_dir>/main_eglr/eglr_outputs/<timestamp>_<model>_<cfg>.jsonl
      Per-sample generations + per-token depths + per-token entropies.

  <output_dir>/main_eglr/eglr_summaries/<timestamp>_<model>_<cfg>_summary.json
      Run summary — accuracy, depth histogram, entropy stats, all hparams.
"""

# IMPORTANT: import `config` first — its _setup_hf_cache() side-effect sets
# HF_HOME / HUGGINGFACE_HUB_CACHE etc. before transformers is imported anywhere.

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
import os
import sys

import torch
import wandb

from utils      import get_logger, set_seed, attach_file_handler, RUN_TIMESTAMP, SCRIPT_NAME, build_run_tag, write_resolved_config
from data       import initialize_tokenizer
from model_eglr import load_eglr_model
from eval_eglr  import evaluate_eglr

logger = get_logger(__name__)


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


def _override_from_args(cfg_dict: dict, args) -> dict:
    """Apply CLI overrides on top of the YAML-loaded dict."""
    if args.model is not None:
        cfg_dict.setdefault("model", {})["name"] = args.model
    if args.dataset is not None:
        cfg_dict.setdefault("dataset", {})["name"] = args.dataset
    if args.samples is not None:
        cfg_dict.setdefault("dataset", {})["samples"] = args.samples
    if args.max_new_tokens is not None:
        cfg_dict.setdefault("generation", {})["max_new_tokens"] = args.max_new_tokens
    if args.system_prompt is not None:
        cfg_dict.setdefault("prompts", {})["system_eval"] = args.system_prompt
    if args.device_map is not None:
        cfg_dict.setdefault("model", {})["device_map"] = args.device_map

    if args.test_batch_size is not None:
        cfg_dict.setdefault("generation", {})["test_batch_size"] = args.test_batch_size
    if args.temperature is not None:
        gen = cfg_dict.setdefault("generation", {})
        gen["temperature"] = args.temperature
        # Auto-enable sampling when a non-zero temperature is requested
        if args.temperature > 0:
            gen["do_sample"] = True
    if args.top_k is not None:
        cfg_dict.setdefault("generation", {})["top_k"] = args.top_k
    if args.top_p is not None:
        cfg_dict.setdefault("generation", {})["top_p"] = args.top_p
    if args.do_sample is not None:
        cfg_dict.setdefault("generation", {})["do_sample"] = args.do_sample

    eglr = cfg_dict.setdefault("eglr", {})
    if args.L           is not None: eglr["L"]           = args.L
    if args.alpha       is not None: eglr["alpha"]       = args.alpha
    if args.K_max       is not None: eglr["K_max"]       = args.K_max
    if args.tau_entropy is not None: eglr["tau_entropy"] = args.tau_entropy
    if args.tau_kl      is not None: eglr["tau_kl"]      = args.tau_kl
    if args.norm_match  is not None: eglr["norm_match"]  = args.norm_match
    if args.kv_cache    is not None: eglr["kv_cache"]    = args.kv_cache

    return cfg_dict


def _setup_run_logging(log_dir: str) -> str:
    """Create a timestamped .log file and attach a file handler to the 'eglr' logger."""
    log_path = os.path.join(log_dir, f"{SCRIPT_NAME}_{RUN_TIMESTAMP}.log")
    return attach_file_handler(log_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run Entropy-Gated Latent Recursion (EGLR) evaluation."
    )
    parser.add_argument("--config",  type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed",    type=int, default=1333)
    parser.add_argument("--model", type=str, default=None,
                        help="Override model.name (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset.name (e.g., math_500, gsm8k, aime24)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Override dataset.samples (-1 for full test set)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override generation.max_new_tokens")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Override prompts.system_eval (per-dataset prompt from .sh)")
    parser.add_argument("--device_map", type=str, default=None,
                        help="Override model.device_map (e.g., 'auto' for multi-GPU sharding)")
    parser.add_argument("--output-format", choices=["int", "float"], default="float")
    parser.add_argument("--no-out-store", dest="out_store",
                        action="store_false", default=True,
                        help="Disable per-sample JSONL + summary JSON output")
    parser.add_argument("--log-dir", type=str, default="outputs/.logs/EGLR",
                        help="Directory for run log files")
    parser.add_argument("--no-wandb", dest="use_wandb",
                        action="store_false", default=True,
                        help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Overrides parameters.wandb_project from YAML")
    parser.add_argument("--wandb-entity",  type=str, default=None,
                        help="Overrides WANDB_ENTITY env var if given")

    # EGLR hyperparameter overrides
    parser.add_argument("--L",           type=int,   default=None)
    parser.add_argument("--alpha",       type=float, default=None)
    parser.add_argument("--K_max",       type=int,   default=None)
    parser.add_argument("--tau_entropy", type=float, default=None)
    parser.add_argument("--tau_kl",      type=float, default=None)
    parser.add_argument("--norm_match",  type=lambda s: s.lower() in ("1", "true", "yes"),
                        default=None)
    parser.add_argument("--kv_cache",    choices=["actual", "modified"], default=None,
                        help="actual = restore original K/V (Fix B); modified = keep refined K/V")
    parser.add_argument("--test_batch_size", type=int, default=None,
                        help="Override generation.test_batch_size")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (auto-enables do_sample when > 0)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling cutoff")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p (nucleus) sampling cutoff")
    parser.add_argument("--do_sample", type=lambda s: s.lower() in ("1", "true", "yes"),
                        default=None, help="Enable/disable sampling explicitly")

    args = parser.parse_args()
    set_seed(args.seed)

    # ── File logging ─────────────────────────────────────────────────
    log_path = _setup_run_logging(args.log_dir)
    logger.info(f"Log file: {log_path}")

    # ── Config ───────────────────────────────────────────────────────
    cfg_dict = load_config(args.config)
    cfg_dict = _override_from_args(cfg_dict, args)
    config   = Config(cfg_dict)

    if not hasattr(config, "eglr"):
        logger.error("Config is missing an 'eglr' section. See configs/math_qwen_3b_eglr.yaml.")
        sys.exit(1)

    logger.info(f"=== EGLR run — config: {args.config} ===")
    logger.info(
        f"Model: {config.model.name} | Dataset: {config.dataset.name} | "
        f"Samples: {config.dataset.samples} | Seed: {args.seed}"
    )
    logger.info(
        f"EGLR: L={cfg_dict['eglr'].get('L')} "
        f"alpha={cfg_dict['eglr'].get('alpha')} "
        f"K_max={cfg_dict['eglr'].get('K_max')} "
        f"tau_entropy={cfg_dict['eglr'].get('tau_entropy')} "
        f"tau_kl={cfg_dict['eglr'].get('tau_kl')} "
        f"norm_match={cfg_dict['eglr'].get('norm_match')}"
    )

    # ── Model + tokenizer ────────────────────────────────────────────
    model = load_eglr_model(config)
    model.eval()

    logger.info("Initializing tokenizer")
    tokenizer, _pad_is_eos = initialize_tokenizer(config, for_generation=True)

    # ── wandb (after model load so run name can use model.L / alpha / etc.) ──
    run_tag = build_run_tag(config, model)  # experiment ID shared with JSONL + summary

    # ── Persist resolved config alongside run artifacts ──────────────
    _model_short  = config.model.name.split("/")[-1]
    _dataset_name = config.dataset.name
    resolved_cfg_path = os.path.join(
        config.parameters.checkpoint_path, SCRIPT_NAME, "resolved_configs",
        _model_short, _dataset_name, f"{run_tag}.yaml"
    )
    write_resolved_config(cfg_dict, resolved_cfg_path)
    logger.info(f"Resolved config → {resolved_cfg_path}")
    if args.use_wandb:
        wandb_config = _flatten_dict(cfg_dict)
        wandb_config["run_timestamp"] = RUN_TIMESTAMP
        wandb_config["script"]        = SCRIPT_NAME
        wandb_config["seed"]          = args.seed
        wandb_config["run_tag"]       = run_tag
        wandb_dir = getattr(config.parameters, "wandb_path", None)
        if wandb_dir:
            os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = run_tag,
            config  = wandb_config,
            dir     = wandb_dir,
            reinit  = True,
        )
        logger.info(f"wandb run: {wandb.run.url if wandb.run else '(disabled)'}")

    # ── Eval ─────────────────────────────────────────────────────────
    run_meta = {
        "config_path": os.path.abspath(args.config),
        "log_path":    log_path,
        "seed":        args.seed,
        "argv":        " ".join(sys.argv),
    }
    results = evaluate_eglr(
        model, tokenizer, config,
        split_name    = "test",
        out_store     = args.out_store,
        output_format = args.output_format,
        run_meta      = run_meta,
    )

    logger.info(
        f"DONE — accuracy={results['accuracy']:.4f}  "
        f"avg_depth={results['avg_depth']:.3f}  "
        f"refinement_rate={results['refinement_rate']:.4f}  "
        f"n={results['n_samples']}"
    )
    if results.get("summary_path"):
        logger.info(f"Summary:     {results['summary_path']}")
    if results.get("per_sample_path"):
        logger.info(f"Per-sample:  {results['per_sample_path']}")
    logger.info(f"Log:         {log_path}")

    # ── Log to wandb ─────────────────────────────────────────────────
    if args.use_wandb and wandb.run is not None:
        entropy_stats = results.get("entropy_stats", {}) or {}
        quantiles     = entropy_stats.get("quantiles", {}) or {}
        depth_hist    = results.get("depth_histogram", {}) or {}

        wandb_metrics = {
            "results/accuracy":          results["accuracy"],
            "results/avg_depth":         results["avg_depth"],
            "results/refinement_rate":   results["refinement_rate"],
            "results/n_samples":         results["n_samples"],
            "results/elapsed_sec":       results.get("elapsed_sec"),
            "entropy/avg_all_tokens":    entropy_stats.get("avg_entropy_all_tokens"),
            "entropy/avg_refined_tokens": entropy_stats.get("avg_entropy_refined_tokens"),
            "entropy/n_tokens_seen":     entropy_stats.get("n_tokens_seen"),
            "entropy/n_tokens_refined":  entropy_stats.get("n_tokens_refined"),
        }
        for q_name, q_val in quantiles.items():
            wandb_metrics[f"entropy/quantile_{q_name}"] = q_val
        for depth, count in depth_hist.items():
            wandb_metrics[f"depth_histogram/d{depth}"] = count

        wandb.log({k: v for k, v in wandb_metrics.items() if v is not None})

        if results.get("summary_path") and os.path.exists(results["summary_path"]):
            wandb.save(results["summary_path"], policy="now")

        wandb.finish()


if __name__ == "__main__":
    main()