"""
Evaluation driver for Entropy-Gated Latent Recursion (EGLR).

Extensions over the Phase 1 version
───────────────────────────────────
1. Adds `entropy_stats.quantiles` to the summary JSON (p50, p75, p80, p85,
   p90, p92.5, p95, p97.5, p99). The 92.5th percentile is the natural
   starting point for tau_entropy when targeting a ~7.5% refinement rate.

2. Returns the flat list of per-token entropies (`all_entropies`) in the
   result dict so the orchestrator (run_sweep.py) can recalibrate tau
   without re-reading the JSONL.

3. Emits `run_meta["attempt"]` into the summary when an orchestrator
   provides it — useful for auditing which attempt of a retry sequence
   produced the winning rate.

Everything else (dataset/extraction dispatch, per-sample JSONL, batched
generation) is identical to the Phase 1 version.
"""

import os
import json
import time
import torch
from tqdm.auto import tqdm
from datetime import datetime

from utils import get_logger, UAE_TZ, SCRIPT_NAME, build_run_tag
from data  import get_dataset
from eval  import _get_extractor, _compare_answers, log_extraction_info
from extraction import get_task_description, lm_eval_status
from calibrate import compute_quantiles

logger = get_logger(__name__)


def evaluate_eglr(
    model,
    tokenizer,
    config,
    split_name: str = "test",
    out_store: bool = True,
    output_format: str = "float",
    run_meta: dict = None,
    num_beams: int = 1,
):
    """
    Run EGLR generation over `split_name` of the configured dataset.

    When model.L == 0 the refinement loop is a no-op, so this function
    doubles as the eval-base driver. The per-token entropies it records
    are identical to what a plain forward pass would see and are the
    correct calibration source for choosing tau_entropy.
    """
    logger.info(f"Running EGLR evaluation on {config.dataset.name} (split: {split_name})")
    log_extraction_info(config)
    extract_answer = _get_extractor(config)
    model.eval()

    device  = next(model.parameters()).device
    dataset = get_dataset(config.dataset.name, split=split_name, config=config)

    is_instruct   = getattr(config.model, "is_instruct", True)
    system_prompt = getattr(config.prompts, "system_eval", "") if hasattr(config, "prompts") else ""

    # ── Output path setup ────────────────────────────────────────────
    ts      = datetime.now(UAE_TZ).strftime("%Y%m%d_%H%M%S")
    run_tag = build_run_tag(config, model, ts=ts)

    per_sample_path = None
    summary_path    = None
    out_f           = None

    if out_store:
        model_short = config.model.name.split("/")[-1]
        dataset_name = config.dataset.name
        out_dir = os.path.join(config.parameters.checkpoint_path, SCRIPT_NAME, "eglr_outputs",
                               model_short, dataset_name)
        sum_dir = os.path.join(config.parameters.checkpoint_path, SCRIPT_NAME, "eglr_summaries",
                               model_short, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(sum_dir, exist_ok=True)
        per_sample_path = os.path.join(out_dir, f"{run_tag}.jsonl")
        summary_path    = os.path.join(sum_dir, f"{run_tag}_summary.json")
        logger.info(f"Per-sample JSONL → {per_sample_path}")
        logger.info(f"Summary JSON     → {summary_path}")
        out_f = open(per_sample_path, "w", encoding="utf-8")

    # ── Running stats ─────────────────────────────────────────────────
    correct, total  = 0, 0
    depth_sum, depth_count = 0, 0
    refine_rate_sum = 0.0
    depth_histogram = {}

    entropy_sum, entropy_n      = 0.0, 0
    entropy_refined_sum, entropy_refined_n = 0.0, 0
    total_gen_tokens = 0
    # Flat list across all samples × tokens — used for quantiles + calibration.
    all_entropies: list = []

    samples_cfg = getattr(config.dataset, "samples", -1)
    # samples <= 0 (or unset) → use the full dataset
    n_samples_target = (
        len(dataset) if samples_cfg is None or samples_cfg <= 0
        else min(samples_cfg, len(dataset))
    )

    t_start = time.time()

    batch_size = max(int(getattr(config.generation, "test_batch_size", 1)), 1)
    logger.info(f"Running EGLR eval at batch_size={batch_size}  L={model.L}  tau_E={model.tau_entropy}")

    def _build_prompt(question: str) -> str:
        if is_instruct:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ]
            messages = [m for m in messages if m["content"]]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"Problem: {question}\nSolution:"

    try:
        indices = list(range(n_samples_target))
        with tqdm(range(0, n_samples_target, batch_size), desc="EGLR eval") as tbar:
            for batch_start in tbar:
                batch_idxs = indices[batch_start : batch_start + batch_size]
                examples  = [dataset[i] for i in batch_idxs]
                questions = [ex["question"] for ex in examples]
                prompts   = [_build_prompt(q) for q in questions]

                enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                ).to(device)

                if batch_start == 0:
                    logger.info("\n" + "=" * 60)
                    logger.info("=== SAMPLE PROMPT PASSED TO MODEL (EGLR EVAL, batch[0]) ===")
                    logger.info(tokenizer.decode(enc.input_ids[0], skip_special_tokens=False))
                    logger.info("=" * 60 + "\n")

                prompt_len = enc.input_ids.shape[1]

                with torch.no_grad():
                    if num_beams > 1:
                        # HuggingFace beam search baseline (no EGLR refinement).
                        # Returns LongTensor [B, prompt_len + new_tokens].
                        gen_ids = model.base_model.generate(
                            input_ids=enc.input_ids,
                            attention_mask=enc.attention_mask,
                            max_new_tokens=config.generation.max_new_tokens,
                            num_beams=num_beams,
                            do_sample=False,
                            early_stopping=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        # Match the dict shape generate_eglr returns. Beam runs
                        # have no per-token entropies/depths — fill with zeros.
                        bsz = gen_ids.shape[0]
                        out = {
                            "generated":           gen_ids,
                            "per_token_depths":    [
                                [0] * (gen_ids.shape[1] - prompt_len)
                                for _ in range(bsz)
                            ],
                            "per_token_entropies": [
                                [0.0] * (gen_ids.shape[1] - prompt_len)
                                for _ in range(bsz)
                            ],
                        }
                    else:
                        out = model.generate_eglr(
                            input_ids=enc.input_ids,
                            attention_mask=enc.attention_mask,
                            max_new_tokens=config.generation.max_new_tokens,
                            tokenizer=tokenizer,
                            do_sample=config.generation.do_sample,
                            temperature=config.generation.temperature,
                            top_p=config.generation.top_p,
                            top_k=config.generation.top_k,
                        )

                for b, idx in enumerate(batch_idxs):
                    question     = questions[b]
                    raw_answer   = examples[b]["answer"]
                    new_tokens   = out["generated"][b][prompt_len:]
                    decoded      = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    prediction   = extract_answer(decoded)
                    ground_truth = extract_answer(raw_answer)
                    is_correct   = _compare_answers(prediction, ground_truth, output_format, config=config)

                    correct += int(is_correct)
                    total   += 1

                    depths    = out["per_token_depths"][b]
                    entropies = out["per_token_entropies"][b]
                    n_gen     = len(depths)
                    total_gen_tokens += n_gen
                    avg_depth_s = sum(depths) / max(n_gen, 1)
                    max_depth_s = max(depths) if depths else 0
                    depth_sum   += sum(depths)
                    depth_count += n_gen
                    per_sample_refine_rate = sum(1 for d in depths if d > 0) / max(n_gen, 1)
                    refine_rate_sum += per_sample_refine_rate
                    for d in depths:
                        depth_histogram[d] = depth_histogram.get(d, 0) + 1

                    entropy_sum += sum(entropies)
                    entropy_n   += len(entropies)
                    all_entropies.extend(float(e) for e in entropies)
                    for e, d in zip(entropies, depths):
                        if d > 0:
                            entropy_refined_sum += e
                            entropy_refined_n   += 1

                    if out_f is not None:
                        record = {
                            "idx":                  idx,
                            "question":             question,
                            "prediction":           prediction,
                            "ground_truth":         ground_truth,
                            "correct":              is_correct,
                            "decoded":              decoded,
                            "n_gen_tokens":         n_gen,
                            "avg_depth":            avg_depth_s,
                            "max_depth":            max_depth_s,
                            "refinement_rate":      per_sample_refine_rate,
                            "per_token_depths":     depths,
                            "per_token_entropies":  [round(float(x), 4) for x in entropies],
                        }
                        out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
                        out_f.flush()

                    if idx < 3:
                        logger.info(
                            f"[Sample {idx+1}] GT: {ground_truth} | Pred: {prediction} | "
                            f"correct={is_correct} | avg_depth={avg_depth_s:.2f} "
                            f"refine_rate={per_sample_refine_rate:.3f}"
                        )
                        logger.info(f"Tail: ...{decoded[-120:].replace(chr(10), ' ')}")

                tbar.set_postfix({
                    "acc":    f"{correct/max(total,1):.3f}",
                    "depth":  f"{depth_sum/max(depth_count,1):.2f}",
                    "refine": f"{refine_rate_sum/max(total,1):.3f}",
                })
    finally:
        if out_f is not None:
            out_f.close()

    elapsed = time.time() - t_start

    accuracy          = correct / total if total > 0 else 0.0
    avg_depth_overall = depth_sum / max(depth_count, 1)
    avg_refine_rate   = refine_rate_sum / max(total, 1)
    depth_hist_sorted = {str(k): v for k, v in sorted(depth_histogram.items())}

    entropy_stats = {
        "avg_entropy_all_tokens":     entropy_sum / max(entropy_n, 1),
        "avg_entropy_refined_tokens": entropy_refined_sum / max(entropy_refined_n, 1)
                                      if entropy_refined_n > 0 else None,
        "n_tokens_seen":              entropy_n,
        "n_tokens_refined":           entropy_refined_n,
        "quantiles":                  compute_quantiles(all_entropies),
    }

    logger.info("=" * 60)
    logger.info("EGLR evaluation complete")
    logger.info(f"  Accuracy              : {correct}/{total} = {accuracy:.4f}")
    logger.info(f"  Avg refinement depth  : {avg_depth_overall:.3f}")
    logger.info(f"  Avg refinement rate   : {avg_refine_rate:.4f}")
    logger.info(f"  Depth histogram       : {depth_hist_sorted}")
    logger.info(f"  Total gen tokens      : {total_gen_tokens}")
    logger.info(f"  Entropy p92.5         : {entropy_stats['quantiles'].get('p92_5')}")
    logger.info(f"  Elapsed               : {elapsed:.1f}s ({elapsed/max(total,1):.2f}s/sample)")
    logger.info("=" * 60)

    # ── Write summary JSON ────────────────────────────────────────
    if summary_path is not None:
        summary = {
            "timestamp":  ts,
            "script":     SCRIPT_NAME,
            "elapsed_sec": elapsed,
            "model":      config.model.name,
            "dataset":    config.dataset.name,
            "split":      split_name,
            "n_samples":  total,
            "extraction_info": {
                "task_description": get_task_description(config.dataset.name),
                "lm_eval_status":   lm_eval_status(),
            },
            "eglr_hparams": {
                "L":           model.L,
                "alpha":       model.alpha,
                "K_max":       model.K_max,
                "tau_entropy": model.tau_entropy,
                "tau_kl":      model.tau_kl,
                "norm_match":  model.norm_match,
                "start_layer_idx": model.start_layer_idx,
                "total_layers":    model.total_layers,
            },
            "generation_hparams": {
                "max_new_tokens": config.generation.max_new_tokens,
                "do_sample":      config.generation.do_sample,
                "temperature":    config.generation.temperature,
                "top_p":          config.generation.top_p,
                "top_k":          config.generation.top_k,
            },
            "results": {
                "accuracy":           accuracy,
                "correct":            correct,
                "total":              total,
                "avg_depth":          avg_depth_overall,
                "avg_refinement_rate": avg_refine_rate,
                "depth_histogram":    depth_hist_sorted,
                "total_gen_tokens":   total_gen_tokens,
                "entropy_stats":      entropy_stats,
            },
            "artifacts": {
                "per_sample_path": per_sample_path,
            },
        }
        if run_meta:
            summary["run_meta"] = run_meta
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        logger.info(f"Wrote summary → {summary_path}")

    return {
        "accuracy":        accuracy,
        "avg_depth":       avg_depth_overall,
        "refinement_rate": avg_refine_rate,
        "depth_histogram": depth_hist_sorted,
        "n_samples":       total,
        "entropy_stats":   entropy_stats,
        "all_entropies":   all_entropies,   # kept in-memory for the orchestrator
        "per_sample_path": per_sample_path,
        "summary_path":    summary_path,
        "elapsed_sec":     elapsed,
    }