"""
Shared evaluation helpers + base-model evaluation driver.

EGLR-specific evaluation lives in eval_eglr.py and imports from this module.
"""

import os
import json
import torch
import wandb
from tqdm.auto import tqdm
from datetime import datetime

from utils import get_logger, extract_answer_number, extract_boxed_answer, UAE_TZ, SCRIPT_NAME
from data  import get_dataset, get_dataloader
from extraction import (
    DATASET_HANDLERS,
    get_extractor as _ext_get_extractor,
    get_comparator as _ext_get_comparator,
    get_task_description,
    lm_eval_status,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-aware dispatch. eval.py now delegates to extraction.DATASET_HANDLERS
# so adding a new dataset is a one-liner there.
# ─────────────────────────────────────────────────────────────────────────────

def _dataset_name(config):
    return getattr(getattr(config, "dataset", None), "name", "") or ""


def _get_extractor(config):
    """Return the correct answer-extraction function for the configured dataset."""
    name = _dataset_name(config)
    fn = _ext_get_extractor(name)
    if fn is not None:
        return fn
    # Back-compat fallback — never hit for any dataset in DATASET_HANDLERS
    logger.warning(f"No extraction handler for dataset='{name}'; falling back to utils.extract_answer_number")
    return extract_answer_number


def _get_comparator(config):
    """Return the correct comparison function for the configured dataset."""
    name = _dataset_name(config)
    fn = _ext_get_comparator(name)
    if fn is not None:
        return fn
    # Default: numeric comparison via legacy logic
    def _legacy(pred, gt, output_format="float"):
        return _compare_answers(pred, gt, output_format, config=None)
    return _legacy


def numeric_answers_match(prediction_numeric, ground_truth_numeric, decimals=1):
    """Compare numeric answers safely without float16 overflow."""
    pred_tensor = torch.round(torch.tensor(float(prediction_numeric), dtype=torch.float64), decimals=decimals)
    gt_tensor   = torch.round(torch.tensor(float(ground_truth_numeric), dtype=torch.float64), decimals=decimals)
    return (pred_tensor == gt_tensor).item()


def _compare_answers(prediction, ground_truth, output_format="float", config=None):
    """
    Flexible answer comparison.

    If `config` is provided and its dataset has a registered comparator in
    extraction.DATASET_HANDLERS, dispatch there (lm-eval-aligned).

    Otherwise fall back to the legacy numeric-or-string logic (preserved so
    evaluate_base keeps working on datasets that don't have a handler).
    """
    if prediction is None or ground_truth is None:
        return False

    if config is not None:
        name = _dataset_name(config)
        fn = _ext_get_comparator(name)
        if fn is not None:
            return fn(prediction, ground_truth, output_format)

    # ── Legacy fallback ──
    if isinstance(prediction, float) and isinstance(ground_truth, float):
        if output_format == 'int':
            return int(round(prediction)) == int(round(ground_truth))
        return numeric_answers_match(prediction, ground_truth)

    def _norm(v):
        s = str(v).lower().strip()
        s = s.replace(' ', '').replace('\\,', '').replace('\\!', '')
        return s

    return _norm(prediction) == _norm(ground_truth)


def log_extraction_info(config):
    """One-liner at run start: which extractor / comparator / lm_eval source."""
    name = _dataset_name(config)
    desc = get_task_description(name)
    status = lm_eval_status()
    logger.info(
        f"Extraction: dataset={name} task={desc} "
        f"lm_eval_source={status['source_module'] or 'not_available'} "
        f"is_equiv={status['is_equiv']} normalize={status['normalize_final_answer']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Base-model eval driver (unchanged flow, now dataset-aware via dispatch)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_base(model, tokenizer, config, split_name="test", global_step=0,
                  out_store=True, mode="eval-base", output_format="float"):
    """Plain base-model evaluation (HuggingFace generate)."""
    logger.info(f"Running Base Model Evaluation on {config.dataset.name} (split: {split_name})...")
    log_extraction_info(config)
    extract_answer = _get_extractor(config)
    model.eval()

    eval_loader = get_dataloader(tokenizer, batch_size=config.generation.test_batch_size, split=split_name, config=config)
    dataset = get_dataset(config.dataset.name, split=split_name, config=config)

    correct, total, sample_idx = 0, 0, 0
    out_f = None
    if out_store:
        _ms = config.model.name.split("/")[-1]
        _ds = config.dataset.name
        out_dir = os.path.join(config.parameters.checkpoint_path, SCRIPT_NAME, "generated_outputs", _ms, _ds)
        os.makedirs(out_dir, exist_ok=True)

        model_short = config.model.name.split('/')[-1]
        checkpoint_identifier = config.generation.eval_checkpoint if mode == "eval" else "latest_training"

        out_path = os.path.join(
            out_dir,
            f"{datetime.now(UAE_TZ).strftime('%Y%m%d_%H%M%S')}_{model_short}_{checkpoint_identifier}_{mode}_s{config.dataset.samples}_bs{config.generation.test_batch_size}_"
            f"temp{config.generation.temperature}_topp{config.generation.top_p}_topk{config.generation.top_k}.jsonl"
        )
        logger.info(f"Writing generations to {out_path}...")
        out_f = open(out_path, "w", encoding="utf-8")

    try:
        with tqdm(eval_loader, desc="Accuracy Gen") as tepoch:
            for batch_idx, (inputs, _) in enumerate(tepoch):
                input_ids      = inputs["input_ids"].to(model.device)
                attention_mask = inputs["attention_mask"].to(model.device)

                if batch_idx == 0:
                    logger.info("\n" + "="*60)
                    logger.info("=== SAMPLE PROMPT PASSED TO MODEL (BASE EVAL) ===")
                    logger.info(tokenizer.decode(input_ids[0], skip_special_tokens=False))
                    logger.info("="*60 + "\n")

                prompt_len = input_ids.shape[1]

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        max_new_tokens=config.generation.max_new_tokens,
                        do_sample=config.generation.do_sample,
                        temperature=config.generation.temperature,
                        top_p=config.generation.top_p, top_k=config.generation.top_k,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
                    )

                batch_size_actual = input_ids.size(0)
                for i in range(batch_size_actual):
                    if sample_idx >= len(dataset): break
                    new_tokens   = output_ids[i][prompt_len:]
                    decoded      = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    prediction   = extract_answer(decoded)
                    question     = dataset[sample_idx]['question']
                    raw_answer   = dataset[sample_idx]['answer']
                    ground_truth = extract_answer(raw_answer)

                    is_correct = _compare_answers(prediction, ground_truth, output_format, config=config)
                    if is_correct:
                        correct += 1
                    total += 1

                    if out_f is not None:
                        record = {"idx": sample_idx, "question": question, "prediction": prediction, "ground_truth": ground_truth, "correct": is_correct, "decoded": decoded}
                        out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
                        out_f.flush()

                    if sample_idx < 3:
                        logger.info(f"[Sample {sample_idx+1}] GT: {ground_truth} | Pred: {prediction}")
                        logger.info(f"Tail: ...{decoded[-100:].replace(chr(10), ' ')}")
                    sample_idx += 1
    finally:
        if out_f is not None: out_f.close()

    accuracy = correct / total if total > 0 else 0
    logger.info(f"{split_name.capitalize()} Accuracy: {accuracy:.2%}")

    if wandb.run is not None and split_name == "test":
        wandb.log({"test/accuracy": accuracy, "test/num_correct": correct, "test/num_total": total})

    return accuracy