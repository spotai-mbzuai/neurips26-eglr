import os
import urllib.request
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BatchEncoding
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from config import CACHE_DIR
from utils import get_logger

logger = get_logger(__name__)

# Datasets whose answers are contained in \boxed{} (math-style, Minerva extraction).
_MATH_STYLE_DATASETS = {
    "hendrycks_math", "math_500", "minerva_math", "amc23", "aime24", "aime25",
}

# ── Few-shot examples for base model evaluation (Minerva-style) ──────────────
MATH_FEWSHOT_EXAMPLES = [
    {
        "problem": "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$."
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$"
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}"
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$"
    },
]

GSM8K_FEWSHOT_EXAMPLES = [
    {
        "problem": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "solution": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6"
    },
    {
        "problem": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "solution": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5"
    },
    {
        "problem": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "solution": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"
    },
    {
        "problem": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "solution": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n#### 8"
    },
]


def initialize_tokenizer(config, for_generation=False):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        add_bos_token=False,
        use_fast=True,
        cache_dir=config.parameters.cache_dir
    )

    pad_is_eos = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_is_eos = True

    # Check for chat template and add a default fallback for models like GPT-2
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )

    tokenizer.use_default_system_prompt = False

    if for_generation:
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
    else:
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'

    return tokenizer, pad_is_eos


# ─────────────────────────────────────────────────────────────────────────────
# GSM8K
# ─────────────────────────────────────────────────────────────────────────────

def get_gsm8k_dataset(split="train", config=None):
    logger.info(f"Loading GSM8K dataset (split: {split})...")
    cache_dir = config.parameters.cache_dir if config else CACHE_DIR

    if split == "test":
        dataset = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
        if config and config.dataset.samples > 0:
            dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), config.dataset.samples)))
        return dataset

    full_train = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)
    if config and config.dataset.samples > 0:
        full_train = full_train.shuffle(seed=42).select(range(min(len(full_train), config.dataset.samples)))

    val_split_size = config.dataset.val_split_size if config else 0.05
    splits = full_train.train_test_split(test_size=val_split_size, seed=42)
    if split == "train":
        return splits["train"]
    elif split == "valid":
        return splits["test"]
    else:
        raise ValueError(f"Unknown split: {split}")


def get_gsm8k_gpt2_dataset(split="train", config=None):
    """Loads and preprocesses the extended Coconut GSM8K dataset."""
    logger.info(f"Loading GSM8K GPT-2 (Coconut) dataset (split: {split})...")
    cache_dir = config.parameters.cache_dir if config else CACHE_DIR
    data_dir = os.path.join(cache_dir, "coconut_gsm8k")
    os.makedirs(data_dir, exist_ok=True)

    urls = {
        "train": "https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/train.txt",
        "valid": "https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/valid.txt",
        "test": "https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/test.txt"
    }

    if split not in urls:
        raise ValueError(f"Unknown split: {split}")

    file_path = os.path.join(data_dir, f"gsm_{split}.txt")

    if not os.path.exists(file_path):
        logger.info(f"Downloading {split} set to {file_path}...")
        urllib.request.urlretrieve(urls[split], file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = f.readlines()

    processed_data = {"question": [], "answer": []}

    for line in raw_data:
        line = line.strip()
        if not line:
            continue
        parts = line.split("||")
        if len(parts) != 2:
            continue
        question = parts[0].strip()

        right_parts = parts[1].split("##")
        steps_list = right_parts[0].strip().split(" ")
        steps_formatted = "\n".join(steps_list)
        final_ans = right_parts[-1].strip()
        reconstructed_answer = f"{steps_formatted}\n#### {final_ans}"

        processed_data["question"].append(question)
        processed_data["answer"].append(reconstructed_answer)

    hf_dataset = HFDataset.from_dict(processed_data)

    if config and config.dataset.samples > 0:
        hf_dataset = hf_dataset.shuffle(seed=42).select(range(min(len(hf_dataset), config.dataset.samples)))

    return hf_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Hendrycks MATH (original 7,500 / 5,000)
# ─────────────────────────────────────────────────────────────────────────────

def _stratified_sample_math(dataset, n_samples, seed=42, level_field="level"):
    """
    Sample exactly n_samples from a math dataset such that every difficulty
    level is represented as evenly as possible. Deterministic via seeded RNG.
    Works on any HuggingFace Dataset with a `level` field (int or string).
    """
    import random as _rng_local
    from collections import defaultdict

    if n_samples <= 0 or n_samples >= len(dataset):
        return dataset

    groups: dict = defaultdict(list)
    for idx in range(len(dataset)):
        level = dataset[idx].get(level_field, "Unknown")
        groups[level].append(idx)

    # If there's only one effective level, fall back to uniform shuffle.
    levels = sorted(groups.keys(), key=lambda x: str(x))
    if len(levels) <= 1:
        rng = _rng_local.Random(seed)
        pool = list(range(len(dataset)))
        rng.shuffle(pool)
        return dataset.select(sorted(pool[:n_samples]))

    n_levels = len(levels)
    base_quota = n_samples // n_levels
    remainder  = n_samples  % n_levels

    levels_by_size = sorted(levels, key=lambda l: len(groups[l]), reverse=True)
    quotas = {lvl: base_quota + (1 if i < remainder else 0)
              for i, lvl in enumerate(levels_by_size)}

    rng = _rng_local.Random(seed)
    selected, shortfall = [], 0
    for lvl in levels:
        indices = groups[lvl][:]
        rng.shuffle(indices)
        take = min(quotas[lvl], len(indices))
        selected.extend(indices[:take])
        shortfall += quotas[lvl] - take

    if shortfall > 0:
        selected_set = set(selected)
        pool = [i for i in range(len(dataset)) if i not in selected_set]
        rng.shuffle(pool)
        selected.extend(pool[:shortfall])

    return dataset.select(sorted(selected))


def get_hendrycks_math_dataset(split="train", config=None):
    """
    EleutherAI/hendrycks_math — full 7,500 train / 5,000 test across 7 subjects.
    """
    logger.info(f"Loading EleutherAI/hendrycks_math (split: {split})...")
    cache_dir = config.parameters.cache_dir if config else CACHE_DIR

    SUBJECTS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]

    def _load_and_concat(hf_split: str):
        parts = []
        for subject in SUBJECTS:
            ds = load_dataset(
                "EleutherAI/hendrycks_math",
                name=subject, split=hf_split, cache_dir=cache_dir,
            )
            parts.append(ds)
        return concatenate_datasets(parts)

    def _remap(example):
        return {
            "question": example["problem"],
            "answer":   example["solution"],
            "level":    example.get("level", ""),
            "type":     example.get("type", ""),
        }

    if split in ("train", "valid"):
        raw_train = _load_and_concat("train")
        dataset   = raw_train.map(_remap, remove_columns=raw_train.column_names)
        if config and getattr(config.dataset, "samples", -1) > 0:
            dataset = _stratified_sample_math(dataset, config.dataset.samples, seed=42)
        val_split_size = config.dataset.val_split_size if config else 0.05
        splits = dataset.train_test_split(test_size=val_split_size, seed=42)
        if split == "train":
            logger.info(f"Train set: {len(splits['train']):,} examples")
            return splits["train"]
        else:
            logger.info(f"Val set: {len(splits['test']):,} examples")
            return splits["test"]

    elif split == "test":
        raw_test = _load_and_concat("test")
        dataset  = raw_test.map(_remap, remove_columns=raw_test.column_names)
        if config and getattr(config.dataset, "samples", -1) > 0:
            dataset = _stratified_sample_math(dataset, config.dataset.samples, seed=42)
        logger.info(f"Test set: {len(dataset):,} examples")
        return dataset

    else:
        raise ValueError(f"Unknown split '{split}' for hendrycks_math.")


# ─────────────────────────────────────────────────────────────────────────────
# MATH-500 — HuggingFaceH4/MATH-500 (Phase 1/2 anchor dataset)
# ─────────────────────────────────────────────────────────────────────────────

def get_math_500_dataset(split="test", config=None):
    """
    HuggingFaceH4/MATH-500 — the curated 500-problem subset of MATH test set
    released by HuggingFace for reasoning evals.

    Only 'test' split exists (500 problems). Fields:
      problem, solution, answer, subject, level, unique_id

    We remap solution → answer so the boxed-extraction path works. Stratified
    sampling by `level` is applied when config.dataset.samples > 0.
    """
    if split != "test":
        raise ValueError(f"MATH-500 only has a 'test' split; got split={split}")

    logger.info("Loading HuggingFaceH4/MATH-500 (test)...")
    cache_dir = config.parameters.cache_dir if config else CACHE_DIR

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=cache_dir)

    def _remap(example):
        return {
            "question": example["problem"],
            # Use solution (has \boxed{}) as the extraction source.
            "answer":   example.get("solution") or example.get("answer", ""),
            "level":    example.get("level", ""),
            "type":     example.get("subject", ""),
        }

    dataset = dataset.map(_remap, remove_columns=dataset.column_names)

    if config and getattr(config.dataset, "samples", -1) > 0:
        dataset = _stratified_sample_math(dataset, config.dataset.samples, seed=42)

    logger.info(f"MATH-500 test set: {len(dataset):,} examples")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# MinervaMath — math-ai/minervamath
# ─────────────────────────────────────────────────────────────────────────────

def get_minerva_math_dataset(split="test", config=None):
    """
    math-ai/minervamath — ~272 problems from the Minerva paper's MATH eval subset.
    Schema varies slightly by release — we normalize common field name variants.
    """
    if split != "test":
        raise ValueError(f"MinervaMath only has a 'test' split; got split={split}")

    logger.info("Loading math-ai/minervamath (test)...")
    cache_dir = config.parameters.cache_dir if config else CACHE_DIR

    dataset = load_dataset("math-ai/minervamath", split="test", cache_dir=cache_dir)

    def _remap(example):
        q = example.get("problem") or example.get("question") or ""
        a = example.get("solution") or example.get("answer") or ""
        return {
            "question": q,
            "answer":   a,
            "level":    example.get("level", ""),
            "type":     example.get("subject", example.get("type", "")),
        }

    dataset = dataset.map(_remap, remove_columns=dataset.column_names)

    if config and getattr(config.dataset, "samples", -1) > 0 and config.dataset.samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(config.dataset.samples))

    logger.info(f"MinervaMath test set: {len(dataset):,} examples")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# AMC23 — math-ai/amc23
# ─────────────────────────────────────────────────────────────────────────────

def get_amc23_dataset(split="test", config=None):
    """
    math-ai/amc23 — ~40 problems from the 2023 American Mathematics Competitions.
    Answers are integers. We map to the same (question, answer) schema.
    """
    if split != "test":
        raise ValueError(f"AMC23 only has a 'test' split; got split={split}")

    logger.info("Loading math-ai/amc23 (test)...")
    cache_dir = config.parameters.cache_dir if config else CACHE_DIR

    dataset = load_dataset("math-ai/amc23", split="test", cache_dir=cache_dir)

    def _remap(example):
        q = example.get("problem") or example.get("question") or ""
        a = example.get("answer")
        if a is None:
            a = example.get("solution", "")
        # answer field may already be an integer; keep as string for consistency
        return {
            "question": q,
            "answer":   str(a),
            "level":    example.get("level", ""),
            "type":     example.get("subject", example.get("type", "")),
        }

    dataset = dataset.map(_remap, remove_columns=dataset.column_names)

    if config and getattr(config.dataset, "samples", -1) > 0 and config.dataset.samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(config.dataset.samples))

    logger.info(f"AMC23 test set: {len(dataset):,} examples")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# AIME24 / AIME25 — math-ai/aime24, math-ai/aime25
# ─────────────────────────────────────────────────────────────────────────────

def _get_aime_dataset(year: str, split="test", config=None):
    """Shared loader for AIME24 and AIME25 (both from math-ai/)."""
    hf_path = f"math-ai/aime{year}"

    if split != "test":
        raise ValueError(f"AIME{year} only has a 'test' split; got split={split}")

    logger.info(f"Loading {hf_path} (test)...")
    cache_dir = config.parameters.cache_dir if config else CACHE_DIR

    dataset = load_dataset(hf_path, split="test", cache_dir=cache_dir)

    def _remap(example):
        q = example.get("problem") or example.get("question") or ""
        a = example.get("answer")
        if a is None:
            a = example.get("solution", "")
        return {
            "question": q,
            "answer":   str(a),
            "level":    example.get("level", ""),
            "type":     example.get("subject", example.get("type", "")),
        }

    dataset = dataset.map(_remap, remove_columns=dataset.column_names)

    if config and getattr(config.dataset, "samples", -1) > 0 and config.dataset.samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(config.dataset.samples))

    logger.info(f"AIME{year} test set: {len(dataset):,} examples")
    return dataset


def get_aime24_dataset(split="test", config=None):
    return _get_aime_dataset("24", split=split, config=config)


def get_aime25_dataset(split="test", config=None):
    return _get_aime_dataset("25", split=split, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_LOADERS = {
    "gsm8k":          get_gsm8k_dataset,
    "gsm8k_gpt2":     get_gsm8k_gpt2_dataset,
    "hendrycks_math": get_hendrycks_math_dataset,
    "math_500":       get_math_500_dataset,
    "minerva_math":   get_minerva_math_dataset,
    "amc23":          get_amc23_dataset,
    "aime24":         get_aime24_dataset,
    "aime25":         get_aime25_dataset,
}


def get_dataset(dataset_name, split="train", config=None):
    loader = _LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Known: {list(_LOADERS.keys())}")
    return loader(split=split, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# SampleDataset / DataLoader — unchanged semantics, extended math-style check
# ─────────────────────────────────────────────────────────────────────────────

class SampleDataset(Dataset):
    def __init__(self, config, tokenizer, dataset, mode="train", pad_is_eos=False):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.mode = mode
        self.pad_is_eos = pad_is_eos

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        question = example['question']
        answer = example['answer']

        is_coconut = (self.config.dataset.name == "gsm8k_gpt2")
        is_instruct = getattr(self.config.model, 'is_instruct', True)
        is_math = (self.config.dataset.name in _MATH_STYLE_DATASETS)

        input_prompt = ""
        prompt_only = ""
        prompt = ""

        if self.mode in ("train", "valid"):
            if is_coconut:
                input_prompt = f"{question}\n{answer}"
                prompt_only = f"{question}\n"
            elif not is_instruct:
                if is_math:
                    input_prompt = f"Problem: {question}\nSolution: {answer}"
                    prompt_only = f"Problem: {question}\nSolution:"
                else:
                    input_prompt = f"Question: {question}\nAnswer: {answer}"
                    prompt_only = f"Question: {question}\nAnswer:"
            else:
                messages = [
                    {"role": "system", "content": getattr(self.config.prompts, 'system_train', '')},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                messages = [m for m in messages if m["content"]]
                input_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

                prompt_messages = [m for m in messages if m["role"] != "assistant"]
                prompt_only = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

            tokenized_inputs = self.tokenizer(
                input_prompt, padding="max_length", truncation=True,
                max_length=self.config.dataset.train_max_length, return_tensors="pt", add_special_tokens=False
            )

            prompt_length = len(self.tokenizer(prompt_only, add_special_tokens=False)['input_ids'])

            labels = tokenized_inputs['input_ids'].clone()
            labels[0, :prompt_length] = -100

            if self.pad_is_eos:
                eos_positions = (labels == self.tokenizer.eos_token_id).nonzero(as_tuple=True)
                if len(eos_positions[1]) > 0:
                    first_eos = eos_positions[1][0].item()
                    labels[0, first_eos+1:] = -100
            else:
                labels[labels == self.tokenizer.pad_token_id] = -100

            tokenized_inputs = BatchEncoding({key: value.squeeze(0) if isinstance(value, torch.Tensor) else value for key, value in tokenized_inputs.items() if value is not None})
            tokenized_labels = BatchEncoding({'input_ids': labels.squeeze(0) if isinstance(labels, torch.Tensor) else labels, 'attention_mask': tokenized_inputs['attention_mask']})

            return tokenized_inputs, tokenized_labels

        if self.mode == "eval":
            if is_coconut:
                prompt = f"{question}\n"
            elif not is_instruct:
                fewshot = MATH_FEWSHOT_EXAMPLES if is_math else GSM8K_FEWSHOT_EXAMPLES
                q_label = "Problem" if is_math else "Question"
                a_label = "Solution" if is_math else "Answer"
                prompt = ""
                for ex in fewshot:
                    prompt += f"{q_label}: {ex['problem']}\n{a_label}: {ex['solution']}\n\n"
                prompt += f"{q_label}: {question}\n{a_label}:"
            else:
                messages = [
                    {"role": "system", "content": getattr(self.config.prompts, 'system_eval', '')},
                    {"role": "user", "content": question}
                ]
                messages = [m for m in messages if m["content"]]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            tokenized_inputs = self.tokenizer(
                prompt, padding="max_length", truncation=True,
                max_length=self.config.dataset.eval_max_length, return_tensors="pt", add_special_tokens=False
            )

            tokenized_inputs = BatchEncoding({key: value.squeeze(0) if isinstance(value, torch.Tensor) else value for key, value in tokenized_inputs.items() if value is not None})
            return tokenized_inputs, tokenized_inputs

        raise ValueError(f"Unknown mode: {self.mode}")


def get_dataloader(tokenizer, batch_size=4, split="train", config=None, pad_is_eos=False):
    dataset = get_dataset(config.dataset.name, split=split, config=config)
    if split == "train":
        mode = "train"
    elif split == "valid":
        mode = "valid"
    else:
        mode = "eval"

    sample_dataset = SampleDataset(config, tokenizer, dataset, mode=mode, pad_is_eos=pad_is_eos)
    loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=4)
    return loader