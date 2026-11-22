import os
import sys
import re
import random
import logging
import torch
from datetime import datetime, timezone, timedelta

# Setup globals
UAE_TZ = timezone(timedelta(hours=4))
RUN_TIMESTAMP = datetime.now(UAE_TZ).strftime('%Y%m%d_%H%M%S')
SCRIPT_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# Setup logging formatting
class UAEFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, UAE_TZ)
        return dt.strftime(datefmt or '%Y-%m-%d %H:%M:%S')

formatter = UAEFormatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

def _setup_base_logger():
    logger = logging.getLogger("eglr")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    return logger

base_logger = _setup_base_logger()

def get_logger(name):
    """Retrieve an appropriately namespaced logger."""
    return logging.getLogger("eglr." + name)


def attach_file_handler(log_path):
    """Attach a file handler to the root 'eglr' logger. Idempotent."""
    logger = logging.getLogger("eglr")
    abs_log_path = os.path.abspath(log_path)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and os.path.abspath(h.baseFilename) == abs_log_path:
            return abs_log_path  # already attached
    os.makedirs(os.path.dirname(abs_log_path), exist_ok=True)
    fh = logging.FileHandler(abs_log_path)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return abs_log_path


def build_run_tag(config, model, ts: str = None) -> str:
    """Canonical experiment ID — used for JSONL, summary JSON, wandb run names.

    Same tag across all three ensures easy cross-referencing of a single run.
    Falls back to RUN_TIMESTAMP when `ts` is not provided.
    """
    ts = ts or RUN_TIMESTAMP
    model_short = config.model.name.split("/")[-1]
    cfg_tag = (
        f"L{model.L}_a{model.alpha}_K{model.K_max}"
        f"_tau{model.tau_entropy}_nm{int(model.norm_match)}"
    )
    gen = config.generation
    sampling_tag = (
        f"_T{getattr(gen, 'temperature', 0.0)}"
        f"_topk{getattr(gen, 'top_k', 0)}"
    )
    return (
        f"{ts}_{model_short}_{cfg_tag}{sampling_tag}"
        f"_s{config.dataset.samples}_mnt{config.generation.max_new_tokens}"
        f"_{config.dataset.name}"
    )


def write_resolved_config(cfg_dict: dict, out_path: str):
    """Write the final post-override config as YAML next to the run artifacts.

    Snapshot of the exact configuration used for this run — stable enough to
    reproduce the run from just the artifact directory.
    """
    import yaml
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=False)


def set_seed(seed=1333):
    """Set global seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reserve_gpu_memory():
    """Reserve 80% of available memory on all GPUs."""
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.empty_cache()
        free_memory = torch.cuda.mem_get_info(device)[0]
        reserve_bytes = int(free_memory * 0.80)
        dummy = torch.empty(reserve_bytes, dtype=torch.uint8, device=device)
        del dummy
        print(f"Reserved {reserve_bytes / 1e9:.2f} GB on GPU {gpu_id}")


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def early_stop(self, value):
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ──────────────────────────────────────────────
# Answer extraction
# ──────────────────────────────────────────────

def extract_answer_number(text):
    if text is None: return None
    text = text.replace(',', '')
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if match: return float(match.group(1))
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        try: return float(matches[-1])
        except ValueError: return None
    return None


def extract_boxed_answer(text):
    """
    Extract the answer from a LaTeX \\boxed{...} expression.

    Strategy (in order):
      1. Find the last \\boxed{...} in the text (handles nested braces correctly).
      2. Try to parse the boxed content as a plain number and return float.
      3. If the content is a fraction (\\frac{a}{b}), evaluate and return float.
      4. Strip common LaTeX decoration (dollar signs, spaces, commas, percent)
         and try again as a number.
      5. Return the raw stripped string so callers can do string comparison if
         numeric comparison fails.
      6. If no \\boxed is found, fall back to extract_answer_number().
    """
    if text is None:
        return None

    # ── 1. Find all \boxed{ occurrences and pick the last one ──
    boxed_contents = []
    search_start = 0
    while True:
        idx = text.find(r'\boxed{', search_start)
        if idx == -1:
            break
        depth = 0
        content_start = idx + len(r'\boxed{')
        i = content_start
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    boxed_contents.append(text[content_start:i])
                    break
                depth -= 1
            i += 1
        search_start = idx + 1

    if not boxed_contents:
        return extract_answer_number(text)

    raw = boxed_contents[-1].strip()

    # ── 2. Plain numeric ──
    cleaned = raw.replace(',', '').replace('$', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        pass

    # ── 3. Simple LaTeX fraction: \frac{a}{b} or -\frac{a}{b} ──
    frac_match = re.fullmatch(
        r'(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}',
        cleaned
    )
    if frac_match:
        sign = -1.0 if frac_match.group(1) == '-' else 1.0
        num, den = float(frac_match.group(2)), float(frac_match.group(3))
        if den != 0:
            return sign * num / den

    # ── 4. Strip LaTeX decorations and retry ──
    stripped = re.sub(r'[\\$%,\s]', '', cleaned)
    stripped = stripped.replace('{', '').replace('}', '')
    try:
        return float(stripped)
    except ValueError:
        pass

    # ── 5. Try to find any trailing number (only when content looks numeric) ──
    _SYMBOLIC_RE = re.compile(r'[a-zA-Z\\^]')
    if not _SYMBOLIC_RE.search(raw):
        raw_no_comma = raw.replace(',', '')
        nums = re.findall(r'-?\d+(?:\.\d+)?', raw_no_comma)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                pass

    # ── 6. Return raw string for string-based comparison upstream ──
    return raw