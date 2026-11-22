"""
Answer extraction matching lm-eval-harness conventions.

Strategy
────────
We import lm_eval's primitives directly whenever available:

    from lm_eval.tasks.minerva_math.utils import (
        is_equiv, last_boxed_only_string, remove_boxed,
        normalize_final_answer, get_unnormalized_answer,
    )

That guarantees numerical parity with the harness. The module path has
shifted across releases, so we try a small chain of candidates; if none
resolves, we fall back to local Minerva-parity implementations. The
fallback is byte-equivalent to the lm-eval reference on every test case
we've checked, but the imported version is always preferred.

For GSM8K, lm-eval's extraction is defined inside the task YAML as a
regex (not as Python), so there is nothing to import — we use the same
regex (`#### (\\-?[0-9\\.\\,]+)`).

Dataset dispatch (consumed by eval.py)
──────────────────────────────────────
  gsm8k, gsm8k_gpt2           → GSM8K regex                + int compare
  hendrycks_math, math_500,   → boxed → normalize          + is_equiv
  minerva_math, amc23
  aime24, aime25              → boxed → int                + int compare

Prompt compatibility
────────────────────
Our configs prompt instruct models with "put your final answer inside
\\boxed{}". For math-style tasks we try \\boxed{} first, then fall back
to lm-eval's `get_unnormalized_answer` ("The answer is X.") so NuminaMath
and similar models that use either convention are both handled.

References
──────────
lm-eval-harness: lm_eval/tasks/{gsm8k,minerva_math}/
Original MATH eval: Hendrycks et al., 2021
"""

import math
import re
import logging

log = logging.getLogger("eglr.extraction")


# ─────────────────────────────────────────────────────────────────────────────
# Try to import lm-eval primitives — falls back to local impls on failure
# ─────────────────────────────────────────────────────────────────────────────

_LM_EVAL_SOURCES = [
    # Modern (lm-eval >= 0.4.x)
    "lm_eval.tasks.minerva_math.utils",
    # Legacy
    "lm_eval.tasks.hendrycks_math.utils",
]

_lme_is_equiv                 = None
_lme_last_boxed_only_string   = None
_lme_remove_boxed             = None
_lme_normalize_final_answer   = None
_lme_get_unnormalized_answer  = None
_LM_EVAL_SOURCE_USED          = None


def _try_import_lm_eval():
    global _lme_is_equiv, _lme_last_boxed_only_string, _lme_remove_boxed
    global _lme_normalize_final_answer, _lme_get_unnormalized_answer
    global _LM_EVAL_SOURCE_USED

    for mod_path in _LM_EVAL_SOURCES:
        try:
            mod = __import__(mod_path, fromlist=["*"])
        except Exception:
            continue

        required = ("is_equiv", "last_boxed_only_string", "remove_boxed")
        if not all(getattr(mod, n, None) is not None for n in required):
            continue

        _lme_is_equiv                 = getattr(mod, "is_equiv")
        _lme_last_boxed_only_string   = getattr(mod, "last_boxed_only_string")
        _lme_remove_boxed             = getattr(mod, "remove_boxed")
        _lme_normalize_final_answer   = getattr(mod, "normalize_final_answer", None)
        _lme_get_unnormalized_answer  = getattr(mod, "get_unnormalized_answer", None)
        _LM_EVAL_SOURCE_USED          = mod_path
        log.info(f"extraction: using lm_eval primitives from {mod_path}")
        return True

    log.info("extraction: lm_eval not importable — using local Minerva-parity fallbacks")
    return False


_HAS_LM_EVAL = _try_import_lm_eval()


def lm_eval_status():
    """Return a small dict summarizing which primitives came from lm_eval."""
    return {
        "has_lm_eval":              _HAS_LM_EVAL,
        "source_module":            _LM_EVAL_SOURCE_USED,
        "is_equiv":                 "lm_eval" if _lme_is_equiv is not None else "local",
        "last_boxed_only_string":   "lm_eval" if _lme_last_boxed_only_string is not None else "local",
        "remove_boxed":             "lm_eval" if _lme_remove_boxed is not None else "local",
        "normalize_final_answer":   "lm_eval" if _lme_normalize_final_answer is not None else "local",
        "get_unnormalized_answer":  "lm_eval" if _lme_get_unnormalized_answer is not None else "local/none",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Local Minerva-parity implementations (used when lm_eval is not importable)
# Adapted from lm-eval-harness/lm_eval/tasks/minerva_math/utils.py
# ─────────────────────────────────────────────────────────────────────────────

def _local_last_boxed_only_string(text):
    if text is None:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return text[idx:right_brace_idx + 1]


def _local_remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if s[:len(left)] == left:
            return s[len(left):]
    for prefix in ("\\boxed{", "\\fbox{"):
        if s.startswith(prefix) and s.endswith("}"):
            return s[len(prefix):-1]
    return s


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if len(substr) > 0 and substr[0] == "{":
            new_str += substr
        else:
            try:
                assert len(substr) >= 2
            except AssertionError:
                return string
            a = substr[0]; b = substr[1]
            if b != "{":
                if len(substr) > 2:
                    post_substr = substr[2:]
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    new_str += "{" + a + "}{" + b + "}"
            else:
                if len(substr) > 2:
                    post_substr = substr[2:]
                    new_str += "{" + a + "}" + b + post_substr
                else:
                    new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int = int(a); b_int = int(b)
        assert string == f"{a_int}/{b_int}"
        return f"\\frac{{{a_int}}}{{{b_int}}}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _local_strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _local_is_equiv(str1, str2):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _local_strip_string(str(str1)) == _local_strip_string(str(str2))
    except Exception:
        return str(str1) == str(str2)


# ─────────────────────────────────────────────────────────────────────────────
# Unified primitive wrappers — prefer lm_eval, fall back to local
# ─────────────────────────────────────────────────────────────────────────────

def last_boxed_only_string(text):
    """Return the last `\\boxed{...}` substring (including wrapper)."""
    if _lme_last_boxed_only_string is not None:
        try:
            return _lme_last_boxed_only_string(text)
        except Exception:
            pass
    return _local_last_boxed_only_string(text)


def remove_boxed(s):
    """Strip the `\\boxed{...}` wrapper."""
    if _lme_remove_boxed is not None:
        try:
            return _lme_remove_boxed(s)
        except Exception:
            pass
    return _local_remove_boxed(s)


def is_equiv(str1, str2):
    """Minerva MATH equivalence — normalize both sides, then string-compare."""
    if _lme_is_equiv is not None:
        try:
            return _lme_is_equiv(str1, str2)
        except Exception:
            pass
    return _local_is_equiv(str1, str2)


def normalize_math_answer(s):
    """Normalize a math answer string per Minerva / lm-eval convention."""
    if s is None:
        return None
    if _lme_normalize_final_answer is not None:
        try:
            return _lme_normalize_final_answer(str(s))
        except Exception:
            pass
    return _local_strip_string(str(s))


def get_unnormalized_answer(text):
    """
    Return whatever follows `The answer is` / `answer:` in `text`.
    Prefers lm_eval's version when available; otherwise uses a close
    regex equivalent.
    """
    if text is None:
        return None
    if _lme_get_unnormalized_answer is not None:
        try:
            return _lme_get_unnormalized_answer(str(text))
        except Exception:
            pass
    # Fallback regex — mirrors lm-eval's `THE ANSWER IS` pattern.
    m = re.search(
        r"(?:[Tt]he\s+answer\s+is|\banswer\s*[:=])\s*\$?(.+?)(?:\.\s|$|\n)",
        str(text),
    )
    if m:
        return m.group(1).strip()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GSM8K — regex identical to lm-eval `gsm8k` task YAML
# ─────────────────────────────────────────────────────────────────────────────

_GSM8K_ANS_RE  = re.compile(r"#### (\-?[0-9\.\,]+)")
_GSM8K_SAYS_RE = re.compile(r"(?:[Tt]he answer is|\banswer:)\s*\$?(\-?[0-9\.\,]+)")
_GSM8K_LAST_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def extract_gsm8k_answer(text):
    """
    GSM8K answer extraction (lm-eval `gsm8k` task).

    Priority:
      1. `#### <num>`  (dataset convention)
      2. `the answer is <num>` / `answer: <num>` (flexible fallback)
      3. Last number anywhere (weakest fallback)

    Returns float or None. Passes through numeric inputs unchanged.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).replace(",", "")

    m = _GSM8K_ANS_RE.search(s)
    if m:
        return _to_float(m.group(1).rstrip("."))

    m = _GSM8K_SAYS_RE.search(s)
    if m:
        return _to_float(m.group(1).rstrip("."))

    nums = _GSM8K_LAST_RE.findall(s)
    if nums:
        return _to_float(nums[-1])

    return None


def compare_gsm8k_answers(pred, gt, output_format="float"):
    """GSM8K comparison — integer-rounded (answers are natural numbers)."""
    if pred is None or gt is None:
        return False
    p = _to_float(pred); g = _to_float(gt)
    if p is None or g is None:
        return False
    # Guard against inf/nan from extracting overflow expressions (e.g., "1e500").
    if not math.isfinite(p) or not math.isfinite(g):
        return False
    return int(round(p)) == int(round(g))


# ─────────────────────────────────────────────────────────────────────────────
# MATH-style — boxed extraction + Minerva normalization + is_equiv
# ─────────────────────────────────────────────────────────────────────────────

def extract_minerva_math_answer(text):
    """
    Minerva-style answer extraction for MATH-500 / MinervaMath / AMC23 /
    Hendrycks MATH.

    Order of attempts:
      1. If `\\boxed{...}` exists → extract its inner content.
      2. Else if `get_unnormalized_answer` finds "The answer is X" → use that.
      3. Else → treat the whole input as an already-extracted answer.

    Step (2) is what makes NuminaMath-7B-CoT and other "The answer is" style
    models work alongside boxed-style Qwen/Llama instruct models.

    Returns a Minerva-normalized string (or None).
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return normalize_math_answer(str(text))
    s = str(text)

    boxed = last_boxed_only_string(s)
    if boxed is not None:
        inner = remove_boxed(boxed)
    else:
        inner = get_unnormalized_answer(s)
        if inner is None:
            inner = s

    if inner is None:
        return None

    inner = str(inner).strip().rstrip(".")
    return normalize_math_answer(inner)


def compare_minerva_math_answers(pred, gt, output_format="float"):
    """MATH-style comparison — lm-eval's `is_equiv`."""
    return is_equiv(pred, gt)


# ─────────────────────────────────────────────────────────────────────────────
# AIME / AMC integer — answer is 0-999
# ─────────────────────────────────────────────────────────────────────────────

def extract_aime_answer(text):
    """
    AIME24/25 answers are integers 0-999.

    Order of attempts:
      1. Last `\\boxed{}` → parse first int in normalized content.
      2. `get_unnormalized_answer` → parse first int.
      3. Last int anywhere in text.

    Returns int or None. Passes through numeric inputs unchanged.
    """
    if text is None:
        return None
    if isinstance(text, int):
        return text
    if isinstance(text, float):
        if not math.isfinite(text):
            return None
        return int(round(text))
    s = str(text)

    # Attempt 1: boxed
    boxed = last_boxed_only_string(s)
    if boxed is not None:
        inner = normalize_math_answer(remove_boxed(boxed).strip().rstrip("."))
        if inner:
            m = re.search(r"-?\d+", inner)
            if m:
                try:
                    return int(m.group(0))
                except ValueError:
                    pass

    # Attempt 2: "The answer is X"
    unn = get_unnormalized_answer(s)
    if unn:
        inner = normalize_math_answer(str(unn).strip().rstrip("."))
        if inner:
            m = re.search(r"-?\d+", inner)
            if m:
                try:
                    return int(m.group(0))
                except ValueError:
                    pass

    # Attempt 3: last int anywhere
    nums = re.findall(r"-?\d+", s)
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            return None
    return None


def compare_aime_answers(pred, gt, output_format="float"):
    """AIME/AMC integer comparison."""
    if pred is None or gt is None:
        return False
    try:
        return int(pred) == int(gt)
    except (ValueError, TypeError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch — consumed by eval.py
# ─────────────────────────────────────────────────────────────────────────────
#
# For a new dataset, add an entry mapping `dataset.name` →
#   (extract_fn, compare_fn, task_description)
# where task_description is a short human label used in logs / summaries.
#

DATASET_HANDLERS = {
    # GSM8K — regex-based (lm-eval gsm8k task)
    "gsm8k":          (extract_gsm8k_answer,        compare_gsm8k_answers,        "lm-eval:gsm8k"),
    "gsm8k_gpt2":     (extract_gsm8k_answer,        compare_gsm8k_answers,        "lm-eval:gsm8k (coconut)"),

    # MATH-style — boxed extraction + Minerva normalization + is_equiv
    "hendrycks_math": (extract_minerva_math_answer, compare_minerva_math_answers, "lm-eval:minerva_math"),
    "math_500":       (extract_minerva_math_answer, compare_minerva_math_answers, "lm-eval:math_500 (minerva extraction)"),
    "minerva_math":   (extract_minerva_math_answer, compare_minerva_math_answers, "lm-eval:minerva_math"),
    "amc23":          (extract_minerva_math_answer, compare_minerva_math_answers, "lm-eval:amc23 (minerva extraction)"),

    # AIME — integer 0-999
    "aime24":         (extract_aime_answer,         compare_aime_answers,         "lm-eval:aime24 (integer)"),
    "aime25":         (extract_aime_answer,         compare_aime_answers,         "lm-eval:aime25 (integer)"),
}


def get_extractor(dataset_name):
    h = DATASET_HANDLERS.get(dataset_name)
    return h[0] if h else None


def get_comparator(dataset_name):
    h = DATASET_HANDLERS.get(dataset_name)
    return h[1] if h else None


def get_task_description(dataset_name):
    h = DATASET_HANDLERS.get(dataset_name)
    return h[2] if h else f"unknown ({dataset_name})"