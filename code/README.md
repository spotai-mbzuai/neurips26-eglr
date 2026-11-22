# EGLR — Entropy-Gated Latent Recursion

Training-free test-time refinement for instruct LLMs. No fine-tuning, no new parameters, no auxiliary modules. Drop-in for any decoder-only causal LM with a `model.model.layers` stack (Llama / Qwen / Gemma).

---

## The idea in one picture

```
           ┌──────────────────────────────────────────────────────────────┐
           │                 Standard autoregressive decoding              │
           │                                                              │
           │   [N layers] ─▶ h_final ─▶ lm_head ─▶ next-token probs        │
           └──────────────────────────────────────────────────────────────┘

                       ──────── EGLR intervenes here ────────
                                      ▼
           ┌──────────────────────────────────────────────────────────────┐
           │   entropy(probs) > τ_entropy ?                                │
           │          │                                                    │
           │          ├── no  ─▶ sample next token, move on (depth=0)      │
           │          │                                                    │
           │          └── yes ─▶ refine  (loop up to K_max times)           │
           │                                                                │
           │                    ┌────────────────────────────────────────┐ │
           │                    │  h_input = (1-α)·h_anchor + α·h_k      │ │
           │                    │  h_k     = top-L layers (h_input)      │ │
           │                    │  probs_k = lm_head(norm(h_k))           │ │
           │                    │                                         │ │
           │                    │  KL(probs_k ‖ probs_{k-1}) < τ_kl ?     │ │
           │                    │   │                                     │ │
           │                    │   ├── yes ─▶ commit, exit               │ │
           │                    │   └── no  ─▶ next iteration              │ │
           │                    └────────────────────────────────────────┘ │
           └──────────────────────────────────────────────────────────────┘
```

**`h_anchor`** = hidden state input to layer `(N-L)` on the current token (unchanged across iterations).
**`h_k`** = hidden state produced by re-running the top-L layers on the fused input.

---

## Why it works

1. **Detect uncertainty** via the next-token entropy (`H(p) > τ_entropy`).
2. **Re-run only the top-L layers** — the ones that shape abstract decisions — against a fused input that anchors on the original hidden state.
3. **Stop when the distribution stabilises** (KL between consecutive iterations < `τ_kl`), or after `K_max` attempts.

Only the **top-L** layers rerun, so refinement cost is `L × K_depth / N` extra FLOPs per refined token — typically 1–5% of generated tokens get refined.

---

## Hyperparameters

| Param         | Default | Meaning |
|---------------|---------|---------|
| `L`           | 2       | Number of top decoder layers to re-invoke. Higher = deeper refinement, more drift risk. |
| `alpha`       | 0.2     | Fusion weight `h = (1-α)·h_anchor + α·h_k`. Small values stay close to base. |
| `K_max`       | 3       | Max recursion iterations per token. |
| `tau_entropy` | 1.0     | Entropy threshold (nats). Refinement fires only above this. |
| `tau_kl`      | 1e-3    | KL early-exit threshold. Stop once the distribution stops moving. |
| `norm_match`  | True    | Rescale `h_k` to match `h_anchor`'s L2 norm before fusion. |

Calibrate `tau_entropy` by running once with `L=0`, then setting `tau_entropy` to the 92.5th percentile of per-token entropies (≈ 7.5% refinement rate). `eval_eglr.py` emits `entropy_stats.quantiles` in the summary JSON for this.

---

## Install

Python 3.11 and a CUDA GPU. From `code/`:

```bash
conda create -n eglr python=3.11 -y
conda activate eglr
pip install -r requirements.txt
```

`requirements.txt` pins the package versions used for the paper's experiments.

---

## Run conventions

**Always run from `code/`** inside the `neurips26-eglr` meta-repo. Every entry-point script enforces this and exits with a friendly error otherwise.

```bash
cd neurips26-eglr/code
python main/main_eglr.py --config main/configs/math_qwen_3b_eglr.yaml
```

Why: yaml paths use `outputs/...` and `../../.env`, both relative to `code/`.

### Smoke test (50 problems)

```bash
python main/main_eglr.py --config main/configs/math_qwen_3b_eglr.yaml --samples 50
```

### Override hyperparameters from CLI

```bash
python main/main_eglr.py --config main/configs/math_qwen_3b_eglr.yaml \
    --L 2 --alpha 0.15 --K_max 3 --tau_entropy 1.0 --norm_match true
```

### Disable refinement (baseline)

Set `L=0` in the config. EGLR becomes a no-op and the run measures base-model behaviour while still logging per-token entropies — exactly what you need for calibration.

---

## Phase 3 sweeps (batched benchmark)

`main/run_sweep.py` loads the model **once** and cycles through every `(dataset, L)` combination in memory.

```
main/run_sweep.py  ─▶  for each dataset in phase3_datasets.yaml:
                                ─▶ L=0 run — record per-token entropies
                                ─▶ set τ_entropy = 92.5th percentile of those entropies
                                ─▶ for L in 1..10:
                                       run eval; if refinement_rate ∉ [5%, 10%],
                                       recompute τ and retry (up to max-attempts)
                                ─▶ seed next dataset's starting τ from this one
```

**Defaults**: datasets `aime24 → aime25 → amc23 → minerva_math → math_500 → gsm8k`; L `0..10`; target refinement rate **[5%, 10%]**.

### Launching a sweep

```bash
cd neurips26-eglr/code
bash main/scripts/phase3_qwen2_5_3b.sh       # Qwen2.5-3B-Instruct
bash main/scripts/phase3_qwen2_5_7b.sh       # Qwen2.5-7B-Instruct
bash main/scripts/phase3_llama_3_1_8b.sh     # Llama-3.1-8B-Instruct
bash main/scripts/phase3_gemma_3_12b.sh      # Gemma-3-12B
# (and others under main/scripts/)
```

Flags after the script name are forwarded to `run_sweep.py`:

```bash
bash main/scripts/phase3_qwen2_5_3b.sh --max-attempts 5
bash main/scripts/phase3_qwen2_5_3b.sh --no-resume
bash main/scripts/phase3_qwen2_5_3b.sh --target-rate 0.08
```

`run_sweep.py` skips any `(dataset, L)` combo that already has an accepted summary (refinement rate in-band). Use `--no-resume` for a clean run.

---

## File layout

```
code/
├── main/                  # main codebase (layer-wise EGLR)
│   ├── main_eglr.py           # entry point
│   ├── model_eglr.py          # EGLRModel + generate_eglr
│   ├── eval_eglr.py           # eval driver (per-sample JSONL, summary JSON)
│   ├── data.py / config.py / utils.py / extraction.py / calibrate.py
│   ├── run_sweep.py           # hands-free Phase 3 sweep
│   ├── configs/               # per (dataset, model, hparam) tuple
│   │   └── extra/             # legacy / unused configs
│   ├── scripts/               # bash launchers
│   ├── output_extractor/      # paper table scripts
│   └── SOTA/                  # beam-search baseline comparison
│       ├── beam/              # SOTA entry points + utils
│       └── configs/
├── outputs/                   # gitignored: runs, logs, checkpoints, calibration
├── LICENSE
└── README.md
```

### Key code paths

| File | What it does |
|---|---|
| `main/model_eglr.py:EGLRModel` | Frozen wrapper exposing `generate_eglr`. |
| `main/model_eglr.py:generate_eglr` | Decoding loop: full forward → entropy gate → top-L refinement → sample. |
| `main/model_eglr.py:_apply_top_L` | Runs layers `[N-L : N]` + final norm on a single hidden state. |
| `main/eval_eglr.py:evaluate_eglr` | Batched eval; writes JSONL + summary JSON. |
| `main/run_sweep.py` | Sweep orchestrator with auto-τ calibration. |

---

## Artifacts

All run outputs land in `code/outputs/`, which is gitignored (never committed):

```
code/outputs/
├── .checkpoints/EGLR/         ← outputs, summaries, resolved configs (main)
│   └── SOTA/                  ← same, but SOTA runs (baselines)
├── .logs/EGLR/                ← per-run .log files
│   └── SOTA/
├── .wandb/EGLR/               ← wandb run data
│   └── SOTA/
└── .calibration/EGLR/         ← per-(model, dataset) calibration journals
```

Yaml fields that point here:

```yaml
parameters:
  env_path:        ../../.env                            # file with WANDB_API_KEY, HF_TOKEN
  checkpoint_path: outputs/.checkpoints/EGLR        # main runs
  wandb_path:      outputs/.wandb/EGLR
  logs_path:       outputs/.logs/EGLR
  # SOTA configs use .../EGLR/SOTA to namespace away from main
```

### Per-run artifact paths

Run tag (same for jsonl + summary + wandb name):
```
{ts}_{model_short}_L{L}_a{α}_K{K_max}_tau{τe}_nm{0|1}_T{T}_topk{topk}_s{samples}_mnt{max_new_tokens}_{dataset}
```

| Where | Contents |
|---|---|
| `outputs/.logs/EGLR/main_eglr_<ts>.log` | full run log (mirrored from stdout) |
| `outputs/.checkpoints/EGLR/main_eglr/resolved_configs/<run_tag>.yaml` | post-override config (reproducible snapshot) |
| `outputs/.checkpoints/EGLR/main_eglr/eglr_outputs/<run_tag>.jsonl` | per-sample generations, per-token depths, per-token entropies |
| `outputs/.checkpoints/EGLR/main_eglr/eglr_summaries/<run_tag>_summary.json` | accuracy, depth histogram, entropy quantiles, hparams |
| `outputs/.calibration/EGLR/<model_short>__<dataset>.jsonl` | sweep calibration history |

SOTA-baseline runs (`main/SOTA/beam/main_eglr.py`) get a `SOTA_` prefix on the run tag and land under `.../EGLR/SOTA/` — easy filtering on wandb.

### Wandb

Put your wandb username in your `.env` file as `WANDB_ENTITY=<username>` (never in a config); override with `--wandb-entity`. Project via `--wandb-project`. Run name = run tag. Disable with `--no-wandb`.

---

## Supported architectures

Any HF causal LM with `model.model.layers` and `model.model.norm`. Tested on:

- Qwen 2.5 (0.5B / 3B / 7B / Math-7B)
- Llama 3.1 (8B)
- Gemma 3 (12B)
- NuminaMath-7B-CoT

Flash Attention 2 used automatically when available.

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{bhattacharjee2026entropy,
  title={Entropy-Gated Latent Recursion},
  author={Bhattacharjee, Soham and Chauhan, Dushyant Singh and Lahlou, Salem and Takac, Martin and Lukas, Nils},
  journal={arXiv preprint arXiv:2606.16620},
  year={2026}
}
```

## License

Free for research and non-commercial use, provided you cite the paper above. Commercial use requires permission from the authors. See [LICENSE](LICENSE).
