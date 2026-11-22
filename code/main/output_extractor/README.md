# output_extractor

Scripts that read the per-(model, dataset) JSONLs under
`outputs/.checkpoints/EGLR/main_eglr/eglr_outputs/<model>/<dataset>/` and emit
LaTeX-ready rows for the paper's main results tables.

Run from `code/` so the relative path to `outputs/` resolves correctly.

## What's here

| Script | Produces | Paper table |
|--------|----------|-------------|
| `compute_table1.py` | Greedy / EGLR (best-L) / EGLR-Oracle / EGLR-SC per (model × dataset) | Table 1 (`tab:main_singlecompletion`) |
| `compute_table2.py` | Per-L greedy accuracy (L=0..10) + cross-L oracle, sectioned per dataset | Table 2 (`tab:L_sweep`) |

Both scripts read **greedy** (T=0.0) JSONLs only — both `topk0` and `topk1`
count as greedy (file naming differs between the older L×T grid run and the
newer per-model sweeps).

## Usage

```bash
# from main/
python main/output_extractor/compute_table1.py                  # all 8 models, 6 datasets
python main/output_extractor/compute_table1.py --no-eglrsc      # skip EGLR-SC row
python main/output_extractor/compute_table1.py --models Qwen2.5-3B-Instruct,Llama-3.1-8B-Instruct

python main/output_extractor/compute_table2.py                  # all 8 models × 6 datasets
python main/output_extractor/compute_table2.py --datasets math_500,gsm8k
```

Both scripts print to stdout. Pipe into a file or paste into the .tex.

## Definitions

- **Greedy** = accuracy at L=0
- **EGLR** = `max` accuracy over L ∈ {1,…,10}
- **EGLR-Oracle** = fraction of problems solved by *any* L ∈ {1,…,10}
- **EGLR-SC** = majority vote across L ∈ {0,…,10} per problem
- **Oracle (Table 2)** = same as EGLR-Oracle — union of correct sets over L=1..10

## When to re-run

Whenever new JSONL files land for any (model, dataset, L=0..10) cell.
The scripts pick the **most recent** file per (model, dataset, L) automatically.
