# TinyStories Autoresearch

## Project Purpose
This repository implements an autonomous research workflow for improving a small Transformer language model trained on the TinyStories GPT-4 Clean dataset. The agent explores simple model and training changes by editing only `train.py`, evaluating each experiment with a fixed metric and budget.

## Dataset
The dataset is `karpathy/tinystories-gpt4-clean` from Hugging Face. It contains short, GPT-4-generated TinyStories samples, which are well suited for fast experimentation with compact language models.

## Research Question
Can an autonomous agent improve validation bits-per-byte (`val_bpb`) on a small Transformer model under a fixed local training budget using only modifications to `train.py`?

## File Structure
- `prepare.py` — fixed data loading, tokenization, batch creation, and evaluation utilities.
- `train.py` — editable baseline training script and sampling mode.
- `program.md` — research workflow and constraints.
- `README.md` — this file.
- `results.tsv` — experimental log of runs.
- `results_final.tsv` — final summarized results.
- `execution_log.md` — agent reasoning, summary, and outcomes.
- `logs/` — saved output logs for each run.
- `samples/` — generated sample text outputs.

## Setup (Windows PowerShell)
1. Open PowerShell in this repository folder.
2. Install dependencies if needed:
   ```powershell
   pip install -r requirements.txt
   ```
3. Ensure Python is available as `py` or `python`.

## How to Run Baseline and Experiments
- Run the baseline training script:
  ```powershell
  py -3 train.py
  ```
- Run the sampling mode:
  ```powershell
  py -3 train.py --sample
  ```
- Logs are stored under `logs/` and samples are written to `samples/samples_final.txt`.

## Metric
The primary metric is `val_bpb` (validation bits per byte). Lower values are better.

## Best Result Summary
- Baseline `val_bpb`: `2.938290`
- Best `val_bpb`: `2.532282`
- Relative improvement: `13.82%`

## Output Files
- `results_final.tsv` — final summary of experiment results.
- `execution_log.md` — detailed experiment log and reasoning.
- `logs/` — saved logs for each run, including baseline and experiments.
- `samples/` — generated model text outputs, including `samples_final.txt`.
