# TinyStories Autoresearch Program

This document defines an autonomous research workflow for improving a small language model trained on the TinyStories GPT-4 Clean dataset. The workflow is modeled after Andrej Karpathy's autoresearch paradigm: the agent modifies only the experiment code, runs fixed-budget experiments, evaluates each change using a single metric, and keeps only changes that improve performance under controlled conditions.

Dataset: `karpathy/tinystories-gpt4-clean` (https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/tree/main)

Primary metric: validation bits per byte (`val_bpb`)  
Goal: minimize `val_bpb`


## 1. Research Problem and Scope

### Research question

Can an autonomous LLM agent improve a small Transformer language model trained on the TinyStories GPT-4 Clean dataset under a fixed compute budget?

### Dataset

The project uses the TinyStories GPT-4 Clean dataset from Hugging Face. The dataset contains GPT-4-generated short stories and is suitable for training small language models because the stories are narrow in scope, relatively simple, and lower in entropy than general web text.

### Optimization metric

The agent must optimize one scalar metric:

```text
val_bpb
```

Lower `val_bpb` indicates better validation performance.

### Fixed resource budget

Each experiment must run under the same resource constraints:

```text
training budget: 180 seconds wall-clock training time
hard timeout: 10 minutes total runtime
device: one local GPU, Apple Silicon MPS device, or CPU if no accelerator is available
soft memory limit: 8 GB
no external API calls during training
no new dependencies after setup
```

### Why this problem is suitable for autonomous experimentation

This project is suitable because it has:

1. a fixed dataset;
2. a single quantitative evaluation metric;
3. a small and bounded experimental search space;
4. fast iteration cycles;
5. many testable design choices, including model size, optimizer, learning rate, dropout, batch size, gradient clipping, and learning-rate scheduling.

The agent can therefore conduct meaningful autonomous experimentation while remaining constrained by a stable evaluation protocol.


## 2. File Structure and Boundaries

The project uses the following file structure:

```text
tinystories-autoresearch/
├── README.md
├── program.md
├── prepare.py
├── train.py
├── requirements.txt
├── results_example.tsv
└── .gitignore
```

### Read-only files

The agent must not modify:

```text
prepare.py
README.md
requirements.txt
program.md
```

`prepare.py` defines the fixed research environment, including:

- dataset loading;
- train/validation split;
- tokenizer or byte-level encoding;
- dataloader construction;
- constants;
- validation metric;
- `evaluate_bpb`;
- utility functions for timing and reproducibility.

These files are read-only to prevent the agent from changing the benchmark, data split, metric, or evaluation procedure.

### Editable file

The agent may modify only:

```text
train.py
```

The agent may change:

- model architecture;
- number of layers;
- embedding size;
- number of attention heads;
- optimizer;
- learning rate;
- learning-rate schedule;
- dropout;
- batch size;
- gradient accumulation;
- weight decay;
- gradient clipping;
- training loop implementation;
- sampling function inside `train.py`.

### Prohibited actions

The agent must not:

- modify `prepare.py`;
- change the dataset;
- change the validation split;
- change `evaluate_bpb`;
- install new dependencies;
- use external APIs;
- hard-code validation data;
- use test-set leakage;
- silently delete failed experiments;
- modify the logging format.

The purpose of these boundaries is to separate fixed infrastructure from editable experimental code. This allows autonomous experimentation while preserving comparability across runs.



## 3. Setup

Before beginning experiments, the agent must inspect:

```text
README.md
program.md
prepare.py
train.py
requirements.txt
```

The agent must verify that:

1. the TinyStories GPT-4 Clean dataset can be loaded;
2. the training and validation split is fixed;
3. `evaluate_bpb` is defined in `prepare.py`;
4. `train.py` can run successfully;
5. the first run establishes a baseline;
6. `results.tsv` exists with the correct header.

The initial `results.tsv` header must be:

```text
experiment_id	commit	val_bpb	memory_gb	params_M	steps	status	description
```

## 4. Output Format

At the end of each successful experiment, `train.py` must print the following lines:

```text
---
val_bpb:          <float>
training_seconds: <float>
total_seconds:    <float>
peak_memory_mb:   <float>
num_params_M:     <float>
num_steps:        <int>
```

The agent should extract the result using:

```bash
grep "^val_bpb:\|^peak_memory_mb:\|^num_params_M:\|^num_steps:" run.log
```


## 5. Logging

After every experiment, the agent must append one row to `results.tsv`.

The format is:

```text
experiment_id	commit	val_bpb	memory_gb	params_M	steps	status	description
```

Column definitions:

1. `commit`: short git commit hash.
2. `val_bpb`: validation bits per byte. Use `0.000000` for crashes or timeouts.
3. `memory_gb`: peak memory usage in GB.
4. `params_M`: number of parameters in millions.
5. `steps`: number of optimization steps completed.
6. `status`: one of `keep`, `discard`, `crash`, or `timeout`.
7. `complexity_delta`: approximate line-count change relative to the previous kept version.
8. `description`: short description of the experimental idea.

The agent must not delete previous rows. Failed experiments must also be logged.


## 6. Simplicity Criterion

The agent should prefer simple, robust, and interpretable changes.

A change should be kept if:

1. it lowers `val_bpb`;
2. it does not substantially increase code complexity;
3. it does not sharply increase memory usage;
4. it does not make the training process fragile.

Decision guidelines:

```text
val_bpb improvement > 0.005: usually keep
val_bpb improvement between 0.001 and 0.005: keep only if complexity is modest
val_bpb improvement < 0.001: keep only if the change simplifies code
equal val_bpb: keep only if code becomes simpler
worse val_bpb: discard
crash or timeout: discard unless caused by a trivial fixable error
```

A small performance gain is not worth keeping if it requires fragile, unreadable, or highly specialized code.


## 7. Experiment Loop

The agent must run experiments on a dedicated branch:

```bash
git checkout -b autoresearch/tinystories-run
```

Each iteration follows this loop.

### Step 1: Inspect current state

```bash
git status
git log --oneline -5
```

Read the current best result from `results.tsv`.

### Step 2: Propose one idea

The agent proposes exactly one specific modification to `train.py`.

Examples include:

- change learning rate;
- adjust model depth;
- adjust embedding size;
- change number of attention heads;
- add or remove dropout;
- change optimizer betas;
- add cosine learning-rate decay;
- change batch size;
- add gradient clipping;
- simplify the attention block;
- modify normalization placement.

The agent should avoid changing many things at once unless the change is a coherent single design idea.

### Step 3: Modify only `train.py`

The agent edits `train.py` and must not modify any read-only files.

### Step 4: Commit the change

```bash
git add train.py
git commit -m "experiment: <short description>"
```

### Step 5: Run the experiment

```bash
python train.py > run.log 2>&1
```

If the experiment runs longer than 10 minutes, kill it and mark it as `timeout`.

### Step 6: Extract metrics

```bash
grep "^val_bpb:\|^peak_memory_mb:\|^num_params_M:\|^num_steps:" run.log
```

### Step 7: Decide whether to keep or discard

Compare the new `val_bpb` with the best previous kept result.

Keep the commit if:

- `val_bpb` is lower; and
- the simplicity criterion is satisfied.

Discard the commit if:

- `val_bpb` is worse;
- the run crashes;
- the run times out;
- memory use becomes excessive;
- the code becomes too complex relative to the improvement.

If discarding, revert to the previous kept commit:

```bash
git reset --hard <previous_best_commit>
```

### Step 8: Handle crashes

If the run crashes, inspect:

```bash
tail -n 50 run.log
```

If the crash is caused by a trivial syntax, import, or device error, the agent may fix it once and rerun.

If the idea is fundamentally broken, the agent must:

1. log the experiment as `crash`;
2. revert to the previous kept commit;
3. move to a new idea.

### Step 9: Log the result

Append one row to `results.tsv`.

### Step 10: Continue or stop for review

Continue the loop unless a human stop-and-check point has been reached.


## 8. Human Stop-and-Check Points

Unlike a fully indefinite autonomous loop, this project requires human review at several points.

### Stop 1: After baseline setup

The agent must stop after the first successful baseline run.

Human review should check:

- whether the dataset loads correctly;
- whether the validation split is fixed;
- whether `val_bpb` is computed correctly;
- whether the baseline completes within the time budget;
- whether the file boundaries are respected.

Success criterion:

```text
baseline run completes successfully and produces valid val_bpb
```

If the check fails, the infrastructure must be corrected before autonomous experimentation continues.

### Stop 2: After 10 kept improvements or 30 total experiments

The agent must stop after either:

```text
10 kept improvements
```

or

```text
30 total experiments
```

whichever comes first.

Human review should check:

- whether performance improvements are real;
- whether code complexity is increasing too much;
- whether memory usage remains acceptable;
- whether the agent is exploring diverse ideas;
- whether results suggest overfitting to the validation set.

Success criterion:

```text
best val_bpb improves without excessive complexity or memory growth
```

If the check fails, revert to the best clean commit and redesign the search strategy.

### Stop 3: Before final merge

Before merging the final result into `main`, the human must review:

- final `train.py`;
- full `results.tsv`;
- final `run.log`;
- reproducibility from scratch;
- sample generated stories.

Success criterion:

```text
final run is reproducible, code is understandable, and generated samples are qualitatively reasonable
```

If the check fails, do not merge. Return to the best reproducible commit.


## 9. Full Autonomy versus Human-in-the-Loop Oversight

Full autonomy is appropriate when:

- the task is low risk;
- the evaluation metric is stable;
- failed experiments are cheap;
- the agent cannot change the evaluation protocol;
- the search space is technically bounded.

Human-in-the-loop oversight is necessary when:

- the agent may exploit the metric;
- the agent may overfit to validation data;
- results require substantive interpretation;
- compute cost may escalate;
- code complexity may grow faster than performance;
- the research output will be used for academic reporting.

In this project, indefinite autonomy is risky because the agent may over-optimize `val_bpb`, increase model complexity unnecessarily, or produce a model that scores well but generates poor TinyStories-style samples.

---

## 10. Final Output

At the end of the project, the repository should contain:

```text
README.md
program.md
prepare.py
train.py
requirements.txt
results_final.tsv
best_run.log
samples_final.txt
.gitignore
```

The final report should state:

1. baseline `val_bpb`;
2. best `val_bpb`;
3. percentage improvement;
4. best experiment description;
5. memory usage;
6. number of total experiments;
7. number of kept changes;
8. main failed ideas;
9. final interpretation.
