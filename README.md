# Property Due-Diligence & Investment Advisor — Building Damage CNN

DSML Batch 42 final project (dibimbing.id) by Brian Hernawan. A CNN fine-tuned to classify
building damage severity (no-damage / minor / major / destroyed) from
satellite imagery, as the condition-assessment component of a larger
property due-diligence and investment advisor. Built on the
[xBD dataset](https://arxiv.org/abs/1911.09296) (Gupta et al., 2019).

**Part of a two-repo project.** This repo trains and evaluates the damage
classifier. The app that serves it (FastAPI + Streamlit UI + advisor chatbot)
is in [`property-agent-advisor`](https://github.com/brianhernawan/property-agent-advisor).

**Status:** EDA, CNN fine-tuning with MLflow tracking, and the test-set
evaluation are complete and in this repo.

## Key results

- 159,794 building-level patches, 4 damage classes, scene-level 70/15/15
  split (zero leakage between near-duplicate crops from the same disaster)
- Staged hyperparameter search: Stage A (unfreeze depth) → Stage B (batch
  size + class imbalance) → Stage C (architecture comparison)
- Test set (n = 24,080, touched once): **85.2% accuracy, 0.727 macro-F1**
- Champion by score: ResNet-50 (val macro-F1 0.6934). **Recommended for
  serving: ResNet-18** (0.6925 — a statistical tie, at about a third of the
  inference cost)
- Weakest class: minor-damage (F1 0.524), mainly confused with major-damage
  (21.8% / 17.7% cross-over) — the two rarest classes, and the two that sit
  next to each other on the severity scale

Full numbers and the read on what's good/bad: [`results.md`](results.md).

## Pipeline

```
Split & prep  →  Stage A        →  Stage B            →  Stage C         →  Register & evaluate
(159,794         (unfreeze depth,  (batch size +          (architecture      (champion → MLflow
 patches,         resnet18,         imbalance handling,    comparison,        registry; test set
 70/15/15          30% subset)       resnet18, 30% subset)  100% of data)      touched once)
 split)
```

## Project structure

```
eda.py                  Checkpoint 1 EDA — class distribution, sample grids, per-disaster breakdown
make_patches.py          Crops building patches from xBD, writes the leakage-safe train/val/test split
train.py                 Fine-tunes one CNN run (model/unfreeze/lr/batch/balance), logs to MLflow
run_experiments.sh        Runs a full stage (A, B, or C) as a sequence of train.py calls
register_best.py          Finds the best MLflow run by validation macro-F1, registers it as champion
evaluate.py               Runs the registered champion on the test set, once
run_pipeline.py            Orchestrates the whole pipeline end to end (wraps the scripts above)
RUNBOOK.md                 Step-by-step guide to running the pipeline manually
results.md                 Full experiment log, metric glossary, and interpretation
requirements.txt           Python dependencies
test_maps_imagery.py       Runs the model(s) on your own satellite screenshots and writes an HTML review gallery
eda_output/                 Checkpoint 1 charts and summary CSV
eval_out/C2_resnet50/       Test-set evaluation artifacts for the champion run
```

## Getting started

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The raw xBD dataset is not included in this repo (too large, and not ours
to redistribute) — download it from the
[xView2 challenge](https://xview2.org/) and point `--data-root` at it.

Then either run the whole pipeline in one go:

```bash
python3 run_pipeline.py --data-root /path/to/xbd --stage all --dry-run   # preview first
python3 run_pipeline.py --data-root /path/to/xbd --stage all
```

or follow the manual, step-by-step version in [`RUNBOOK.md`](RUNBOOK.md).

## Docs

- [`RUNBOOK.md`](RUNBOOK.md) — exact commands, in order, with what to check between stages
- [`results.md`](results.md) — the full 11-run experiment table, column/metric glossary, and the good/bad read on the numbers

## Data and licence

Trained and evaluated on the [xBD dataset](https://xview2.org/) (Gupta et al.,
2019), which is distributed under CC BY-NC-SA 4.0 (non-commercial, share-alike;
check the terms on xview2.org before reuse). The dataset, the trained
checkpoints and the MLflow database are not in this repo. Sample-grid images
built from xBD are also kept out, and any Google Maps/Earth screenshots used
for testing are never committed.
