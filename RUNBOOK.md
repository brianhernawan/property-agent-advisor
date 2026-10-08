# Checkpoint 2 runbook: fine-tuning and evaluation

Order of operations. Every command runs from the folder holding these scripts.

## -1. Checkpoint 1 EDA (once, if you haven't already)
```bash
python3 eda.py --data-root /Users/briancahernawan/Documents/data_science/fp1/xbd --output-dir eda_output
```
`eda.py` replaces the earlier `explore_xbd_dataset_v01.py` and `xbd_disaster_eda_v01.py` —
delete those two, this one script now produces every chart used in the Checkpoint 1 deck
(class distribution, sample grid, per-disaster-type breakdown, per-disaster stacked chart,
disaster-type sample grid, summary CSV, and the pairing/corruption report).

## 0. Setup (once)
```bash
source ../venv/bin/activate            # your existing venv
pip install -r requirements.txt
export MLFLOW_DISABLE_AGENT_HINT=1     # optional, silences an MLflow hint line
```

## 1. Build patches + manifest (once, ~10-30 min)
```bash
python3 make_patches.py --data-root /Users/briancahernawan/Documents/data_science/fp1/xbd --out-dir patches --workers 4
```
Check `patches/split_report.txt`: each split should show a similar class mix.
Expect roughly 160K patches, ~3 GB. Do not regenerate the split after you start training.

## 2. Speed test, then decide how much to run overnight
```bash
python3 train.py --manifest patches/manifest.csv --patch-root patches --model resnet50 --benchmark 30 --batch-size 32
python3 train.py --manifest patches/manifest.csv --patch-root patches --model resnet18 --benchmark 30
```
It prints images/s and minutes per epoch on the full train set. Use that to set `SUBSET` and `EPOCHS` (stages A/B) and `C_SUBSET` (stage C).
If resnet50 is over ~60 min/epoch on the full data, run stage C on Colab instead.

## 3. Experiments (sequential, keep the Mac plugged in and awake)
```bash
caffeinate -i ./run_experiments.sh A       # unfreeze depth        (resnet18, 25% subset)
mlflow ui --backend-store-uri sqlite:///mlflow.db     # read results, pick winners
BEST_UNFREEZE=<x> BEST_LR=<x> caffeinate -i ./run_experiments.sh B    # batch size + imbalance handling
BEST_UNFREEZE=<x> BEST_LR=<x> BEST_BS=<x> BEST_BALANCE=<x> BEST_LOSS=<x> caffeinate -i ./run_experiments.sh C   # 3 architectures, full data
```
A crashed run resumes with the same command plus `--resume` (run it directly via `python3 train.py ... --run-name <same name> --resume`).

## 4. Pick the winner, register, evaluate once on test
```bash
python3 register_best.py                                  # highest validation macro-F1 -> MLflow Model Registry
python3 evaluate.py --checkpoint checkpoints/<best run>/best.pt \
    --manifest patches/manifest.csv --patch-root patches --mlflow-run-id <run id>
```
The test set is used only here, once.

## What each rubric item maps to
| Rubric item | Where it comes from |
|---|---|
| Model selection and reasoning | ResNet paper (residual learning, Table 1 FLOPs), VGG as baseline, stage C comparison table |
| Hyperparameter experiments | MLflow runs A1-A3, B1-B5, C1-C3 (compare in the UI, export the table) |
| Loss/accuracy curves | `checkpoints/<run>/curves.png` (also stored in MLflow) |
| Macro-F1 + per-class recall | `eval_out/<run>/classification_report.txt`, `metrics.json` |
| Confusion matrix | `eval_out/<run>/confusion_matrix.png` |
| Actual vs predicted | `eval_out/<run>/actual_vs_predicted.png` |
| Experiment tracking | `mlflow.db` + registered model `property-dd-damage-cnn` (alias `champion`) |

## Note on the rename
The MLflow experiment/model defaults changed from `proptech-cnn` / `proptech-damage-cnn`
to `property-dd-cnn` / `property-dd-damage-cnn`, to match the new project title. This does
**not** touch your existing `mlflow.db` — old runs stay recorded under the old experiment
name. Two ways to reconcile, pick one:
  - Easiest: rename the already-registered model in place, no re-training needed —
    `python3 -c "from mlflow.tracking import MlflowClient; MlflowClient().rename_registered_model('proptech-damage-cnn', 'property-dd-damage-cnn')"`
  - Or: keep passing `--experiment proptech-cnn --name proptech-damage-cnn` to
    `train.py` / `register_best.py` if you'd rather leave the old names alone.
