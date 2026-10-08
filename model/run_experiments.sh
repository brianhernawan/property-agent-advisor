#!/usr/bin/env bash
# run_experiments.sh -- the Checkpoint 2 experiment plan, one run at a time.
#
#   ./run_experiments.sh A            # stage A: how much of the network to unfreeze
#   ./run_experiments.sh B            # stage B: batch size + imbalance handling
#   ./run_experiments.sh C            # stage C: architecture comparison
#
# Stages are run in order; after A and B, read the MLflow UI and set the winners:
#   BEST_UNFREEZE=full BEST_LR=1e-4 BEST_BS=64 BEST_BALANCE=weights ./run_experiments.sh C
#
# Runs are sequential on purpose: one GPU / one MacBook cannot train several models
# at once without each of them slowing down.
set -euo pipefail

MANIFEST=${MANIFEST:-patches/manifest.csv}
ROOT=${ROOT:-patches}
SUBSET=${SUBSET:-0.25}            # fraction of train/val used in stages A and B
EPOCHS=${EPOCHS:-8}
FULL_EPOCHS=${FULL_EPOCHS:-6}     # stage C runs on the full data, so fewer epochs
C_SUBSET=${C_SUBSET:-1.0}         # stage C data fraction; 1.0 = all training patches

BEST_UNFREEZE=${BEST_UNFREEZE:-last}
BEST_LR=${BEST_LR:-1e-4}
BEST_BS=${BEST_BS:-64}
BEST_BALANCE=${BEST_BALANCE:-weights}
BEST_LOSS=${BEST_LOSS:-ce}

run() { echo; echo "=== $1 ==="; python3 train.py --manifest "$MANIFEST" --patch-root "$ROOT" --run-name "$@"; }

case "${1:-}" in
  A)  # resnet18, subset: head only vs last block vs full fine-tune
    run A1_r18_head --model resnet18 --unfreeze head --head-lr 1e-3 --subset "$SUBSET" --epochs "$EPOCHS"
    run A2_r18_last --model resnet18 --unfreeze last --lr 1e-4 --head-lr 1e-3 --subset "$SUBSET" --epochs "$EPOCHS"
    run A3_r18_full --model resnet18 --unfreeze full --lr 1e-4 --head-lr 1e-3 --subset "$SUBSET" --epochs "$EPOCHS"
    ;;
  B)  # resnet18, subset, best unfreeze/lr from A: batch size, then imbalance handling
    common="--model resnet18 --unfreeze $BEST_UNFREEZE --lr $BEST_LR --subset $SUBSET --epochs $EPOCHS"
    run B1_bs32  $common --batch-size 32  --balance weights
    run B2_bs128 $common --batch-size 128 --balance weights
    run B3_sqrtw $common --batch-size "$BEST_BS" --balance sqrt_weights
    run B4_sampler $common --batch-size "$BEST_BS" --balance sampler
    run B5_focal $common --batch-size "$BEST_BS" --balance weights --loss focal
    ;;
  C)  # full data, best config, three architectures
    common="--unfreeze $BEST_UNFREEZE --lr $BEST_LR --batch-size $BEST_BS --balance $BEST_BALANCE --loss $BEST_LOSS --epochs $FULL_EPOCHS --subset $C_SUBSET"
    run C1_resnet18 --model resnet18 $common
    run C2_resnet50 --model resnet50 $common
    run C3_effb0    --model efficientnet_b0 $common
    ;;
  *) echo "usage: $0 {A|B|C}"; exit 1 ;;
esac
