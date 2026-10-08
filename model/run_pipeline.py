#!/usr/bin/env python3
"""
run_pipeline.py  --  Checkpoint 2 pipeline, orchestrated as one script.

This replaces typing the RUNBOOK.md commands into the terminal by hand. It
calls the existing project scripts (make_patches.py, train.py,
register_best.py, evaluate.py) in the same order, with the same staged
design: cheap experiments first on a data subset (Stage A, B), the expensive
full-data comparison only once (Stage C), then register the winner and
evaluate it on the test set exactly once.

Keep this file next to make_patches.py / train.py / register_best.py /
evaluate.py / requirements.txt -- it does not reimplement them, it drives
them as subprocesses, so it is also the reference for how a similar future
project's pipeline should be wired up.

Examples
--------
    # see every command this would run, without running anything
    python3 run_pipeline.py --data-root /path/to/xbd --stage all --dry-run

    # run everything end to end: patches -> stage A -> B -> C -> register -> evaluate
    python3 run_pipeline.py --data-root /path/to/xbd --stage all

    # or step by step, reading the MLflow UI between stages (recommended the
    # first time, so you can sanity-check each stage before paying for the next)
    python3 run_pipeline.py --data-root /path/to/xbd --stage patches
    python3 run_pipeline.py --stage A
    python3 run_pipeline.py --stage B --best-unfreeze full --best-lr 1e-4
    python3 run_pipeline.py --stage C --best-unfreeze full --best-lr 1e-4 \\
        --best-bs 64 --best-balance sqrt_weights
    python3 run_pipeline.py --stage register
    python3 run_pipeline.py --stage evaluate --eval-run-name C2_resnet50

    # re-run just the evaluation for a specific checkpoint later
    python3 run_pipeline.py --stage evaluate --eval-run-name C2_resnet50 \\
        --mlflow-run-id 5f8212a0eb574cbf834734feeaf4b22a
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run(cmd: list[str], dry_run: bool, capture: bool = False) -> str:
    """Print then execute a command (relative to this file's folder). Raises on failure."""
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}")
    if dry_run:
        return ""
    result = subprocess.run(cmd, cwd=HERE, check=True,
                             capture_output=capture, text=True)
    if capture:
        print(result.stdout)
        return result.stdout
    return ""


def py(args: list[str]) -> list[str]:
    return [sys.executable] + args


# ----------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------
def stage_patches(a) -> None:
    banner("Step 0 -- build patches + manifest (scene-level 70/15/15 split)")
    if not a.data_root:
        sys.exit("--data-root is required for --stage patches (path to the xBD dataset)")
    run(py(["make_patches.py", "--data-root", a.data_root, "--out-dir", a.root,
            "--workers", str(a.workers)]), a.dry_run)


def _train(a, run_name: str, extra: list[str]) -> None:
    run(py(["train.py", "--manifest", a.manifest, "--patch-root", a.root,
            "--run-name", run_name, "--experiment", a.experiment] + extra), a.dry_run)


def stage_a(a) -> None:
    banner("Stage A -- unfreeze depth (resnet18, subset, cheap)")
    common = ["--model", "resnet18", "--subset", str(a.subset), "--epochs", str(a.epochs)]
    _train(a, "A1_r18_head", common + ["--unfreeze", "head", "--head-lr", "1e-3"])
    _train(a, "A2_r18_last", common + ["--unfreeze", "last", "--lr", "1e-4", "--head-lr", "1e-3"])
    _train(a, "A3_r18_full", common + ["--unfreeze", "full", "--lr", "1e-4", "--head-lr", "1e-3"])
    print("\nRead the MLflow UI, then re-run with --stage B --best-unfreeze <winner> --best-lr <winner's lr>.")


def stage_b(a) -> None:
    banner("Stage B -- batch size + imbalance handling (resnet18, subset, cheap)")
    common = ["--model", "resnet18", "--unfreeze", a.best_unfreeze, "--lr", a.best_lr,
              "--subset", str(a.subset), "--epochs", str(a.epochs)]
    _train(a, "B1_bs32", common + ["--batch-size", "32", "--balance", "weights"])
    _train(a, "B2_bs128", common + ["--batch-size", "128", "--balance", "weights"])
    _train(a, "B3_sqrtw", common + ["--batch-size", a.best_bs, "--balance", "sqrt_weights"])
    _train(a, "B4_sampler", common + ["--batch-size", a.best_bs, "--balance", "sampler"])
    _train(a, "B5_focal", common + ["--batch-size", a.best_bs, "--balance", "weights", "--loss", "focal"])
    print("\nRead the MLflow UI, then re-run with --stage C and all four --best-* flags set.")


def stage_c(a) -> None:
    banner("Stage C -- architecture comparison (100% of data, winning recipe from A+B)")
    common = ["--unfreeze", a.best_unfreeze, "--lr", a.best_lr, "--batch-size", a.best_bs,
              "--balance", a.best_balance, "--loss", a.best_loss,
              "--epochs", str(a.full_epochs), "--subset", str(a.c_subset)]
    _train(a, "C1_resnet18", ["--model", "resnet18"] + common)
    _train(a, "C2_resnet50", ["--model", "resnet50"] + common)
    _train(a, "C3_effb0", ["--model", "efficientnet_b0"] + common)
    print("\nRead the MLflow UI, then run --stage register to pick the winner and register it.")


def stage_register(a) -> tuple[str | None, str | None]:
    """Returns (run_id, run_name) of the champion, parsed from register_best.py's own output."""
    banner("Register the best run (highest validation macro-F1) as the registry champion")
    out = run(py(["register_best.py", "--experiment", a.experiment, "--name", a.model_name]),
              a.dry_run, capture=True)
    if a.dry_run:
        return None, None
    m = re.search(r"from run (\S+)\s+\[(\S+),", out)
    if not m:
        print("Could not parse the champion run id/name from register_best.py output -- "
              "pass --eval-run-name explicitly to --stage evaluate.")
        return None, None
    run_id, run_name = m.group(1), m.group(2)
    print(f"\nparsed champion -> run_id={run_id}  run_name={run_name}")
    return run_id, run_name


def stage_evaluate(a, run_id: str | None = None, run_name: str | None = None) -> None:
    banner("Evaluate the champion on the TEST set (touched exactly once)")
    run_name = run_name or a.eval_run_name
    if not run_name:
        if a.dry_run:
            run_name = "<champion-run-name, known only after --stage register actually runs>"
        else:
            sys.exit("--stage evaluate needs --eval-run-name <run>, or run --stage register first "
                      "(or --stage all, which chains the two automatically)")
    ckpt = f"checkpoints/{run_name}/best.pt"
    cmd = ["evaluate.py", "--checkpoint", ckpt, "--manifest", a.manifest, "--patch-root", a.root]
    run_id = run_id or a.mlflow_run_id
    if run_id:
        cmd += ["--mlflow-run-id", run_id]
    run(py(cmd), a.dry_run)


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                     choices=["patches", "A", "B", "C", "register", "evaluate", "all"])
    ap.add_argument("--dry-run", action="store_true", help="print every command, run nothing")

    # shared paths
    ap.add_argument("--data-root", default=None, help="path to the raw xBD dataset (needed for --stage patches)")
    ap.add_argument("--root", default="patches", help="patch output dir (default: patches)")
    ap.add_argument("--manifest", default="patches/manifest.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--experiment", default="property-dd-cnn")
    ap.add_argument("--model-name", default="property-dd-damage-cnn", help="MLflow registered model name")

    # stage A/B sizing
    ap.add_argument("--subset", type=float, default=0.3, help="train/val fraction used in stages A and B")
    ap.add_argument("--epochs", type=int, default=8, help="epochs for stages A and B")

    # stage C sizing
    ap.add_argument("--full-epochs", type=int, default=6, help="epochs for stage C (full data, so fewer)")
    ap.add_argument("--c-subset", type=float, default=1.0, help="train/val fraction used in stage C")

    # winning recipe, carried from A/B into C (and shown on the hyperparams slide)
    ap.add_argument("--best-unfreeze", default="last", choices=["head", "last", "full"])
    ap.add_argument("--best-lr", default="1e-4")
    ap.add_argument("--best-bs", default="64")
    ap.add_argument("--best-balance", default="weights",
                     choices=["none", "weights", "sqrt_weights", "sampler"])
    ap.add_argument("--best-loss", default="ce", choices=["ce", "focal"])

    # evaluate
    ap.add_argument("--eval-run-name", default=None, help="checkpoints/<this>/best.pt (e.g. C2_resnet50)")
    ap.add_argument("--mlflow-run-id", default=None, help="log test metrics onto this existing MLflow run")

    a = ap.parse_args()

    if a.stage == "patches":
        stage_patches(a)
    elif a.stage == "A":
        stage_a(a)
    elif a.stage == "B":
        stage_b(a)
    elif a.stage == "C":
        stage_c(a)
    elif a.stage == "register":
        stage_register(a)
    elif a.stage == "evaluate":
        stage_evaluate(a)
    elif a.stage == "all":
        stage_patches(a)
        stage_a(a)
        stage_b(a)
        stage_c(a)
        run_id, run_name = stage_register(a)
        stage_evaluate(a, run_id=run_id, run_name=run_name)

    banner("done")


if __name__ == "__main__":
    main()
