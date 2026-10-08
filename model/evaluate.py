#!/usr/bin/env python3
"""
evaluate.py  --  Checkpoint 2: final evaluation of ONE chosen checkpoint.

Run this once, on the model you picked using validation results, so the test
set stays an honest estimate.

    python3 evaluate.py --checkpoint checkpoints/<run>/best.pt \
        --manifest patches/manifest.csv --patch-root patches [--mlflow-run-id <id>]

Writes to eval_out/<run>/ :
    metrics.json, classification_report.txt, predictions.csv
    confusion_matrix.png          counts + row-normalised (= per-class recall)
    actual_vs_predicted.png       random patches per true class, green = correct, red = wrong
    per_disaster.csv              macro-F1 / accuracy per disaster
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from train import CLASSES, K, PatchDataset, build_model, eval_transform, pick_device

COLORS = {"no-damage": "#2a78d6", "minor-damage": "#eda100", "major-damage": "#eb6834", "destroyed": "#e34948"}


def heatmap(ax, m, title, fmt):
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
    ax.set_title(title)
    ax.set_xticks(range(K)); ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, fmt(m[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if m[i, j] > m.max() * 0.55 else "#0b0b0b")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--patch-root", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--grid-n", type=int, default=6, help="patches per class in the actual-vs-predicted grid")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-run-id", default=None, help="log results to this existing run")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    a = ap.parse_args()

    random.seed(a.seed)
    device = pick_device(a.device)
    out = a.out_dir or Path("eval_out") / a.checkpoint.parent.name
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model, _, _ = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    df = pd.read_csv(a.manifest)
    df = df[df.split == a.split].reset_index(drop=True)
    loader = DataLoader(PatchDataset(df, a.patch_root, eval_transform()), a.batch_size, shuffle=False,
                        num_workers=a.num_workers)
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    probs = np.concatenate(probs)
    y_true, y_pred = df.label_idx.to_numpy(), probs.argmax(1)

    macro_f1 = f1_score(y_true, y_pred, labels=range(K), average="macro", zero_division=0)
    acc = float((y_true == y_pred).mean())
    rep = classification_report(y_true, y_pred, labels=range(K), target_names=CLASSES, digits=3, zero_division=0)
    rep_d = classification_report(y_true, y_pred, labels=range(K), target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(K))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    (out / "classification_report.txt").write_text(f"{a.split} split, n={len(df):,}\n"
                                                    f"accuracy {acc:.4f}   macro-F1 {macro_f1:.4f}\n\n{rep}")
    print((out / "classification_report.txt").read_text())

    mi, ma = CLASSES.index("minor-damage"), CLASSES.index("major-damage")
    conf = {"minor_predicted_as_major": float(cmn[mi, ma]), "major_predicted_as_minor": float(cmn[ma, mi])}
    print(f"minor -> major confusion: {conf['minor_predicted_as_major']:.1%}   "
          f"major -> minor: {conf['major_predicted_as_minor']:.1%}")
    metrics = {"split": a.split, "n": len(df), "accuracy": acc, "macro_f1": macro_f1,
               **{f"recall_{c}": rep_d[c]["recall"] for c in CLASSES},
               **{f"precision_{c}": rep_d[c]["precision"] for c in CLASSES}, **conf,
               "checkpoint": str(a.checkpoint), "arch": ck["arch"], "best_epoch": ck.get("epoch")}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pred = df[["path", "label", "scene", "disaster"]].copy()
    pred["pred"] = [CLASSES[i] for i in y_pred]
    pred["confidence"] = probs.max(1).round(4)
    pred.to_csv(out / "predictions.csv", index=False)

    # per-disaster breakdown (does the model transfer between events?)
    rows = []
    for d, g in pred.assign(t=y_true, p=y_pred).groupby("disaster"):
        rows.append({"disaster": d, "n": len(g), "accuracy": (g.t == g.p).mean(),
                     "macro_f1": f1_score(g.t, g.p, labels=range(K), average="macro", zero_division=0)})
    pd.DataFrame(rows).sort_values("n", ascending=False).round(4).to_csv(out / "per_disaster.csv", index=False)

    # confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="white")
    heatmap(ax[0], cm.astype(float), f"Confusion matrix ({a.split}, counts)", lambda v: f"{int(v):,}")
    heatmap(ax[1], cmn, "Row-normalised (diagonal = per-class recall)", lambda v: f"{v:.0%}")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    # actual vs predicted grid
    n = a.grid_n
    fig, axes = plt.subplots(K, n, figsize=(2.2 * n, 2.5 * K), facecolor="white", squeeze=False)
    for r, c in enumerate(CLASSES):
        idx = np.where(y_true == r)[0]
        picks = random.sample(list(idx), min(n, len(idx)))
        for j in range(n):
            ax = axes[r][j]; ax.axis("off")
            if j >= len(picks):
                continue
            i = picks[j]
            with Image.open(Path(a.patch_root) / df.path[i]) as im:
                ax.imshow(im.convert("RGB"))
            ok = y_pred[i] == r
            ax.set_title(f"pred: {CLASSES[y_pred[i]]}\n{probs[i].max():.0%}", fontsize=8,
                         color="#1a7f37" if ok else "#c62828")
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor("#1a7f37" if ok else "#c62828"); s.set_linewidth(3)
            if j == 0:
                ax.text(-0.08, 0.5, f"actual:\n{c}", transform=ax.transAxes, ha="right", va="center", fontsize=9,
                        color=COLORS[c], fontweight="bold")
    fig.suptitle("Actual vs predicted (random patches; green = correct, red = wrong)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "actual_vs_predicted.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    if a.mlflow_run_id:
        import mlflow
        mlflow.set_tracking_uri(a.tracking_uri)
        with mlflow.start_run(run_id=a.mlflow_run_id):
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_artifacts(str(out), artifact_path="test_eval")
        print(f"logged to MLflow run {a.mlflow_run_id}")
    print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()
