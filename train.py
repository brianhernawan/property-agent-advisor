#!/usr/bin/env python3
"""
train.py  --  Checkpoint 2: fine-tune a pretrained CNN on xBD building patches,
tracking every run in MLflow.

Examples:
    # quick speed test on this machine (no MLflow run created)
    python3 train.py --manifest patches/manifest.csv --patch-root patches --model resnet50 --benchmark 30

    # sweep run on 25% of the data
    python3 train.py --manifest patches/manifest.csv --patch-root patches \
        --model resnet18 --unfreeze last --lr 1e-4 --head-lr 1e-3 --batch-size 64 \
        --epochs 8 --subset 0.25 --run-name A2_r18_last

The TEST split is never touched here. Model selection uses validation macro-F1;
evaluate.py runs the test set once, on the chosen checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
K = len(CLASSES)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats (pretrained backbones)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
class Rot90:
    """Random multiple-of-90-degree rotation: satellite crops have no canonical 'up'."""
    def __call__(self, img):
        k = random.randint(0, 3)
        return img.rotate(90 * k) if k else img


def train_transform(size=224):
    return T.Compose([
        Rot90(),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class PatchDataset(Dataset):
    def __init__(self, df, root, tf):
        self.paths = [str(Path(root) / p) for p in df["path"]]
        self.y = df["label_idx"].to_numpy()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, int(self.y[i])


def stratified_fraction(df, frac, seed):
    if frac >= 1:
        return df
    return pd.concat([g.sample(frac=frac, random_state=seed) for _, g in df.groupby("label_idx")]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def build_model(name, pretrained=True, dropout=0.2):
    m = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
    if name.startswith("resnet"):
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, K))
        head, last = m.fc, [m.layer4]
    elif name.startswith("efficientnet"):
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, K))
        head, last = m.classifier, [m.features[-2], m.features[-1]]
    else:
        raise ValueError(f"unsupported model {name}")
    return m, head, last


def set_trainable(model, head, last, mode):
    for p in model.parameters():
        p.requires_grad = mode == "full"
    if mode in ("head", "last"):
        for p in head.parameters():
            p.requires_grad = True
    if mode == "last":
        for blk in last:
            for p in blk.parameters():
                p.requires_grad = True


def freeze_bn_if_frozen(model):
    """BatchNorm layers whose weights are frozen must also stop updating running stats."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, 1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logpt.exp()) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# loops
# ----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, amp=False, max_steps=None):
    train = optimizer is not None
    model.train(train)
    if train:
        freeze_bn_if_frozen(model)
    tot_loss, n, ys, ps = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for step, (x, y) in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * len(y)
            n += len(y)
            ys.append(y.cpu().numpy())
            ps.append(out.argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    return tot_loss / n, float((y_true == y_pred).mean()), y_true, y_pred


def metrics_from(y_true, y_pred):
    rec = recall_score(y_true, y_pred, labels=list(range(K)), average=None, zero_division=0)
    return f1_score(y_true, y_pred, labels=list(range(K)), average="macro", zero_division=0), rec


def plot_curves(hist, path):
    ep = hist["epoch"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), facecolor="white")
    for a, (k1, k2, title) in zip(ax, [("train_loss", "val_loss", "Loss"), ("train_acc", "val_acc", "Accuracy"),
                                       (None, "val_macro_f1", "Validation macro-F1")]):
        if k1:
            a.plot(ep, hist[k1], color="#2a78d6", lw=2, marker="o", ms=4, label="train")
        a.plot(ep, hist[k2], color="#eb6834", lw=2, marker="o", ms=4, label="validation")
        a.set_title(title)
        a.set_xlabel("epoch")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e1e0d9", lw=0.8)
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--patch-root", required=True, type=Path)
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--unfreeze", default="last", choices=["head", "last", "full"])
    ap.add_argument("--lr", type=float, default=1e-4, help="LR for unfrozen backbone layers")
    ap.add_argument("--head-lr", type=float, default=None, help="LR for the new classifier head (default 10x --lr)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--sched", default="cosine", choices=["cosine", "none"])
    ap.add_argument("--balance", default="weights", choices=["none", "weights", "sqrt_weights", "sampler"])
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"])
    ap.add_argument("--subset", type=float, default=1.0, help="stratified fraction of train and val to use (0-1]")
    ap.add_argument("--patience", type=int, default=4, help="early stop after N epochs without val macro-F1 gain (0=off)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (only for smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (CUDA only)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--experiment", default="property-dd-cnn")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--resume", action="store_true", help="continue from <ckpt-dir>/<run-name>/last.pt")
    ap.add_argument("--benchmark", type=int, default=0, help="time N training steps, print img/s, exit")
    a = ap.parse_args()
    if a.head_lr is None:
        a.head_lr = a.lr * 10

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = pick_device(a.device)
    print(f"device: {device}")

    df = pd.read_csv(a.manifest)
    tr = stratified_fraction(df[df.split == "train"], a.subset, a.seed)
    va = stratified_fraction(df[df.split == "val"], a.subset, a.seed)
    counts = np.bincount(tr.label_idx, minlength=K)
    print(f"train {len(tr):,}  val {len(va):,}  class counts (train): {dict(zip(CLASSES, counts.tolist()))}")

    inv = counts.sum() / (K * np.maximum(counts, 1))
    cls_w = {"none": None, "weights": inv, "sqrt_weights": np.sqrt(inv), "sampler": None}[a.balance]
    w_tensor = torch.tensor(cls_w, dtype=torch.float32, device=device) if cls_w is not None else None
    criterion = FocalLoss(w_tensor) if a.loss == "focal" else nn.CrossEntropyLoss(weight=w_tensor)

    pin = device.type == "cuda"
    nw = a.num_workers
    if a.balance == "sampler":
        sw = (1.0 / np.maximum(counts, 1))[tr.label_idx.to_numpy()]
        sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), num_samples=len(tr), replacement=True)
        tr_loader = DataLoader(PatchDataset(tr, a.patch_root, train_transform()), a.batch_size, sampler=sampler,
                               num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    else:
        tr_loader = DataLoader(PatchDataset(tr, a.patch_root, train_transform()), a.batch_size, shuffle=True,
                               num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    va_loader = DataLoader(PatchDataset(va, a.patch_root, eval_transform()), a.batch_size * 2, shuffle=False,
                           num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)

    model, head, last = build_model(a.model, pretrained=not a.no_pretrained)
    set_trainable(model, head, last, a.unfreeze)
    model.to(device)
    head_ids = {id(p) for p in head.parameters()}
    head_p = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    body_p = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_p, "lr": a.head_lr}] + ([{"params": body_p, "lr": a.lr}] if body_p else [])
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs) if a.sched == "cosine" else None
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train_params:,}")

    # ---- speed test --------------------------------------------------------
    if a.benchmark:
        t0, seen = time.time(), 0
        model.train(); freeze_bn_if_frozen(model)
        for step, (x, y) in enumerate(tr_loader):
            if step == 3:  # skip warm-up steps
                t0, seen = time.time(), 0
            if step >= a.benchmark + 3:
                break
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            if device.type == "mps":
                torch.mps.synchronize()
            seen += len(y)
        ips = seen / (time.time() - t0)
        print(f"\n~{ips:.1f} images/s  ->  ~{len(tr) / ips / 60:.1f} min per epoch on {len(tr):,} train patches")
        return

    # ---- run setup / resume --------------------------------------------------
    run_name = a.run_name or f"{a.model}_{a.unfreeze}_lr{a.lr:g}_bs{a.batch_size}_{a.balance}"
    ck_dir = a.ckpt_dir / run_name
    ck_dir.mkdir(parents=True, exist_ok=True)
    hist = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1"]
            + [f"val_recall_{c}" for c in CLASSES]}
    start_ep, best_f1, best_ep, run_id, bad = 0, -1.0, -1, None, 0
    last_pt = ck_dir / "last.pt"
    if a.resume and last_pt.exists():
        s = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        if sched and s.get("sched"):
            sched.load_state_dict(s["sched"])
        hist, start_ep, best_f1, best_ep, run_id = s["hist"], s["epoch"] + 1, s["best_f1"], s["best_ep"], s["run_id"]
        print(f"resumed from epoch {start_ep}")

    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        if run_id is None:
            mlflow.log_params({**{k: v for k, v in vars(a).items() if k not in ("manifest", "patch_root", "ckpt_dir")},
                               "n_train": len(tr), "n_val": len(va), "trainable_params": n_train_params,
                               "device": str(device), "class_counts_train": json.dumps(counts.tolist()),
                               "class_weights": json.dumps(None if cls_w is None else np.round(cls_w, 3).tolist())})
        for ep in range(start_ep, a.epochs):
            t0 = time.time()
            tl, ta, _, _ = run_epoch(model, tr_loader, criterion, device, opt, a.amp)
            vl, vacc, yt, yp = run_epoch(model, va_loader, criterion, device)
            f1, rec = metrics_from(yt, yp)
            if sched:
                sched.step()
            row = {"epoch": ep + 1, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": vacc, "val_macro_f1": f1,
                   **{f"val_recall_{c}": float(r) for c, r in zip(CLASSES, rec)}}
            for k, v in row.items():
                hist[k].append(v)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=ep + 1)
            print(f"ep {ep + 1:>2}/{a.epochs}  loss {tl:.3f}/{vl:.3f}  acc {ta:.3f}/{vacc:.3f}  val macroF1 {f1:.3f}  "
                  f"recall " + " ".join(f"{r:.2f}" for r in rec) + f"  ({time.time() - t0:.0f}s)")
            if f1 > best_f1:
                best_f1, best_ep, bad = f1, ep + 1, 0
                torch.save({"model": model.state_dict(), "arch": a.model, "classes": CLASSES, "epoch": ep + 1,
                            "val_macro_f1": f1, "args": vars(a)}, ck_dir / "best.pt")
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
                        "hist": hist, "epoch": ep, "best_f1": best_f1, "best_ep": best_ep, "run_id": run.info.run_id},
                       last_pt)
            if a.patience and bad >= a.patience:
                print(f"early stop: no val macro-F1 gain for {a.patience} epochs")
                break

        # ---- wrap up: best weights, curves, model artifact -----------------------
        mlflow.log_metrics({"best_val_macro_f1": best_f1, "best_epoch": best_ep})
        pd.DataFrame(hist).to_csv(ck_dir / "history.csv", index=False)
        plot_curves(hist, ck_dir / "curves.png")
        mlflow.log_artifact(str(ck_dir / "history.csv"))
        mlflow.log_artifact(str(ck_dir / "curves.png"))
        best = torch.load(ck_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"])
        example = np.zeros((1, 3, 224, 224), dtype=np.float32)
        try:  # newer MLflow defaults to a traced 'pt2' format; plain pickle is simpler and version-proof
            mlflow.pytorch.log_model(model.cpu(), "model", serialization_format="pickle", input_example=example)
        except TypeError:  # older MLflow has no serialization_format argument
            mlflow.pytorch.log_model(model.cpu(), "model")
        print(f"\nbest val macro-F1 {best_f1:.4f} at epoch {best_ep}  ->  {ck_dir / 'best.pt'}")
        print(f"MLflow run id: {run.info.run_id}")


if __name__ == "__main__":
    main()
