#!/usr/bin/env python3
"""
eda.py  --  Checkpoint 1 exploratory data analysis for the xBD dataset.
Property Due-Diligence & Investment Advisor

Replaces explore_xbd_dataset_v01.py + xbd_disaster_eda_v01.py: those two
scripts each re-implemented the same file discovery and label parsing to
produce two halves of the same report. This is that logic once, producing
every chart that ended up in the Checkpoint 1 deck.

Usage:
    python3 eda.py --data-root /path/to/xbd --output-dir eda_output

Outputs (in --output-dir):
    class_distribution.png         buildings per damage class, all disasters
    sample_grid.png                example building patches per damage class
    disaster_type_breakdown.png    scenes vs. buildings, by disaster TYPE
    disaster_breakdown.png         buildings per disaster EVENT, stacked by class
    disaster_type_sample_grid.png  example post-disaster scenes per type
    disaster_summary.csv           per-disaster counts + destroyed %
    eda_report.txt                 pairing/corruption checks, class counts, decision

Expects the standard xBD layout:
    <data-root>/.../images/<name>_post_disaster.png
    <data-root>/.../images/<name>_pre_disaster.png
    <data-root>/.../labels/<name>_post_disaster.json

Dependencies: pip install pillow matplotlib numpy tqdm
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]
COLORS = {"no-damage": "#2a78d6", "minor-damage": "#eda100", "major-damage": "#eb6834",
          "destroyed": "#e34948", "un-classified": "#898781"}
NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------------------
# discovery + parsing
# ---------------------------------------------------------------------------
def find_post_labels(root: Path):
    return sorted(p for p in root.rglob("*_post_disaster.json") if p.parent.name == "labels")


def image_for(label_path: Path, stem_suffix: str = "_post_disaster"):
    """The post-disaster image for a label, or its pre-disaster sibling with stem_suffix='_pre_disaster'."""
    stem = label_path.stem.replace("_post_disaster", stem_suffix)
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        for cand in (label_path.parent.parent / "images" / f"{stem}{ext}", label_path.parent / f"{stem}{ext}"):
            if cand.exists():
                return cand
    return None


def bbox_from_wkt(wkt: str):
    nums = [float(n) for n in NUM_RE.findall(wkt)]
    xs, ys = nums[0::2], nums[1::2]
    return (min(xs), min(ys), max(xs), max(ys)) if xs and ys else None


def split_of(label_path: Path):
    return next((p for p in label_path.parts if p.lower() in ("train", "test", "tier3", "hold")), "unknown")


def parse_label(label_path: Path, example_pool: int):
    """One post-disaster label -> scene stats + a capped pool of (image, bbox, class) examples."""
    try:
        data = json.loads(label_path.read_text())
    except Exception:
        return None
    post_img = image_for(label_path)
    if post_img is None:
        return None
    meta = data.get("metadata", {})
    disaster = meta.get("disaster") or label_path.stem.rsplit("_", 3)[0]
    dtype = meta.get("disaster_type") or "unknown"
    counts, examples = Counter(), []
    for f in data.get("features", {}).get("xy", []):
        props = f.get("properties", {})
        if props.get("feature_type", "building") != "building":
            continue
        cls = props.get("subtype", "un-classified")
        counts[cls] += 1
        if len(examples) < example_pool:
            bbox = bbox_from_wkt(f.get("wkt", ""))
            if bbox:
                examples.append((post_img, bbox, cls))
    return {"label": label_path, "post_img": post_img, "pre_img": image_for(label_path, "_pre_disaster"),
            "disaster": disaster, "disaster_type": dtype, "split": split_of(label_path),
            "counts": counts, "examples": examples}


def check_images(scenes):
    """Missing pre-disaster images, corrupted files, and distinct sizes seen."""
    missing_pre = [s for s in scenes if s["pre_img"] is None]
    corrupted, sizes = [], Counter()
    for s in tqdm(scenes, desc="checking images"):
        for img in (s["post_img"], s["pre_img"]):
            if img is None:
                continue
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:  # verify() invalidates the handle, reopen to read size
                    sizes[im.size] += 1
            except Exception as e:
                corrupted.append((img, str(e)))
    return missing_pre, corrupted, sizes


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def bar_labels(ax, bars, fmt="{:,}"):
    for b in bars:
        w = b.get_width()
        ax.text(w, b.get_y() + b.get_height() / 2, " " + fmt.format(int(w)), va="center", fontsize=8)


def plot_class_distribution(class_counts, out_dir: Path):
    classes = [c for c in CLASSES if class_counts.get(c, 0) > 0]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.bar(classes, [class_counts[c] for c in classes], color=[COLORS[c] for c in classes], width=0.6)
    for bar, c in zip(bars, classes):
        ax.annotate(f"{class_counts[c]:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    ax.set_title("xBD building damage class distribution", pad=14)
    ax.set_ylabel("Number of labeled buildings")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_sample_grid(examples_by_class, n, out_dir: Path, seed):
    rng = random.Random(seed)
    classes = [c for c in CLASSES if examples_by_class.get(c)]
    if not classes:
        return
    fig, axes = plt.subplots(len(classes), n, figsize=(2.2 * n, 2.4 * len(classes)), facecolor="#fcfcfb",
                              squeeze=False)
    for row, cls in enumerate(classes):
        picks = rng.sample(examples_by_class[cls], min(n, len(examples_by_class[cls])))
        for col in range(n):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(picks):
                continue
            img_path, (x0, y0, x1, y1) = picks[col]
            pad = 10
            with Image.open(img_path) as im:
                box = (max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad))
                ax.imshow(im.crop(box).convert("RGB"))
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor(COLORS[cls]); s.set_linewidth(3)
            if col == 0:
                ax.text(-0.15, 0.5, cls, transform=ax.transAxes, ha="right", va="center", fontsize=10,
                        color=COLORS[cls], fontweight="bold", rotation=90)
    fig.suptitle("Sample building patches by damage class")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_grid.png", dpi=150)
    plt.close(fig)


def plot_disaster_type_breakdown(scenes_by_type, bldg_by_type, out_dir: Path):
    types = sorted(scenes_by_type, key=lambda t: -sum(bldg_by_type[t].values()))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, 0.6 * len(types) + 1.5)))
    y = range(len(types))
    b1 = axes[0].barh(y, [scenes_by_type[t] for t in types], color="#3b6ea5")
    axes[0].set_yticks(list(y)); axes[0].set_yticklabels(types); axes[0].invert_yaxis()
    axes[0].set_title("Post-disaster scenes per disaster type")
    bar_labels(axes[0], b1)
    b2 = axes[1].barh(y, [sum(bldg_by_type[t].values()) for t in types], color="#7a5195")
    axes[1].set_yticks(list(y)); axes[1].set_yticklabels([]); axes[1].invert_yaxis()
    axes[1].set_title("Labeled buildings per disaster type")
    bar_labels(axes[1], b2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(out_dir / "disaster_type_breakdown.png", dpi=150)
    plt.close(fig)
    return types


def plot_disaster_breakdown(bldg_by_disaster, disaster_type_of, out_dir: Path):
    disasters = sorted(bldg_by_disaster, key=lambda d: -sum(bldg_by_disaster[d].values()))
    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(disasters) + 1.5)))
    left = [0] * len(disasters)
    for c in CLASSES:
        vals = [bldg_by_disaster[d][c] for d in disasters]
        ax.barh(range(len(disasters)), vals, left=left, color=COLORS[c], label=c)
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(range(len(disasters)))
    ax.set_yticklabels([f"{d} ({disaster_type_of[d]})" for d in disasters], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Labeled buildings per disaster, by damage class")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "disaster_breakdown.png", dpi=150)
    plt.close(fig)
    return disasters


def plot_disaster_type_sample_grid(types, scenes_of_type, n, out_dir: Path, seed):
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(types), n, figsize=(3 * n, 3 * len(types)), squeeze=False)
    for row, t in enumerate(types):
        pool = scenes_of_type[t]
        picks = rng.sample(pool, min(n, len(pool))) if pool else []
        for col in range(n):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(picks):
                continue
            s = picks[col]
            try:
                with Image.open(s["post_img"]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((512, 512))
                    ax.imshow(im)
                ax.set_title(s["label"].stem.replace("_post_disaster", ""), fontsize=7)
            except Exception:
                ax.text(0.5, 0.5, "image missing", ha="center", va="center", fontsize=8)
            if col == 0:
                ax.text(-0.04, 0.5, t, transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=11, weight="bold")
    fig.suptitle("Post-disaster samples by disaster type")
    fig.tight_layout()
    fig.savefig(out_dir / "disaster_type_sample_grid.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report + summary
# ---------------------------------------------------------------------------
def write_summary_csv(disasters, bldg_by_disaster, disaster_type_of, scenes_by_disaster, out_dir: Path):
    with open(out_dir / "disaster_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["disaster", "disaster_type", "scenes"] + CLASSES + ["total_buildings", "destroyed_pct"])
        for d in disasters:
            c, tot = bldg_by_disaster[d], sum(bldg_by_disaster[d].values())
            w.writerow([d, disaster_type_of[d], scenes_by_disaster[d]] + [c[k] for k in CLASSES]
                       + [tot, f"{100 * c['destroyed'] / tot:.2f}" if tot else "0"])


def write_report(out_dir, num_labels, missing_pre, corrupted, sizes, class_counts, class_counts_by_split):
    total = sum(class_counts.values())
    lines = ["xBD Checkpoint 1 -- EDA report", "=" * 40,
             f"Post-disaster label files  : {num_labels}",
             f"Missing pre-disaster image : {len(missing_pre)}",
             f"Corrupted/unreadable files : {len(corrupted)}",
             f"Distinct image sizes seen  : {len(sizes)}", "", "Class distribution (all buildings):"]
    for c in CLASSES:
        n = class_counts.get(c, 0)
        pct = f"  ({100 * n / total:5.1f}%)" if total else ""
        lines.append(f"  {c:15s} {n:8d}{pct}")
    lines += ["", "By split:"]
    for split, counts in class_counts_by_split.items():
        lines.append(f"  {split}:")
        lines += [f"    {c:15s} {counts[c]:8d}" for c in CLASSES if counts.get(c)]

    no_dmg = 100 * class_counts.get("no-damage", 0) / total if total else 0
    destroyed = 100 * class_counts.get("destroyed", 0) / total if total else 0
    lines += ["", "Insight -> preprocessing decision:",
              f"  'no-damage' is {no_dmg:.1f}% of labels vs {destroyed:.1f}% 'destroyed'."]
    if total and no_dmg > 50 and destroyed < 15:
        lines.append("  -> Confirmed imbalance: class-weighted loss + oversample major/destroyed in")
        lines.append("     Checkpoint 2, or the model defaults to 'no-damage' and still looks accurate.")

    if corrupted:
        lines += ["", "Corrupted files (first 20):"] + [f"  {p}: {e}" for p, e in corrupted[:20]]
    (out_dir / "eda_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--output-dir", default=Path("eda_output"), type=Path)
    ap.add_argument("--max-files", type=int, default=None, help="debug: only scan the first N label files")
    ap.add_argument("--samples-per-class", type=int, default=6, help="patches per class in sample_grid.png")
    ap.add_argument("--scenes-per-type", type=int, default=4, help="scenes per type in disaster_type_sample_grid.png")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)

    labels = find_post_labels(a.data_root)
    if not labels:
        raise SystemExit(f"No *_post_disaster.json under a 'labels' folder in {a.data_root}")
    if a.max_files:
        labels = labels[: a.max_files]

    print(f"Scanning {len(labels)} post-disaster label files ...")
    pool = max(a.samples_per_class, a.scenes_per_type) * 3
    scenes = [s for s in (parse_label(p, pool) for p in tqdm(labels)) if s]
    print(f"  usable scenes: {len(scenes)}")

    missing_pre, corrupted, sizes = check_images(scenes)

    class_counts, class_counts_by_split = Counter(), defaultdict(Counter)
    scenes_by_type, bldg_by_type = Counter(), defaultdict(Counter)
    disaster_type_of, scenes_of_type = {}, defaultdict(list)
    scenes_by_disaster, bldg_by_disaster = Counter(), defaultdict(Counter)
    examples_by_class = defaultdict(list)

    for s in scenes:
        class_counts.update(s["counts"])
        class_counts_by_split[s["split"]].update(s["counts"])
        disaster_type_of[s["disaster"]] = s["disaster_type"]
        scenes_by_type[s["disaster_type"]] += 1
        bldg_by_type[s["disaster_type"]].update(s["counts"])
        scenes_by_disaster[s["disaster"]] += 1
        bldg_by_disaster[s["disaster"]].update(s["counts"])
        if sum(s["counts"].values()) > 0:
            scenes_of_type[s["disaster_type"]].append(s)
        for img, bbox, cls in s["examples"]:
            if len(examples_by_class[cls]) < pool:
                examples_by_class[cls].append((img, bbox))

    plot_class_distribution(class_counts, a.output_dir)
    plot_sample_grid(examples_by_class, a.samples_per_class, a.output_dir, a.seed)
    types = plot_disaster_type_breakdown(scenes_by_type, bldg_by_type, a.output_dir)
    disasters = plot_disaster_breakdown(bldg_by_disaster, disaster_type_of, a.output_dir)
    plot_disaster_type_sample_grid(types, scenes_of_type, a.scenes_per_type, a.output_dir, a.seed)
    write_summary_csv(disasters, bldg_by_disaster, disaster_type_of, scenes_by_disaster, a.output_dir)
    write_report(a.output_dir, len(labels), missing_pre, corrupted, sizes, class_counts, class_counts_by_split)

    print(f"\nDone. Everything is in: {a.output_dir.resolve()}")


if __name__ == "__main__":
    main()
