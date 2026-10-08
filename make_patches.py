#!/usr/bin/env python3
"""
make_patches.py  --  Checkpoint 2, step 0
Crop every labelled xBD building out of its POST-disaster image, resize to
size x size, and write a manifest with a fixed scene-level 70/15/15 split.

Usage:
    python3 make_patches.py --data-root /path/to/xbd --out-dir patches

Outputs (inside --out-dir):
    <split>/<class>/<scene>_<idx>.jpg     the patches
    manifest.csv                          one row per patch (see COLUMNS)
    split_report.txt                      class x split and disaster x split tables

Design decisions (put these on the slide):
  * 4 classes only. 'un-classified' (1.8%) is dropped; xBD has no 'no building' label.
  * Split is by SCENE, never by patch, so near-duplicate neighbours can't leak
    across train/val/test. Scenes are assigned greedily so every split gets a
    similar class mix (rare classes are placed first).
  * Square crop centred on the polygon bbox, side = max(w,h) * (1 + 2*pad),
    never smaller than --min-ctx px, so tiny buildings keep some context and
    aspect ratio is not distorted on resize.
  * The split is written once. Re-running with the same --seed gives the same split.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
SPLITS = ["train", "val", "test"]
COLUMNS = ["path", "label", "label_idx", "split", "scene", "disaster", "uid", "bbox_w", "bbox_h"]
NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


# ----------------------------------------------------------------------------
# discovery + parsing
# ----------------------------------------------------------------------------
def find_post_labels(root: Path):
    return sorted(p for p in root.rglob("*_post_disaster.json") if p.parent.name == "labels")


def image_for(label_path: Path):
    stem = label_path.stem
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        for cand in (label_path.parent.parent / "images" / f"{stem}{ext}",
                     label_path.with_suffix(ext)):
            if cand.exists():
                return cand
    return None


def bbox_from_wkt(wkt: str):
    nums = [float(n) for n in NUM_RE.findall(wkt)]
    xs, ys = nums[0::2], nums[1::2]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def parse_scene(label_path: Path):
    try:
        d = json.loads(label_path.read_text())
    except Exception:
        return None
    img = image_for(label_path)
    if img is None:
        return None
    stem = label_path.stem
    scene = stem.replace("_post_disaster", "")
    disaster = d.get("metadata", {}).get("disaster") or stem.rsplit("_", 3)[0]
    blds, counts = [], Counter()
    for i, f in enumerate(d.get("features", {}).get("xy", [])):
        props = f.get("properties", {})
        if props.get("feature_type", "building") != "building":
            continue
        cls = props.get("subtype", "un-classified")
        if cls not in CLASS_IDX:
            continue
        bb = bbox_from_wkt(f.get("wkt", ""))
        if bb is None:
            continue
        blds.append((props.get("uid") or f"b{i}", cls, bb))
        counts[cls] += 1
    if not blds:
        return None
    return {"scene": scene, "disaster": disaster, "image": str(img), "buildings": blds, "counts": counts}


# ----------------------------------------------------------------------------
# scene-level stratified split
# ----------------------------------------------------------------------------
def assign_splits(scenes, fracs, seed):
    rng = random.Random(seed)
    total = Counter()
    for s in scenes:
        total.update(s["counts"])
    cur = {sp: Counter() for sp in SPLITS}
    order = scenes[:]
    rng.shuffle(order)
    # rare classes first so they get spread evenly; stable sort keeps random tie-break
    order.sort(key=lambda s: -(2 * s["counts"]["destroyed"] + s["counts"]["major-damage"]
                              + s["counts"]["minor-damage"]))
    for s in order:
        best, best_score = None, None
        for sp in SPLITS:
            score = 0.0
            for c in CLASSES:
                if total[c] == 0 or s["counts"][c] == 0:
                    continue
                target = fracs[sp] * total[c]
                score += (s["counts"][c] / total[c]) * (1 - cur[sp][c] / target)
            score += rng.random() * 1e-9
            if best_score is None or score > best_score:
                best, best_score = sp, score
        s["split"] = best
        cur[best].update(s["counts"])
    return scenes


# ----------------------------------------------------------------------------
# cropping worker
# ----------------------------------------------------------------------------
def crop_scene(job):
    scene, out_dir, size, pad, min_ctx, min_side, quality, overwrite = job
    out_dir = Path(out_dir)
    rows, skipped = [], 0
    try:
        im = Image.open(scene["image"]).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return rows, len(scene["buildings"]), f"{scene['image']}: {e}"
    for i, (uid, cls, (x0, y0, x1, y1)) in enumerate(scene["buildings"]):
        w, h = x1 - x0, y1 - y0
        if max(w, h) < min_side:
            skipped += 1
            continue
        side = max(max(w, h) * (1 + 2 * pad), min_ctx)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = (int(round(cx - side / 2)), int(round(cy - side / 2)),
               int(round(cx + side / 2)), int(round(cy + side / 2)))
        rel = Path(scene["split"]) / cls / f"{scene['scene']}_{i:04d}.jpg"
        dst = out_dir / rel
        if overwrite or not dst.exists():
            im.crop(box).resize((size, size), Image.BICUBIC).save(dst, quality=quality)
        rows.append([str(rel), cls, CLASS_IDX[cls], scene["split"], scene["scene"],
                     scene["disaster"], uid, round(w, 1), round(h, 1)])
    return rows, skipped, None


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("patches"), type=Path)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.15, help="context padding as fraction of bbox size per side")
    ap.add_argument("--min-ctx", type=int, default=48, help="minimum crop side in source pixels")
    ap.add_argument("--min-side", type=int, default=4, help="skip buildings whose larger bbox side is below this (px)")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-scenes", type=int, default=None, help="debug: only use the first N scenes")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    labels = find_post_labels(a.data_root)
    if not labels:
        sys.exit(f"No *_post_disaster.json under a 'labels' folder in {a.data_root}")
    if a.max_scenes:
        labels = labels[: a.max_scenes]

    print(f"[1/3] parsing {len(labels)} label files ...")
    scenes = [s for s in (parse_scene(p) for p in tqdm(labels)) if s]
    print(f"      usable scenes: {len(scenes)}")

    fracs = {"train": 1 - a.val_frac - a.test_frac, "val": a.val_frac, "test": a.test_frac}
    scenes = assign_splits(scenes, fracs, a.seed)

    for sp in SPLITS:
        for c in CLASSES:
            (a.out_dir / sp / c).mkdir(parents=True, exist_ok=True)

    print("[2/3] cropping patches ...")
    jobs = [(s, str(a.out_dir), a.size, a.pad, a.min_ctx, a.min_side, a.quality, a.overwrite) for s in scenes]
    rows, skipped, errors = [], 0, []
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            results = list(tqdm(ex.map(crop_scene, jobs, chunksize=4), total=len(jobs)))
    else:
        results = [crop_scene(j) for j in tqdm(jobs)]
    for r, sk, err in results:
        rows.extend(r)
        skipped += sk
        if err:
            errors.append(err)

    print("[3/3] writing manifest + report ...")
    rows.sort(key=lambda r: (r[3], r[4], r[6]))
    with open(a.out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)

    by = defaultdict(Counter)
    by_dis = defaultdict(Counter)
    for r in rows:
        by[r[3]][r[1]] += 1
        by_dis[r[5]][r[3]] += 1
    lines = [f"patches written: {len(rows):,}   skipped (tiny bbox): {skipped:,}   scene errors: {len(errors)}", ""]
    lines.append(f"{'class':<14}" + "".join(f"{sp:>12}" for sp in SPLITS) + f"{'total':>12}")
    for c in CLASSES:
        n = [by[sp][c] for sp in SPLITS]
        lines.append(f"{c:<14}" + "".join(f"{x:>12,}" for x in n) + f"{sum(n):>12,}")
    lines.append("")
    lines.append("class share within each split (%):")
    for sp in SPLITS:
        tot = sum(by[sp].values()) or 1
        lines.append(f"  {sp:<6}" + "".join(f"{c}={100 * by[sp][c] / tot:5.1f}  " for c in CLASSES))
    lines.append("")
    lines.append(f"{'disaster':<22}" + "".join(f"{sp:>10}" for sp in SPLITS))
    for d in sorted(by_dis):
        lines.append(f"{d:<22}" + "".join(f"{by_dis[d][sp]:>10,}" for sp in SPLITS))
    if errors:
        lines += ["", "scene errors (first 10):"] + errors[:10]
    (a.out_dir / "split_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nmanifest: {a.out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
