#!/usr/bin/env python3
"""
test_maps_imagery.py -- sanity-check the damage CNN on real, UNLABELED Google Maps
house crops, including normal houses, to check for false alarms.

There is no ground truth here, so this script computes NO accuracy. It runs the
crops through one or two models and builds a gallery you review by eye, with
predicted-damage crops first (those are the ones to check for false alarms).

    python3 test_maps_imagery.py --images maps_crops \
        --model checkpoints/C2_resnet50/best.pt checkpoints/C1_resnet18/best.pt

    # or straight from the registry
    python3 test_maps_imagery.py --images maps_crops --model models:/property-dd-damage-cnn@champion

Outputs (in --out-dir, default maps_review/):
    predictions.csv        one row per image, per-model label / confidence / p_damaged
    review_gallery.html    self-contained page (open in a browser); mark each crop OK /
                           false alarm / missed damage, then export your verdicts as CSV

How to crop (this matters more than any setting):
  * Use the SATELLITE top-down view, not Street View. The model was trained on
    top-down satellite crops of buildings (xBD); a street-level photo is a
    different kind of image and its output means nothing.
  * One building per crop, centred, with a little roof/yard context around it.
  * Zoom in until one house fills roughly half to two thirds of the frame.
  * Mix in plenty of obviously normal houses: the question is whether the model
    wrongly flags them as damaged.
  * Images are centre-cropped to a square and resized to 224x224.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from inference import (CLASSES, DEFAULT_TRACKING_URI, SIZE, load_any, pick_device, predict_probs, preprocess)

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
BADGE = {"no-damage": "#2a78d6", "minor-damage": "#eda100", "major-damage": "#eb6834", "destroyed": "#e34948"}


def thumb_b64(img: Image.Image, side: int = 280) -> str:
    t = ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), (side, side), Image.BICUBIC)
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def unique_names(specs: list[str], infos: list[dict]) -> list[str]:
    names, seen = [], {}
    for s, i in zip(specs, infos):
        n = i.get("alias") or i.get("name") or Path(s).stem
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return names


def build_html(rows: list[dict], names: list[str], summary: dict) -> str:
    cards = []
    for r in rows:
        badges = "".join(
            f'<div class="m"><b>{html.escape(n)}</b> '
            f'<span class="tag" style="background:{BADGE[r[n + "_label"]]}">{r[n + "_label"]}</span> '
            f'{r[n + "_confidence"]:.0%} conf &middot; p(damaged) {r[n + "_p_damaged"]:.0%}</div>'
            for n in names)
        dis = '<div class="dis">models disagree</div>' if r["models_disagree"] else ""
        cards.append(
            f'<div class="card" data-img="{html.escape(r["image"])}" data-v="">'
            f'<img src="data:image/jpeg;base64,{r["_thumb"]}" alt="">'
            f'<div class="fn">{html.escape(r["image"])}</div>{badges}{dis}'
            f'<div class="btns"><button data-v="ok">OK</button>'
            f'<button data-v="false_alarm">False alarm</button>'
            f'<button data-v="missed_damage">Missed damage</button></div></div>')
    per_model = "".join(
        f'<li><b>{html.escape(n)}</b>: {summary["damaged"][n]} of {summary["n"]} flagged damaged '
        f'({summary["damaged"][n] / summary["n"]:.0%})</li>' for n in names)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Google Maps review gallery</title>
<style>
:root{{--bg:#fafaf7;--fg:#0b0b0b;--card:#fff;--line:#e1e0d9;--mut:#5f5e57}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#f2f1ec;--card:#1b1b1a;--line:#333;--mut:#a5a49c}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}}
h1{{font-size:20px;margin:0 0 6px}}ul{{margin:6px 0 10px;padding-left:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:var(--card);border:2px solid var(--line);border-radius:10px;padding:10px}}
.card img{{width:100%;border-radius:6px;display:block}}
.card[data-v=ok]{{border-color:#1a7f37}}.card[data-v=false_alarm]{{border-color:#c62828}}
.card[data-v=missed_damage]{{border-color:#eda100}}
.fn{{color:var(--mut);font-size:12px;margin:6px 0;word-break:break-all}}
.tag{{color:#fff;padding:1px 7px;border-radius:9px;font-size:12px}}.m{{margin:3px 0}}
.dis{{color:#c62828;font-weight:600;margin-top:4px}}
.btns{{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}}
button{{border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;padding:5px 9px;cursor:pointer}}
button.primary{{background:var(--fg);color:var(--bg)}}
#tally{{margin:8px 0 14px}}
</style></head><body>
<h1>Google Maps review gallery</h1>
<div>{summary["n"]} unlabeled crops, predicted-damage first. No ground truth: you are the judge.</div>
<ul>{per_model}<li>Models disagree (damaged vs normal) on {summary["disagree"]} crops</li></ul>
<div id="tally"></div>
<button class="primary" id="exp">Export my verdicts (CSV)</button>
<div class="grid" id="grid">{"".join(cards)}</div>
<script>
const cards=[...document.querySelectorAll('.card')];
function tally(){{
  const c={{ok:0,false_alarm:0,missed_damage:0}};
  cards.forEach(x=>{{if(x.dataset.v)c[x.dataset.v]++}});
  document.getElementById('tally').textContent=
   'Reviewed '+(c.ok+c.false_alarm+c.missed_damage)+' of '+cards.length+' | OK '+c.ok+' | false alarms '+c.false_alarm+' | missed damage '+c.missed_damage;
}}
cards.forEach(x=>x.querySelectorAll('.btns button').forEach(b=>b.onclick=()=>{{
  x.dataset.v=(x.dataset.v===b.dataset.v)?'':b.dataset.v;tally();}}));
document.getElementById('exp').onclick=()=>{{
  const lines=['image,verdict'].concat(cards.map(x=>'"'+x.dataset.img.replace(/"/g,'""')+'",'+(x.dataset.v||'unreviewed')));
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')],{{type:'text/csv'}}));
  a.download='my_verdicts.csv';a.click();}};
tally();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder of cropped house images")
    ap.add_argument("--model", required=True, nargs="+", metavar="PT_OR_URI",
                    help="one or two: path to best.pt, or models:/<name>@<alias>")
    ap.add_argument("--out-dir", type=Path, default=Path("maps_review"))
    ap.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if len(a.model) > 2:
        raise SystemExit("give at most two models (more makes the gallery unreadable)")

    paths = sorted(p for p in a.images.rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"no images found in {a.images} (looked for {sorted(EXTS)})")
    device = pick_device(a.device)
    print(f"{len(paths)} images, device {device}")

    loaded = [load_any(s, a.tracking_uri, device) for s in a.model]
    names = unique_names(a.model, [i for _, i in loaded])
    print("models:", ", ".join(f"{n} ({i['source']})" for n, (_, i) in zip(names, loaded)))

    rows, tensors, keep = [], [], []
    for p in paths:  # read once, skip unreadable files with a warning
        try:
            with Image.open(p) as im:
                im.load()
                tensors.append(preprocess(im))
                keep.append({"image": str(p.relative_to(a.images)), "_thumb": thumb_b64(im)})
        except Exception as e:
            print(f"  skipped {p.name}: {e}")
    import torch
    stack = torch.stack(tensors)
    probs = {n: [] for n in names}
    for i in range(0, len(stack), a.batch_size):
        for n, (m, _) in zip(names, loaded):
            probs[n].append(predict_probs(m, stack[i:i + a.batch_size], device))
    probs = {n: __import__("numpy").concatenate(v) for n, v in probs.items()}

    for j, row in enumerate(keep):
        flags = []
        for n in names:
            p = probs[n][j]
            k = int(p.argmax())
            row.update({f"{n}_label": CLASSES[k], f"{n}_confidence": round(float(p[k]), 4),
                        f"{n}_p_damaged": round(float(1 - p[0]), 4),
                        **{f"{n}_p_{c}": round(float(x), 4) for c, x in zip(CLASSES, p)}})
            flags.append(1 - p[0] >= 0.5)
        row["max_p_damaged"] = max(row[f"{n}_p_damaged"] for n in names)
        row["models_disagree"] = len(set(flags)) > 1
        rows.append(row)
    rows.sort(key=lambda r: r["max_p_damaged"], reverse=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_thumb"} for r in rows])
    df.to_csv(a.out_dir / "predictions.csv", index=False)
    summary = {"n": len(rows), "disagree": int(df["models_disagree"].sum()),
               "damaged": {n: int((df[f"{n}_p_damaged"] >= 0.5).sum()) for n in names}}
    (a.out_dir / "review_gallery.html").write_text(build_html(rows, names, summary), encoding="utf-8")
    print(json.dumps(summary))
    print(f"wrote {a.out_dir}/predictions.csv and {a.out_dir}/review_gallery.html  (open the HTML in a browser)")


if __name__ == "__main__":
    main()
