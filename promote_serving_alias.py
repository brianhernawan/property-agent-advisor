#!/usr/bin/env python3
"""
promote_serving_alias.py -- put the model you want to SERVE behind the `serving`
alias in the MLflow Model Registry, without touching the `champion` alias.

Background: register_best.py registered the highest validation macro-F1 run
(C2_resnet50) as version 1 and gave it the alias `champion`. The Checkpoint 2
recommendation is to serve ResNet-18 (C1_resnet18): val macro-F1 0.6925 vs 0.6934,
about a third of the inference cost. This script registers that run as a new
version and points `serving` at it. serve_api.py always loads @serving, so
swapping the served model later is one command, not a code change.

    python3 promote_serving_alias.py --list                 # see runs, versions, aliases
    python3 promote_serving_alias.py --dry-run              # show what would happen
    python3 promote_serving_alias.py                        # C1_resnet18 -> alias 'serving'
    python3 promote_serving_alias.py --run-name C2_resnet50 # serve the champion instead

Safe to re-run: if the run is already a registered version it is reused, not duplicated.
"""
import argparse

import mlflow
from mlflow.tracking import MlflowClient

ap = argparse.ArgumentParser()
ap.add_argument("--experiment", default="property-dd-cnn")
ap.add_argument("--name", default="property-dd-damage-cnn", help="registered model name")
ap.add_argument("--run-name", default="C1_resnet18", help="MLflow run to serve")
ap.add_argument("--alias", default="serving")
ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
ap.add_argument("--list", action="store_true", help="print runs, versions and aliases, then exit")
ap.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")
a = ap.parse_args()

mlflow.set_tracking_uri(a.tracking_uri)
client = MlflowClient()


def show_registry():
    try:
        versions = client.search_model_versions(f"name='{a.name}'")
        rm = client.get_registered_model(a.name)
    except Exception as e:  # model not registered yet
        print(f"registered model '{a.name}' not found ({e}). Run register_best.py first.")
        return
    aliases = {}
    for al, ver in (rm.aliases or {}).items():
        aliases.setdefault(str(ver), []).append(al)
    print(f"\nRegistry: {a.name}")
    for v in sorted(versions, key=lambda v: int(v.version)):
        r = client.get_run(v.run_id)
        print(f"  v{v.version}  run {r.data.tags.get('mlflow.runName', v.run_id)}  "
              f"aliases={aliases.get(str(v.version), [])}")


runs = mlflow.search_runs(experiment_names=[a.experiment], order_by=["metrics.best_val_macro_f1 DESC"])
runs = runs[runs["metrics.best_val_macro_f1"].notna()]
if a.list:
    print(runs[["tags.mlflow.runName", "params.model", "metrics.best_val_macro_f1"]].to_string(index=False))
    show_registry()
    raise SystemExit

match = runs[runs["tags.mlflow.runName"] == a.run_name]
if match.empty:
    names = ", ".join(runs["tags.mlflow.runName"].tolist())
    raise SystemExit(f"no finished run named '{a.run_name}' in '{a.experiment}'. Available: {names}")
row = match.iloc[0]
rid = row["run_id"]
print(f"run: {a.run_name}  ({row['params.model']}, val macro-F1 {row['metrics.best_val_macro_f1']:.4f})  id {rid}")

existing = [v for v in client.search_model_versions(f"name='{a.name}'") if v.run_id == rid]
if existing:
    version = existing[0].version
    print(f"already registered as {a.name} v{version}, reusing it")
elif a.dry_run:
    print(f"[dry-run] would register runs:/{rid}/model as a new version of {a.name}")
    version = None
else:
    mv = mlflow.register_model(f"runs:/{rid}/model", a.name)
    version = mv.version
    print(f"registered {a.name} v{version}")

if a.dry_run:
    print(f"[dry-run] would set alias '{a.alias}' -> v{version or 'NEW'}; 'champion' is not touched")
else:
    client.set_registered_model_alias(a.name, a.alias, version)
    check = client.get_model_version_by_alias(a.name, a.alias)
    assert check.run_id == rid, "alias check failed: alias does not point at the expected run"
    print(f"alias '{a.alias}' -> v{check.version}  (verified)")
show_registry()
print(f"\nServe it with:\n  MODEL_URI=models:/{a.name}@{a.alias} uvicorn serve_api:app --port 8000")
