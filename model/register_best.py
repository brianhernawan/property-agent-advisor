#!/usr/bin/env python3
"""
register_best.py -- pick the run with the highest best_val_macro_f1 in the MLflow
experiment and register its model in the MLflow Model Registry.

    python3 register_best.py --experiment property-dd-cnn --name property-dd-damage-cnn
"""
import argparse

import mlflow
from mlflow.tracking import MlflowClient

ap = argparse.ArgumentParser()
ap.add_argument("--experiment", default="property-dd-cnn")
ap.add_argument("--name", default="property-dd-damage-cnn")
ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
a = ap.parse_args()

mlflow.set_tracking_uri(a.tracking_uri)
runs = mlflow.search_runs(experiment_names=[a.experiment], order_by=["metrics.best_val_macro_f1 DESC"])
runs = runs[runs["metrics.best_val_macro_f1"].notna()]
if runs.empty:
    raise SystemExit("no finished runs with best_val_macro_f1 found")

print(runs[["tags.mlflow.runName", "params.model", "params.unfreeze", "params.lr", "params.batch_size",
            "params.balance", "metrics.best_val_macro_f1"]].head(10).to_string(index=False))
best = runs.iloc[0]
rid = best["run_id"]
mv = mlflow.register_model(f"runs:/{rid}/model", a.name)
MlflowClient().set_registered_model_alias(a.name, "champion", mv.version)
print(f"\nregistered {a.name} v{mv.version} (alias 'champion') from run {rid}  "
      f"[{best['tags.mlflow.runName']}, val macro-F1 {best['metrics.best_val_macro_f1']:.4f}]")
