#!/usr/bin/env python3
"""
inference.py -- shared model loading + preprocessing for serve_api.py and
test_maps_imagery.py.

Why a separate file: train.py pulls in matplotlib, scikit-learn and the whole
training stack. Serving only needs torch, torchvision, pillow and mlflow, so the
Docker image stays smaller when the API imports this file instead of train.py.

Everything here mirrors train.py (same classes, same ImageNet normalisation, same
224x224 input, same classifier head), so a checkpoint saved by train.py loads
here with no changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms as T

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
K = len(CLASSES)
SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # same as train.py
DEFAULT_MODEL_NAME = "property-dd-damage-cnn"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

_to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def pick_device(arg: str = "auto") -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess(img: Image.Image) -> torch.Tensor:
    """Any photo -> one normalised 3x224x224 tensor.

    Training patches were square crops around one building, resized to 224x224.
    So we centre-crop to a square (no stretching) and resize. Crop the house
    tightly BEFORE uploading; the model sees only the centre square.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = ImageOps.fit(img, (SIZE, SIZE), Image.BICUBIC)
    return _to_tensor(img)


def build_arch(name: str) -> nn.Module:
    """Same architecture and classifier head as train.build_model, no pretrained download."""
    m = torchvision.models.get_model(name, weights=None)
    if name.startswith("resnet"):
        m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, K))
    elif name.startswith("efficientnet"):
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, K))
    else:
        raise ValueError(f"unsupported model {name}")
    return m


def load_checkpoint(path: str | Path, device: torch.device):
    """Load a best.pt written by train.py. Only open files you trained yourself:
    torch.load with weights_only=False can run code stored inside the file."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_arch(ck["arch"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    info = {"source": str(path), "arch": ck["arch"], "name": Path(path).parent.name,
            "val_macro_f1": ck.get("val_macro_f1")}
    return model, info


def load_registry(uri: str, tracking_uri: str, device: torch.device):
    """Load a model from the MLflow Model Registry, e.g. models:/property-dd-damage-cnn@serving."""
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    info = {"source": uri, "name": uri.split("@")[-1] if "@" in uri else uri}
    if uri.startswith("models:/") and "@" in uri:  # resolve the alias to a version, for /health
        name, alias = uri[len("models:/"):].split("@", 1)
        mv = MlflowClient().get_model_version_by_alias(name, alias)
        run = MlflowClient().get_run(mv.run_id)
        info.update({"registered_model": name, "alias": alias, "version": mv.version, "run_id": mv.run_id,
                     "run_name": run.data.tags.get("mlflow.runName"), "arch": run.data.params.get("model")})
    model = mlflow.pytorch.load_model(uri, map_location="cpu")
    model.to(device).eval()
    info.setdefault("arch", type(model).__name__)
    return model, info


def load_any(spec: str, tracking_uri: str, device: torch.device):
    """A models:/... URI goes to the registry; anything else is treated as a .pt path."""
    if spec.startswith("models:/"):
        return load_registry(spec, tracking_uri, device)
    return load_checkpoint(spec, device)


@torch.inference_mode()
def predict_probs(model: nn.Module, batch: torch.Tensor, device: torch.device) -> np.ndarray:
    """batch: N x 3 x 224 x 224 -> N x 4 softmax probabilities."""
    return torch.softmax(model(batch.to(device)), 1).cpu().numpy()


def summarise(probs: np.ndarray) -> dict:
    """One row of probabilities -> label, confidence, and the binary 'damaged' view.

    p_damaged = 1 - P(no-damage). This is the number the recommender should use:
    Rizky's priority is damaged vs normal, not the 4-way label.
    """
    i = int(np.argmax(probs))
    p_damaged = float(1.0 - probs[0])
    return {"label": CLASSES[i], "confidence": round(float(probs[i]), 4),
            "p_damaged": round(p_damaged, 4), "damaged": p_damaged >= 0.5,
            "probs": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}}
