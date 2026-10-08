#!/usr/bin/env python3
"""
serve_api.py -- FastAPI service that classifies one building photo.

    uvicorn serve_api:app --port 8000

Which model it loads (set before starting):
    MODEL_URI     default models:/property-dd-damage-cnn@serving  (MLflow Model Registry)
    TRACKING_URI  default sqlite:///mlflow.db
    CHECKPOINT    optional path to a best.pt; if set it wins over MODEL_URI
    DEVICE        auto | cpu | cuda | mps

Endpoints:
    GET  /health     which model is loaded (503 until it is ready)
    GET  /metrics    Prometheus metrics (HTTP traffic + predictions, latency, p_damaged)
    POST /classify   multipart field 'file' = image  ->  label, confidence, p_damaged, damaged, probs

    curl -F "file=@house.jpg" http://localhost:8000/classify

p_damaged = 1 - P(no-damage). `damaged` is p_damaged >= 0.5. This is the binary
view Rizky said matters most; `label` is the 4-way argmax and can disagree with
`damaged` when the probability is split between minor/major/destroyed.
"""
from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from inference import (CLASSES, DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI, load_checkpoint, load_registry,
                       pick_device, predict_probs, preprocess, summarise)

MAX_BYTES = 10 * 1024 * 1024  # reject uploads over 10 MB
state: dict = {}

# Model metrics for Prometheus/Grafana (HTTP request metrics come from the Instrumentator below)
PREDICTIONS = Counter("dd_predictions_total", "Classified images", ["label", "damaged"])
REJECTED = Counter("dd_rejected_total", "Uploads refused before inference", ["reason"])
INFER_SECONDS = Histogram("dd_inference_seconds", "Preprocess + model forward time",
                          buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
P_DAMAGED = Histogram("dd_p_damaged", "Distribution of p_damaged = 1 - P(no-damage)",
                      buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
MODEL_LOADED = Gauge("dd_model_loaded", "1 when the classifier is loaded and serving")


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = pick_device(os.getenv("DEVICE", "auto"))
    ckpt = os.getenv("CHECKPOINT")
    if ckpt:
        model, info = load_checkpoint(ckpt, device)
    else:
        uri = os.getenv("MODEL_URI", f"models:/{DEFAULT_MODEL_NAME}@serving")
        model, info = load_registry(uri, os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI), device)
    state.update(model=model, device=device, info={**info, "device": str(device)})
    print(f"model loaded: {state['info']}")
    MODEL_LOADED.set(1)
    yield
    MODEL_LOADED.set(0)
    state.clear()


app = FastAPI(title="Property Due-Diligence: damage classifier", lifespan=lifespan)
Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(app, include_in_schema=False)


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    p_damaged: float
    damaged: bool
    probs: dict[str, float]
    model: dict


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(503, "model not loaded yet")
    return {"status": "ok", "classes": CLASSES, "model": state["info"]}


@app.post("/classify", response_model=ClassifyResponse)
def classify(file: UploadFile = File(...)):
    if "model" not in state:
        raise HTTPException(503, "model not loaded yet")
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        REJECTED.labels("too_large").inc()
        raise HTTPException(413, f"image larger than {MAX_BYTES // (1024 * 1024)} MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        REJECTED.labels("not_an_image").inc()
        raise HTTPException(400, "not a readable image (use JPG or PNG)")
    t0 = time.perf_counter()
    batch = preprocess(img).unsqueeze(0)
    probs = predict_probs(state["model"], batch, state["device"])[0]
    INFER_SECONDS.observe(time.perf_counter() - t0)
    out = summarise(probs)
    PREDICTIONS.labels(out["label"], str(out["damaged"]).lower()).inc()
    P_DAMAGED.observe(out["p_damaged"])
    return {**out, "model": state["info"]}
