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
    POST /classify   multipart field 'file' = image  ->  label, confidence, p_damaged, damaged, probs

    curl -F "file=@house.jpg" http://localhost:8000/classify

p_damaged = 1 - P(no-damage). `damaged` is p_damaged >= 0.5. This is the binary
view Rizky said matters most; `label` is the 4-way argmax and can disagree with
`damaged` when the probability is split between minor/major/destroyed.
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from inference import (CLASSES, DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI, load_checkpoint, load_registry,
                       pick_device, predict_probs, preprocess, summarise)

MAX_BYTES = 10 * 1024 * 1024  # reject uploads over 10 MB
state: dict = {}


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
    yield
    state.clear()


app = FastAPI(title="Property Due-Diligence: damage classifier", lifespan=lifespan)


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
        raise HTTPException(413, f"image larger than {MAX_BYTES // (1024 * 1024)} MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "not a readable image (use JPG or PNG)")
    batch = preprocess(img).unsqueeze(0)
    probs = predict_probs(state["model"], batch, state["device"])[0]
    return {**summarise(probs), "model": state["info"]}
