"""client.py -- tiny helper the Streamlit app uses to call serve_api.py."""
from __future__ import annotations

import requests


def classify_image(api_url: str, data: bytes, filename: str) -> dict:
    """POST one image to serve_api.py and return its JSON. Raises RuntimeError with a readable message."""
    try:
        r = requests.post(f"{api_url}/classify", files={"file": (filename, data)}, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"cannot reach the classifier API at {api_url} ({e.__class__.__name__})")
    if r.status_code != 200:
        raise RuntimeError(f"classifier API said {r.status_code}: {r.json().get('detail', r.text)}")
    return r.json()
