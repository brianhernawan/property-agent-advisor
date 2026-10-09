#!/usr/bin/env python3
"""
floodrisk.py -- flood-hazard lookup for one coordinate from BNPB InaRISK (public ArcGIS ImageServer).

    python3 floodrisk.py --lat -6.23 --lon 106.91

Returns the RAW hazard index the service gives for that point (a number between 0 and 1) and nothing
else. No class names (low / medium / high) are invented here: the official class thresholds for this
layer were not verified, so the app shows the number and says where it comes from.

    status "ok"       value is the index at that point
    status "no_data"  the point is outside the mapped hazard raster. This is NOT proof of zero risk.
    status "error"    the service could not be reached or returned something unexpected

Source layer: https://gis.bnpb.go.id/server/rest/services/inarisk/layer_bahaya_banjir/ImageServer
"""
from __future__ import annotations

import argparse
import json

import requests

URL = "https://gis.bnpb.go.id/server/rest/services/inarisk/layer_bahaya_banjir/ImageServer/identify"
SOURCE = "BNPB InaRISK flood hazard layer"
TIMEOUT_S = 15

# Successful answers (ok / no_data) are cached for the life of the process. Errors are NOT cached,
# so a temporary outage of the BNPB server does not stick: the next lookup simply tries again.
_cache: dict = {}


def _fetch(lat: float, lon: float) -> dict:
    """One HTTP call. Separate function so tests can replace it."""
    params = {
        "geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryPoint",
        "returnGeometry": "false",
        "f": "json",
    }
    r = requests.get(URL, params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def flood_index(lat: float, lon: float) -> dict:
    """Look up the InaRISK flood hazard index at (lat, lon), WGS84 degrees. Never raises.
    ok and no_data results are cached; errors are retried on the next call."""
    key = (round(lat, 6), round(lon, 6))
    if key in _cache:
        return _cache[key]
    result = _lookup(lat, lon)
    if result["status"] in ("ok", "no_data"):
        _cache[key] = result
    return result


def clear_cache() -> None:
    """Forget every cached answer (used by tests)."""
    _cache.clear()


def _lookup(lat: float, lon: float) -> dict:
    """One uncached lookup, turned into a status dict."""
    if not (-12.0 <= lat <= 7.0 and 94.0 <= lon <= 142.0):
        return {"status": "error", "source": SOURCE, "note": "coordinates are outside Indonesia"}
    try:
        data = _fetch(lat, lon)
    except Exception as e:  # network, timeout, bad JSON
        return {"status": "error", "source": SOURCE, "note": f"service unreachable: {type(e).__name__}"}
    if "error" in data:
        return {"status": "error", "source": SOURCE, "note": f"service error: {data['error'].get('message', 'unknown')}"}
    raw = data.get("value")
    if raw is None:
        return {"status": "error", "source": SOURCE, "note": "no value field in the response"}
    if str(raw).strip().lower() == "nodata":
        return {"status": "no_data", "source": SOURCE,
                "note": "point is outside the mapped flood-hazard area; this is not proof of zero risk"}
    try:
        value = round(float(raw), 3)
    except ValueError:
        return {"status": "error", "source": SOURCE, "note": f"unexpected value: {raw!r}"}
    return {"status": "ok", "value": value, "scale": "0 to 1 (raw index from the service)", "source": SOURCE}


def short(result: dict) -> str:
    """One cell of text for a table: the number, 'no data' or 'unavailable'."""
    return {"ok": f"{result.get('value', 0):.2f}", "no_data": "no data"}.get(result["status"], "unavailable")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    a = ap.parse_args()
    print(json.dumps(flood_index(a.lat, a.lon), indent=2))
