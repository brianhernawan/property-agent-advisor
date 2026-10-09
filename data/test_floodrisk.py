#!/usr/bin/env python3
"""test_floodrisk.py -- offline checks for floodrisk.py (the HTTP call is replaced). Run: python3 test_floodrisk.py"""
import floodrisk as fr


def run(payload=None, exc=None, lat=-6.23, lon=106.91):
    def fake(la, lo):
        if exc:
            raise exc
        return payload
    fr._fetch = fake
    fr.clear_cache()
    return fr.flood_index(lat, lon)


r = run({"value": "0.703704"})
assert r["status"] == "ok" and r["value"] == 0.704 and "class" not in r          # raw number, no invented class
assert run({"value": "NoData"})["status"] == "no_data"                           # outside raster is not "zero risk"
assert "not proof" in run({"value": "NoData"})["note"]
assert run({"error": {"message": "bad"}})["status"] == "error"
assert run({})["status"] == "error"
assert run({"value": "abc"})["status"] == "error"
assert run(exc=TimeoutError())["status"] == "error"                              # never raises
assert run({"value": "0.5"}, lat=40.0, lon=-74.0)["status"] == "error"           # not Indonesia
assert fr.short({"status": "ok", "value": 0.7037}) == "0.70"
assert fr.short({"status": "no_data"}) == "no data" and fr.short({"status": "error"}) == "unavailable"

calls = []
fr._fetch = lambda la, lo: calls.append(1) or {"value": "0.2"}
fr.clear_cache()
fr.flood_index(-6.2, 106.8); fr.flood_index(-6.2, 106.8)
assert len(calls) == 1                                                            # repeated lookups are cached

# an error is NOT cached: once the service is back, the next call gets the real value
state = {"down": True}
def flaky(la, lo):
    if state["down"]:
        raise TimeoutError()
    return {"value": "0.703704"}
fr._fetch = flaky
fr.clear_cache()
assert fr.flood_index(-6.3, 106.9)["status"] == "error"
state["down"] = False
assert fr.flood_index(-6.3, 106.9)["value"] == 0.704
print("all floodrisk checks passed")
