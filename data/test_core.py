#!/usr/bin/env python3
"""
test_core.py -- checks for db.py and recommender.py. Run:  python3 test_core.py
Uses a throwaway database in a temp folder; your real storage/advisor.db is never touched.
"""
import tempfile
from pathlib import Path

import db
import recommender as rec
import seed_demo

tmp = Path(tempfile.mkdtemp())
conn = db.connect(tmp / "t.db")

# --- empty database: no crash, empty answer ---------------------------------
assert rec.recommend(conn, 2_000_000_000).empty
assert db.latest_rppi(conn, "Jakarta").empty

# --- seed is idempotent ---------------------------------------------------------
assert seed_demo.seed(conn) == 10
assert seed_demo.seed(conn) == 0
assert conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0] == 10

# --- unclassified properties are flagged, not silently scored as damaged ----------
out = rec.recommend(conn, 2_000_000_000, city="Jakarta", top_n=10)
assert not out.assessed.any()
assert (out.condition_part == rec.UNKNOWN_CONDITION).all()
assert "condition not assessed" in out.why.iloc[0]

# --- a damaged property ranks below an identical clean one ------------------------
a = db.add_property(conn, title="Twin clean", city="Jakarta", district="Duren Sawit", price_idr=1_000_000_000, source="test")
b = db.add_property(conn, title="Twin damaged", city="Jakarta", district="Duren Sawit", price_idr=1_000_000_000, source="test")
clean = {"label": "no-damage", "confidence": 0.95, "p_damaged": 0.05, "damaged": False, "model": {"arch": "resnet18"}}
bad = {"label": "destroyed", "confidence": 0.9, "p_damaged": 0.95, "damaged": True, "model": {"arch": "resnet18"}}
db.save_prediction(conn, a, clean, "a.jpg")
db.save_prediction(conn, b, bad, "b.jpg")
top = rec.recommend(conn, 1_500_000_000, city="Jakarta", district="Duren Sawit", top_n=20)
rank = {t: i for i, t in enumerate(top.title)}
assert rank["Twin clean"] < rank["Twin damaged"]
assert abs(top[top.title == "Twin clean"].condition_part.iloc[0] - 0.95) < 1e-9

# --- the latest prediction wins (re-classifying a property replaces its condition) --
db.save_prediction(conn, b, clean, "b2.jpg")
assert db.get_condition(conn, b)["label"] == "no-damage"
assert db.get_condition(conn, 9999) is None

# --- budget fit: inside = 1, 15% over = 0.5, 30%+ over = 0 ------------------------
import pandas as pd
s = rec.budget_score(pd.Series([900, 1000, 1150, 1300, 2000]), 1000)
assert [round(x, 3) for x in s] == [1.0, 1.0, 0.5, 0.0, 0.0], list(s)

# --- location: district match 1.0, city match 0.6, other 0.0, no city 1.0 ----------
df = pd.DataFrame({"city": ["Jakarta", "Jakarta", "Bandung"], "district": ["Duren Sawit", "Matraman", "Cibiru"]})
assert list(rec.location_score(df, "jakarta", "duren sawit")) == [1.0, 0.6, 0.0]
assert list(rec.location_score(df, "Jakarta", None)) == [0.6, 0.6, 0.0]
assert list(rec.location_score(df, None, None)) == [1.0, 1.0, 1.0]

# --- bad input -------------------------------------------------------------------
for bad_budget in (0, -5):
    try:
        rec.recommend(conn, bad_budget)
        raise SystemExit("expected ValueError")
    except ValueError:
        pass
try:
    db.add_property(conn, title="x", city="y", price_idr=0, source="test")
    raise SystemExit("expected price check to fail")
except Exception as e:
    assert "CHECK" in str(e) or "constraint" in str(e).lower()
try:
    db.add_property(conn, title="x", city="y", price_idr=5, source="test", colour="red")
    raise SystemExit("expected ValueError")
except ValueError:
    pass

# --- RPPI csv load + latest quarter lookup (rows below are TEST values, not BI data) --
csv = tmp / "rppi.csv"
csv.write_text("city,year,quarter,house_type,index_value,yoy_growth_pct\n"
               "Testville,2025,4,small,100.0,1.0\nTestville,2026,1,small,101.0,\nTestville,2026,1,large,102.0,2.0\n")
assert db.load_rppi_csv(conn, csv) == 3
assert db.load_rppi_csv(conn, csv) == 3  # reloading replaces, no duplicates
assert conn.execute("SELECT COUNT(*) FROM rppi").fetchone()[0] == 3
lt = db.latest_rppi(conn, "testville")
assert set(lt.house_type) == {"small", "large"} and (lt.year == 2026).all()
assert pd.isna(lt[lt.house_type == "small"].yoy_growth_pct.iloc[0])  # missing stays missing, not 0

# --- listings csv ---------------------------------------------------------------
lc = tmp / "listings.csv"
lc.write_text("title,city,district,price_idr,bedrooms\nCSV home,Depok,Beji,1200000000,3\n")
assert db.load_properties_csv(conn, lc) == 1
assert conn.execute("SELECT source FROM properties WHERE title='CSV home'").fetchone()[0] == "listing_csv"

print("all core checks passed")
