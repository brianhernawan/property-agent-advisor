#!/usr/bin/env python3
"""
test_content.py -- checks for the TF-IDF content-based recommender (content.py) and the text match
inside recommender.py. Throwaway database; storage/advisor.db is never touched. Run:  python3 test_content.py
"""
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

import content
import db
import recommender as rec
import seed_demo

tmp = Path(tempfile.mkdtemp())
conn = db.connect(tmp / "t.db")

# --- empty database: no crash -------------------------------------------------
assert content.similar(conn, 1).empty

seed_demo.seed(conn)
ids = {r["title"]: r["id"] for r in conn.execute("SELECT id, title FROM properties")}
assert conn.execute("SELECT COUNT(*) FROM properties WHERE description IS NULL").fetchone()[0] == 0

# --- similar(): never returns itself, sorted, parts in [0, 1] -------------------
sim = content.similar(conn, ids["Demo townhouse C"], 9)
assert ids["Demo townhouse C"] not in set(sim.id)
assert list(sim.similarity) == sorted(sim.similarity, reverse=True)
for c in ("similarity", "text_sim", "numeric_sim"):
    assert sim[c].between(-1e-9, 1 + 1e-9).all(), c
# the other gated, low-maintenance townhouse is the closest match to townhouse C
assert sim.title.iloc[0] == "Demo townhouse G", sim[["title", "similarity", "shared_terms"]]
assert "townhouse" in sim.shared_terms.iloc[0]
# the shophouse (commercial, busy road, no bedrooms) is not near the top for a family house
fam = content.similar(conn, ids["Demo house A"], 9)
assert fam.title.iloc[-1] == "Demo shophouse J" or fam[fam.title == "Demo shophouse J"].index[0] >= 6
assert content.similar(conn, 99999).empty

# --- query_match(): best = 1, unrelated words = 0 ---------------------------------
df = pd.read_sql_query(content.LOAD_SQL, conn)
m = content.query_match(df, "shophouse retail space on a busy main road")
assert abs(m.max() - 1.0) < 1e-9 and df.title[m.idxmax()] == "Demo shophouse J"
assert (content.query_match(df, "zzzz qqqq") == 0).all()
assert (content.query_match(df, "   ") == 0).all()

# --- recommend(query=...): text part used, weights switch, no query = old behaviour --
plain = rec.recommend(conn, 2_000_000_000, top_n=10)
assert plain.text_part.isna().all()
assert abs(rec.W_CONDITION + rec.W_BUDGET + rec.W_LOCATION - 1) < 1e-9
assert abs(rec.W_CONDITION_Q + rec.W_BUDGET_Q + rec.W_LOCATION_Q + rec.W_TEXT_Q - 1) < 1e-9
q = rec.recommend(conn, 2_000_000_000, top_n=10, query="townhouse near the university, rental demand from students")
assert q.text_part.between(0, 1).all() and abs(q.text_part.max() - 1) < 1e-9
assert q[q.title == "Demo townhouse G"].text_part.iloc[0] == q.text_part.max()
assert "matches your description" in q.why.iloc[0]
row = q.iloc[0]
expect = (rec.W_CONDITION_Q * row.condition_part + rec.W_BUDGET_Q * row.budget_part
          + rec.W_LOCATION_Q * row.location_part + rec.W_TEXT_Q * row.text_part)
assert abs(row.score - expect) < 1e-9

# --- old database without the description column is upgraded in place ------------
old = tmp / "old.db"
c = sqlite3.connect(old)
c.execute("CREATE TABLE properties (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, city TEXT NOT NULL, "
          "district TEXT, property_type TEXT, price_idr INTEGER NOT NULL, land_m2 REAL, building_m2 REAL, bedrooms INTEGER, "
          "source TEXT NOT NULL, listing_url TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now')))")
c.execute("INSERT INTO properties (title, city, price_idr, source) VALUES ('Demo house A', 'Jakarta', 1, 'synthetic_demo')")
c.commit(); c.close()
up = db.connect(old)
seed_demo.seed(up)
assert up.execute("SELECT description FROM properties WHERE title = 'Demo house A'").fetchone()[0]

# --- avoid flood-prone areas: flood safety joins the score -----------------------------
def fake_flood(lat, lon):
    return {"status": "ok", "value": 0.9} if lon > 106.9 else {"status": "no_data"}
f = rec.recommend(conn, 2_000_000_000, top_n=10, flood_fn=fake_flood)
nf = rec.recommend(conn, 2_000_000_000, top_n=10)
assert nf.flood_index.isna().all() and nf.flood_part.isna().all()
wet = f[f.flood_index.notna()]
dry = f[f.flood_index.isna() & f.latitude.notna()]
assert len(wet) and len(dry)
assert (abs(wet.flood_part - 0.1) < 1e-9).all() and (dry.flood_part == rec.FLOOD_UNKNOWN).all()
row = wet.iloc[0]; base = nf.set_index("id").loc[row.id, "score"]
assert abs(row.score - ((1 - rec.W_FLOOD) * base + rec.W_FLOOD * 0.1)) < 1e-9
assert "flood index 0.90" in row.why and "no data" in dry.iloc[0].why

print("all content checks passed")
