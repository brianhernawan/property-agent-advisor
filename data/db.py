#!/usr/bin/env python3
"""
db.py -- the SQLite database behind the advisor: schema + small helper functions.

Three tables:
    properties        one row per property for sale (price in IDR)
    rppi              Bank Indonesia Residential Property Price Index, loaded once from a CSV
    cnn_predictions   every damage-classifier result, linked to a property

and one view:
    v_latest_condition   the newest cnn_predictions row per property

Run directly to create an empty database:
    python3 db.py --db storage/advisor.db
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS properties (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT    NOT NULL,
    city          TEXT    NOT NULL,
    district      TEXT,
    property_type TEXT,                       -- house, townhouse, shophouse, ...
    price_idr     INTEGER NOT NULL CHECK (price_idr > 0),   -- asking price, Indonesian rupiah (IDR)
    land_m2       REAL,
    building_m2   REAL,
    bedrooms      INTEGER,
    source        TEXT    NOT NULL,           -- where the row came from, e.g. 'synthetic_demo' or 'listing_csv'
    listing_url   TEXT,
    description   TEXT,                       -- free-text listing description; used by the TF-IDF recommender
    latitude      REAL,                       -- WGS84 degrees; used for the flood-hazard lookup
    longitude     REAL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- BI publishes the index per city, per quarter, per house size class.
CREATE TABLE IF NOT EXISTS rppi (
    city            TEXT    NOT NULL,
    year            INTEGER NOT NULL,
    quarter         INTEGER NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    house_type      TEXT    NOT NULL,         -- e.g. small / medium / large
    index_value     REAL    NOT NULL,
    yoy_growth_pct  REAL,                     -- year-on-year change in %, may be missing
    PRIMARY KEY (city, year, quarter, house_type)
);

CREATE TABLE IF NOT EXISTS cnn_predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id   INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    image_name    TEXT,
    label         TEXT    NOT NULL,           -- no-damage / minor-damage / major-damage / destroyed
    confidence    REAL    NOT NULL,           -- probability of that label, 0-1
    p_damaged     REAL    NOT NULL,           -- 1 - P(no-damage), 0-1
    damaged       INTEGER NOT NULL,           -- 1 if p_damaged >= 0.5
    model_info    TEXT,                       -- which model produced it (JSON text)
    predicted_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_pred_property ON cnn_predictions(property_id);

CREATE VIEW IF NOT EXISTS v_latest_condition AS
SELECT p.*
FROM cnn_predictions p
JOIN (SELECT property_id, MAX(id) AS last_id FROM cnn_predictions GROUP BY property_id) m
  ON p.id = m.last_id;
"""

PROPERTY_COLS = ["title", "city", "district", "property_type", "price_idr", "land_m2", "building_m2",
                 "bedrooms", "source", "listing_url", "description", "latitude", "longitude"]


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (and create if needed) the database, with foreign keys on."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)  # streamlit reruns on other threads
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    # older databases lack columns added later (coordinates, description): add them in place
    have = {r[1] for r in conn.execute("PRAGMA table_info(properties)")}
    for col, typ in (("latitude", "REAL"), ("longitude", "REAL"), ("description", "TEXT")):
        if col not in have:
            conn.execute(f"ALTER TABLE properties ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


def add_property(conn: sqlite3.Connection, **fields) -> int:
    """Insert one property and return its id. Missing optional fields become NULL."""
    unknown = set(fields) - set(PROPERTY_COLS)
    if unknown:
        raise ValueError(f"unknown property fields: {sorted(unknown)}")
    row = {c: fields.get(c) for c in PROPERTY_COLS}
    cur = conn.execute(f"INSERT INTO properties ({', '.join(PROPERTY_COLS)}) "
                       f"VALUES ({', '.join('?' for _ in PROPERTY_COLS)})", [row[c] for c in PROPERTY_COLS])
    conn.commit()
    return cur.lastrowid


def save_prediction(conn: sqlite3.Connection, property_id: int, result: dict, image_name: str | None = None) -> int:
    """Store one /classify response (the dict returned by serve_api.py) for a property."""
    import json
    cur = conn.execute(
        "INSERT INTO cnn_predictions (property_id, image_name, label, confidence, p_damaged, damaged, model_info) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (property_id, image_name, result["label"], result["confidence"], result["p_damaged"],
         int(result["damaged"]), json.dumps(result.get("model", {}))))
    conn.commit()
    return cur.lastrowid


def get_condition(conn: sqlite3.Connection, property_id: int) -> dict | None:
    """Latest CNN result for a property, or None if it was never classified."""
    row = conn.execute("SELECT label, confidence, p_damaged, damaged, image_name, predicted_at "
                       "FROM v_latest_condition WHERE property_id = ?", (property_id,)).fetchone()
    return dict(row) if row else None


def get_location(conn: sqlite3.Connection, property_id: int) -> dict | None:
    """Coordinates and basic facts for one property, or None if the id does not exist."""
    row = conn.execute("SELECT id, title, city, district, latitude, longitude, source "
                       "FROM properties WHERE id = ?", (property_id,)).fetchone()
    return dict(row) if row else None


def load_rppi_csv(conn: sqlite3.Connection, csv_path: str | Path) -> int:
    """Load the BI RPPI table once. Columns: city,year,quarter,house_type,index_value,yoy_growth_pct.
    Re-loading the same file replaces rows with the same (city, year, quarter, house_type)."""
    df = pd.read_csv(csv_path)
    need = ["city", "year", "quarter", "house_type", "index_value"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"RPPI csv is missing columns: {missing}")
    if "yoy_growth_pct" not in df.columns:
        df["yoy_growth_pct"] = None
    df = df[need + ["yoy_growth_pct"]].astype(object).where(df[need + ["yoy_growth_pct"]].notna(), None)
    conn.executemany("INSERT OR REPLACE INTO rppi VALUES (?, ?, ?, ?, ?, ?)", df.values.tolist())
    conn.commit()
    return len(df)


def load_properties_csv(conn: sqlite3.Connection, csv_path: str | Path, source: str = "listing_csv") -> int:
    """Load real listings from a CSV. Required columns: title,city,price_idr. Others are optional."""
    df = pd.read_csv(csv_path)
    for c in ("title", "city", "price_idr"):
        if c not in df.columns:
            raise ValueError(f"listings csv is missing column: {c}")
    n = 0
    for rec in df.to_dict("records"):
        clean = {k: (None if pd.isna(v) else v) for k, v in rec.items() if k in PROPERTY_COLS}
        clean["price_idr"] = int(clean["price_idr"])
        clean.setdefault("source", source)
        clean["source"] = clean.get("source") or source
        add_property(conn, **clean)
        n += 1
    return n


def latest_rppi(conn: sqlite3.Connection, city: str) -> pd.DataFrame:
    """Newest RPPI quarter available for a city, one row per house_type. Empty if the table has no data."""
    return pd.read_sql_query(
        "SELECT city, year, quarter, house_type, index_value, yoy_growth_pct FROM rppi "
        "WHERE lower(city) = lower(?) AND (year, quarter) = "
        "(SELECT year, quarter FROM rppi WHERE lower(city) = lower(?) ORDER BY year DESC, quarter DESC LIMIT 1)",
        conn, params=(city, city))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="storage/advisor.db")
    a = ap.parse_args()
    c = connect(a.db)
    print("tables:", [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")])
