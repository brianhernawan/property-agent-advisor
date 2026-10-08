#!/usr/bin/env python3
"""
recommender.py -- rank properties by condition, budget fit and location.

A property's score is a weighted sum of three parts, each between 0 and 1:

    score = W_CONDITION * condition + W_BUDGET * budget_fit + W_LOCATION * location

    condition   1 - p_damaged from the CNN (1.0 = clearly undamaged, 0.0 = clearly damaged).
                Properties never classified get UNKNOWN_CONDITION and are labelled 'not assessed'.
    budget_fit  1.0 when the price is within budget. Over budget it falls in a straight line to
                0.0 at (1 + OVER_BUDGET_TOLERANCE) x budget.
    location    1.0 same city and district, LOCATION_SAME_CITY same city only, 0.0 otherwise.
                No preferred city given -> 1.0 for everyone (location is not used to rank).

All constants below are DESIGN CHOICES, not measured values. They are shown in the app so
the reader can change them. Each part is kept as its own column so every ranking can be explained.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

W_CONDITION, W_BUDGET, W_LOCATION = 0.40, 0.35, 0.25   # design choice, must sum to 1
OVER_BUDGET_TOLERANCE = 0.30                           # 30% over budget scores 0 on budget fit
LOCATION_SAME_CITY = 0.60                              # same city, different district
UNKNOWN_CONDITION = 0.50                               # neutral score when no photo was classified

LOAD_SQL = """
SELECT p.id, p.title, p.city, p.district, p.property_type, p.price_idr, p.land_m2, p.building_m2,
       p.bedrooms, p.source, p.latitude, p.longitude,
       c.label AS condition_label, c.confidence AS condition_confidence, c.p_damaged
FROM properties p
LEFT JOIN v_latest_condition c ON c.property_id = p.id
"""


def condition_score(p_damaged: pd.Series) -> pd.Series:
    """1 - p_damaged; properties without a prediction (NaN) get the neutral UNKNOWN_CONDITION."""
    return (1 - p_damaged).fillna(UNKNOWN_CONDITION)


def budget_score(price_idr: pd.Series, budget_idr: float) -> pd.Series:
    """1.0 within budget, then a straight line down to 0.0 at budget x (1 + tolerance)."""
    over = (price_idr - budget_idr) / (budget_idr * OVER_BUDGET_TOLERANCE)
    return (1 - over).clip(lower=0.0, upper=1.0)


def location_score(df: pd.DataFrame, city: str | None, district: str | None) -> pd.Series:
    """Same city + district = 1.0, same city = LOCATION_SAME_CITY, else 0.0. No city given = 1.0."""
    if not city:
        return pd.Series(1.0, index=df.index)
    same_city = df["city"].str.lower() == city.lower()
    if district:
        same_district = same_city & (df["district"].fillna("").str.lower() == district.lower())
    else:
        same_district = same_city & False
    return same_city.astype(float) * LOCATION_SAME_CITY + same_district.astype(float) * (1 - LOCATION_SAME_CITY)


def explain(row: pd.Series) -> str:
    """One plain sentence on why a property ranked where it did."""
    cond = (f"chance of damage {row.p_damaged:.0%} (top label: {row.condition_label})" if pd.notna(row.condition_label)
            else "condition not assessed")
    over = row.price_idr - row.budget_idr
    budget = "within budget" if over <= 0 else f"IDR {over:,.0f} over budget"
    return f"{cond}; {budget}; location score {row.location_part:.2f}"


def recommend(conn: sqlite3.Connection, budget_idr: float, city: str | None = None,
              district: str | None = None, top_n: int = 5) -> pd.DataFrame:
    """Return the top_n properties, best first, with every score part and an explanation."""
    if budget_idr <= 0:
        raise ValueError("budget_idr must be positive (Indonesian rupiah)")
    df = pd.read_sql_query(LOAD_SQL, conn)
    if df.empty:
        return df
    df["budget_idr"] = budget_idr
    df["condition_part"] = condition_score(df["p_damaged"])
    df["budget_part"] = budget_score(df["price_idr"], budget_idr)
    df["location_part"] = location_score(df, city, district)
    df["score"] = (W_CONDITION * df["condition_part"] + W_BUDGET * df["budget_part"]
                   + W_LOCATION * df["location_part"])
    df["assessed"] = df["p_damaged"].notna()
    df["why"] = df.apply(explain, axis=1)
    return df.sort_values(["score", "price_idr"], ascending=[False, True]).head(top_n).reset_index(drop=True)
