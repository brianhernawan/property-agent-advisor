#!/usr/bin/env python3
"""
recommender.py -- rank properties by condition, budget fit, location and (optionally) a text match.

A property's score is a weighted sum of parts, each between 0 and 1:

    score = W_CONDITION * condition + W_BUDGET * budget_fit + W_LOCATION * location

When the user also describes what they want in words, a fourth, content-based part is added and the
weights switch to the W_*_Q set below:

    score = W_CONDITION_Q * condition + W_BUDGET_Q * budget_fit + W_LOCATION_Q * location + W_TEXT_Q * text_match

    condition   1 - p_damaged from the CNN (1.0 = clearly undamaged, 0.0 = clearly damaged).
                Properties never classified get UNKNOWN_CONDITION and are labelled 'not assessed'.
    budget_fit  1.0 when the price is within budget. Over budget it falls in a straight line to
                0.0 at (1 + OVER_BUDGET_TOLERANCE) x budget.
    location    1.0 same city and district, LOCATION_SAME_CITY same city only, 0.0 otherwise.
                No preferred city given -> 1.0 for everyone (location is not used to rank).
    flood_safety  1 - BNPB InaRISK flood index at the property (only when "avoid flood-prone areas" is on).
                No data / lookup failed -> FLOOD_UNKNOWN (neutral). Then:
                score = (1 - W_FLOOD) * score + W_FLOOD * flood_safety
    text_match  TF-IDF cosine similarity between the user's description and the listing (content.py),
                scaled so the best match = 1.0. Only used when a description is given.

All constants below are DESIGN CHOICES, not measured values. They are shown in the app so
the reader can change them. Each part is kept as its own column so every ranking can be explained.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

import content

W_CONDITION, W_BUDGET, W_LOCATION = 0.40, 0.35, 0.25   # design choice, must sum to 1
OVER_BUDGET_TOLERANCE = 0.30                           # 30% over budget scores 0 on budget fit
LOCATION_SAME_CITY = 0.60                              # same city, different district
UNKNOWN_CONDITION = 0.50                               # neutral score when no photo was classified
# with a free-text wish: design choice, must sum to 1
W_CONDITION_Q, W_BUDGET_Q, W_LOCATION_Q, W_TEXT_Q = 0.35, 0.30, 0.20, 0.15
W_FLOOD = 0.25                                          # share of the score given to flood safety when switched on
FLOOD_UNKNOWN = 0.50                                    # neutral flood safety when BNPB has no value for the point

LOAD_SQL = """
SELECT p.id, p.title, p.city, p.district, p.property_type, p.price_idr, p.land_m2, p.building_m2,
       p.bedrooms, p.description, p.source, p.latitude, p.longitude,
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


def flood_value(flood_fn, lat, lon) -> float:
    """The raw flood index for one point, or NaN when there are no coordinates / no data / an error."""
    if pd.isna(lat) or pd.isna(lon):
        return float("nan")
    res = flood_fn(float(lat), float(lon))
    return float(res["value"]) if res.get("status") == "ok" else float("nan")


def explain(row: pd.Series) -> str:
    """One plain sentence on why a property ranked where it did."""
    cond = (f"chance of damage {row.p_damaged:.0%} (top label: {row.condition_label})" if pd.notna(row.condition_label)
            else "condition not assessed")
    over = row.price_idr - row.budget_idr
    budget = "within budget" if over <= 0 else f"IDR {over:,.0f} over budget"
    text = f"; matches your description {row.text_part:.2f}" if pd.notna(row.get("text_part", float("nan"))) else ""
    return f"{cond}; {budget}; location score {row.location_part:.2f}{text}"


def recommend(conn: sqlite3.Connection, budget_idr: float, city: str | None = None,
              district: str | None = None, top_n: int = 5, query: str | None = None,
              flood_fn=None) -> pd.DataFrame:
    """Return the top_n properties, best first, with every score part and an explanation.
    query: optional free-text wish ("quiet house near schools with a garden") for the TF-IDF text match.
    flood_fn: optional function (lat, lon) -> floodrisk result dict. When given, flood safety joins the score."""
    if budget_idr <= 0:
        raise ValueError("budget_idr must be positive (Indonesian rupiah)")
    df = pd.read_sql_query(LOAD_SQL, conn)
    if df.empty:
        return df
    df["budget_idr"] = budget_idr
    df["condition_part"] = condition_score(df["p_damaged"])
    df["budget_part"] = budget_score(df["price_idr"], budget_idr)
    df["location_part"] = location_score(df, city, district)
    if query and query.strip():
        df["text_part"] = content.query_match(df, query)
        df["score"] = (W_CONDITION_Q * df["condition_part"] + W_BUDGET_Q * df["budget_part"]
                       + W_LOCATION_Q * df["location_part"] + W_TEXT_Q * df["text_part"])
    else:
        df["text_part"] = float("nan")
        df["score"] = (W_CONDITION * df["condition_part"] + W_BUDGET * df["budget_part"]
                       + W_LOCATION * df["location_part"])
    df["flood_index"] = float("nan")
    df["flood_part"] = float("nan")
    if flood_fn is not None:
        df["flood_index"] = [flood_value(flood_fn, la, lo) for la, lo in zip(df["latitude"], df["longitude"])]
        df["flood_part"] = (1 - df["flood_index"]).fillna(FLOOD_UNKNOWN)
        df["score"] = (1 - W_FLOOD) * df["score"] + W_FLOOD * df["flood_part"]
    df["assessed"] = df["p_damaged"].notna()
    df["why"] = df.apply(explain, axis=1)
    if flood_fn is not None:
        df["why"] = df["why"] + df["flood_index"].map(
            lambda v: f"; flood index {v:.2f}" if pd.notna(v) else "; flood index: no data (scored neutral)")
    return df.sort_values(["score", "price_idr"], ascending=[False, True]).head(top_n).reset_index(drop=True)
