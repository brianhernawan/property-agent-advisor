#!/usr/bin/env python3
"""
content.py -- content-based recommender (TF-IDF + cosine similarity).

Two uses:

    similar(conn, property_id)   properties most like one property ("more like #3")
    query_match(df, text)        how well each property matches a free-text wish, 0 to 1

Each property becomes one text "profile": its listing description plus its type, city, district
and bedroom count written as words. TF-IDF turns every profile into a vector in which words that
are frequent in this listing but rare across all listings weigh most; cosine similarity compares
two vectors (1 = same words in the same proportions, 0 = nothing in common).

similar() blends that text similarity with a numeric similarity on price, land, building size and
bedrooms, because two listings can share words and still be in very different price brackets:

    similarity = W_TEXT * cosine(TF-IDF) + W_NUMERIC * (1 - mean scaled gap on the numbers)

The weights are DESIGN CHOICES, not fitted values. With only a handful of listings TF-IDF has little
to learn from; it becomes more useful as real listings with descriptions are loaded.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

W_TEXT, W_NUMERIC = 0.6, 0.4                       # design choice, must sum to 1
NUMERIC_COLS = ["price_idr", "land_m2", "building_m2", "bedrooms"]

LOAD_SQL = """
SELECT id, title, city, district, property_type, price_idr, land_m2, building_m2, bedrooms,
       description, source
FROM properties
"""


def _txt(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def profile_text(row) -> str:
    """One listing as text: description + structured fields written as words."""
    parts = [_txt(row.get(c)) for c in ("description", "property_type", "city", "district")]
    beds = row.get("bedrooms")
    if beds is not None and not pd.isna(beds):
        parts.append(f"{int(beds)} bedroom" if int(beds) else "commercial no bedroom")
    return " ".join(p for p in parts if p).lower()


def fit(df: pd.DataFrame):
    """Fit TF-IDF on all profiles. Returns (vectorizer, sparse matrix with one row per property)."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                          token_pattern=r"(?u)\b[\w-]+\b")
    return vec, vec.fit_transform(df.apply(profile_text, axis=1))


def numeric_similarity(df: pd.DataFrame, i: int) -> np.ndarray:
    """1 - mean min-max-scaled absolute gap to row i on the numeric columns. Missing values are skipped."""
    num = df[NUMERIC_COLS].astype(float)
    span = (num.max() - num.min()).replace(0, 1.0)
    gaps = ((num - num.iloc[i]).abs() / span)
    return (1 - gaps.mean(axis=1, skipna=True).fillna(1.0)).to_numpy()


def shared_terms(vec, X, i: int, j: int, k: int = 4) -> str:
    """The k terms that contribute most to the text similarity of rows i and j."""
    contrib = X[i].multiply(X[j]).toarray().ravel()
    top = [t for t in contrib.argsort()[::-1][:k] if contrib[t] > 0]
    names = vec.get_feature_names_out()
    return ", ".join(names[t] for t in top)


def similar(conn: sqlite3.Connection, property_id: int, top_n: int = 5) -> pd.DataFrame:
    """The top_n properties most similar to property_id, best first, with each similarity part."""
    df = pd.read_sql_query(LOAD_SQL, conn)
    if df.empty or property_id not in set(df["id"]):
        return pd.DataFrame()
    vec, X = fit(df)
    i = int(np.flatnonzero(df["id"].to_numpy() == property_id)[0])
    df["text_sim"] = cosine_similarity(X[i], X).ravel()
    df["numeric_sim"] = numeric_similarity(df, i)
    df["similarity"] = W_TEXT * df["text_sim"] + W_NUMERIC * df["numeric_sim"]
    df["shared_terms"] = [shared_terms(vec, X, i, j) for j in range(len(df))]
    out = df.drop(index=i)
    return out.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)


def query_match(df: pd.DataFrame, query: str) -> pd.Series:
    """Cosine similarity between a free-text wish and each property's profile, scaled so the best match = 1.
    Empty query -> 0 for everyone. Words the listings never use are ignored by TF-IDF."""
    if not query or not query.strip() or df.empty:
        return pd.Series(0.0, index=df.index)
    vec, X = fit(df)
    sims = cosine_similarity(vec.transform([query.lower()]), X).ravel()
    best = sims.max()
    return pd.Series(sims / best if best > 0 else sims, index=df.index)
