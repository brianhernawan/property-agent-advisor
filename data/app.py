#!/usr/bin/env python3
"""
app.py -- Streamlit front end: classify a photo -> rank properties -> ask the advisor.

    streamlit run app.py

Environment:
    API_URL          damage-classifier API (default http://localhost:8000)
    DB_PATH          SQLite file (default storage/advisor.db)
    GOOGLE_API_KEY, TAVILY_API_KEY   for the chatbot (see agent.py)
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import agent
import content
import db
import floodrisk
import recommender as rec
import seed_demo
from client import classify_image

API_URL = os.getenv("API_URL", "http://localhost:8000")
DB_PATH = os.getenv("DB_PATH", "storage/advisor.db")


def money(x) -> str:
    return f"IDR {float(x):,.0f}"


@st.cache_resource
def get_conn():
    conn = db.connect(DB_PATH)
    if os.getenv("SEED_DEMO", "1") == "1":
        seed_demo.seed(conn)
    return conn


def flood_cell(lat, lon) -> str:
    """Raw InaRISK index for the table: a number, 'no data' or 'unavailable'.
    floodrisk caches good answers; an 'unavailable' is retried on the next page load."""
    if pd.isna(lat) or pd.isna(lon):
        return "no coordinates"
    return floodrisk.short(floodrisk.flood_index(float(lat), float(lon)))


def property_label(row) -> str:
    return f"#{row['id']}  {row['title']} ({row['city']}, {money(row['price_idr'])})"


st.set_page_config(page_title="Property Due-Diligence Advisor", layout="wide")
st.title("Property Due-Diligence & Investment Advisor")
conn = get_conn()
props = pd.read_sql_query("SELECT * FROM properties ORDER BY id", conn)

if (props["source"] == "synthetic_demo").any():
    st.warning("Some properties are SYNTHETIC DEMO DATA (invented titles and prices). Load real listings with "
               "db.load_properties_csv before relying on any ranking.")

tab1, tab2, tab3 = st.tabs(["1. Classify a photo", "2. Rank properties", "3. Ask the advisor"])

# ---------------------------------------------------------------- 1. classify
with tab1:
    st.header("1. Classify a property photo")
    st.caption("Use a top-down satellite crop of ONE building. The model was trained on satellite images, not street photos. "
               "Its label is a screening signal, not an inspection.")
    if props.empty:
        st.info("No properties yet.")
    else:
        choice = st.selectbox("Property", props.to_dict("records"), format_func=property_label)
        photo = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
        if photo is not None:
            st.image(photo, width=260)
            if st.button("Classify and save"):
                try:
                    res = classify_image(API_URL, photo.getvalue(), photo.name)
                    db.save_prediction(conn, int(choice["id"]), res, photo.name)
                    st.success(f"{res['label']} ({res['confidence']:.0%} confidence). "
                               f"Chance of damage: {res['p_damaged']:.0%}. Saved to property #{choice['id']}.")
                    st.bar_chart(pd.Series(res["probs"], name="probability"))
                except RuntimeError as e:
                    st.error(str(e))

# ---------------------------------------------------------------- 2. recommend
shortlist = pd.DataFrame()
with tab2:
    st.header("2. Rank properties")
    c1, c2, c3, c4 = st.columns(4)
    budget = c1.number_input("Budget (IDR)", min_value=100_000_000, value=2_000_000_000, step=100_000_000)
    cities = ["(any)"] + sorted(props["city"].unique().tolist()) if not props.empty else ["(any)"]
    city = c2.selectbox("City", cities)
    districts = (["(any)"] + sorted(props[props["city"] == city]["district"].dropna().unique().tolist())
                 if city != "(any)" else ["(any)"])
    district = c3.selectbox("District", districts)
    top_n = c4.slider("Show top", 1, 10, 5)
    wish = st.text_input("Describe what you want (optional, content-based TF-IDF match)",
                         placeholder="e.g. quiet family house near schools with a garden and carport")

    if not props.empty:
        shortlist = rec.recommend(conn, budget, None if city == "(any)" else city,
                                  None if district == "(any)" else district, top_n, query=wish)
        shortlist["flood index"] = [flood_cell(la, lo) for la, lo in zip(shortlist["latitude"], shortlist["longitude"])]
        cols = ["id", "title", "city", "district", "price_idr", "condition_label", "flood index"]
        cols += (["text_part"] if wish.strip() else []) + ["score", "why"]
        show = shortlist[cols].rename(columns={"price_idr": "price (IDR)", "condition_label": "condition",
                                               "text_part": "text match"})
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={"price (IDR)": st.column_config.NumberColumn(format="IDR %d"),
                                    "text match": st.column_config.NumberColumn(format="%.2f"),
                                    "score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")})
        st.caption("Flood index: raw BNPB InaRISK flood-hazard value (0 to 1) at the property's coordinates. "
                   "It is shown for information and is NOT part of the score. 'no data' means outside the mapped area, "
                   "not zero risk. Demo properties use the approximate district centre.")
        with st.expander("How the score works"):
            st.markdown(
                f"`score = {rec.W_CONDITION} x condition + {rec.W_BUDGET} x budget fit + {rec.W_LOCATION} x location`  \n"
                f"- **condition** = 1 - chance of damage from the CNN; not assessed = {rec.UNKNOWN_CONDITION} (neutral).  \n"
                f"- **budget fit** = 1 within budget, falling to 0 at {rec.OVER_BUDGET_TOLERANCE:.0%} over budget.  \n"
                f"- **location** = 1 same district, {rec.LOCATION_SAME_CITY} same city, 0 otherwise.  \n"
                f"With a description: `score = {rec.W_CONDITION_Q} x condition + {rec.W_BUDGET_Q} x budget fit + "
                f"{rec.W_LOCATION_Q} x location + {rec.W_TEXT_Q} x text match`, where **text match** is the TF-IDF "
                "cosine similarity between your words and each listing (best match = 1).  \n"
                "These weights are design choices, not fitted values.")

        st.subheader("Similar properties (content-based)")
        base = st.selectbox("More like this property", props.to_dict("records"), format_func=property_label,
                            key="similar_base")
        sim = content.similar(conn, int(base["id"]), top_n)
        if sim.empty:
            st.info("Not enough properties to compare.")
        else:
            st.dataframe(sim[["id", "title", "city", "district", "price_idr", "similarity", "text_sim", "numeric_sim",
                              "shared_terms"]].rename(columns={"price_idr": "price (IDR)", "text_sim": "text (TF-IDF)",
                                                               "numeric_sim": "numbers", "shared_terms": "shared keywords"}),
                         hide_index=True, width="stretch",
                         column_config={"price (IDR)": st.column_config.NumberColumn(format="IDR %d"),
                                        "similarity": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
                                        "text (TF-IDF)": st.column_config.NumberColumn(format="%.2f"),
                                        "numbers": st.column_config.NumberColumn(format="%.2f")})
            st.caption(f"similarity = {content.W_TEXT} x TF-IDF cosine similarity of the listing text "
                       f"(description, type, city, district, bedrooms) + {content.W_NUMERIC} x closeness on price, land, "
                       "building size and bedrooms. Design-choice weights. Demo descriptions are synthetic.")

# ---------------------------------------------------------------- 3. chat
with tab3:
    st.header("3. Ask the advisor")
    st.caption(f"Gemini key: {'loaded' if os.getenv('GOOGLE_API_KEY') else 'MISSING'}  |  "
               f"Tavily key: {'loaded' if os.getenv('TAVILY_API_KEY') else 'missing (falls back to DuckDuckGo)'}  |  "
               f"model: {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}")
    if not os.getenv("GOOGLE_API_KEY"):
        st.info("Chat is off: set GOOGLE_API_KEY (and TAVILY_API_KEY for live price search) in the .env file and restart.")
    else:
        if "chat" not in st.session_state:
            st.session_state.chat = []
        # chat_input inside a tab is rendered inline, so keep the history in a
        # fixed-height scroll box and put the input directly under it.
        box = st.container(height=560, border=True)
        q = st.chat_input("e.g. What do 3-bedroom houses for sale in Duren Sawit cost, and how is property #3 doing?",
                          key="advisor_input")
        with box:
            for m in st.session_state.chat:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            if q:
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"), st.spinner("Searching and thinking..."):
                    try:
                        ag = agent.build_agent(conn, agent.shortlist_text(shortlist))
                        out = agent.ask(ag, st.session_state.chat, q)
                        text = out["answer"]
                        if out["sources"]:
                            text += "\n\nSources:\n" + "\n".join(f"- {u}" for u in out["sources"])
                    except Exception as e:  # network, quota, bad key: show it, keep the app alive
                        text = f"The advisor hit an error: {e}"
                    st.markdown(text)
                st.session_state.chat += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]
