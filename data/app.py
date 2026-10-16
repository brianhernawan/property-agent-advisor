#!/usr/bin/env python3
"""
app.py -- Streamlit front end, one page:

    1. Check a property   pick the house on a map (or search an address), upload a top-down photo, then one
                          button returns the area name, the damage estimate and the BNPB flood hazard together
    2. Rank properties    weighted + TF-IDF ranking, optional flood-aware score, similar properties
    3. Ask the advisor    chatbot with live price search and the same tools

    streamlit run app.py

Environment:
    API_URL          damage-classifier API (default http://localhost:8000)
    DB_PATH          SQLite file (default storage/advisor.db)
    GOOGLE_API_KEY, TAVILY_API_KEY   for the chatbot (see agent.py)
"""
from __future__ import annotations

import os

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import agent
import content
import db
import floodrisk
import recommender as rec
import seed_demo
from client import classify_image

API_URL = os.getenv("API_URL", "http://localhost:8000")
DB_PATH = os.getenv("DB_PATH", "storage/advisor.db")
DAMAGED_AT = 0.5                         # chance of damage at which a building is called damaged
JAKARTA = (-6.2, 106.85)
SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
NO_PROPERTY = "(not a listed property)"


def money(x) -> str:
    return f"IDR {float(x):,.0f}"


@st.cache_resource
def get_conn():
    conn = db.connect(DB_PATH)
    if os.getenv("SEED_DEMO", "1") == "1":
        seed_demo.seed(conn)
    return conn


def flood_cell(lat, lon) -> str:
    """Flood index for the table: '0.70 high', 'no data' or 'unavailable'. Good answers are cached in floodrisk."""
    if pd.isna(lat) or pd.isna(lon):
        return "no coordinates"
    res = floodrisk.flood_index(float(lat), float(lon))
    return f"{res['value']:.2f} {res['hazard_class'].split()[0]}" if res["status"] == "ok" else floodrisk.short(res)


def property_label(row) -> str:
    return f"#{row['id']}  {row['title']} ({row['city']}, {money(row['price_idr'])})"


def set_pin(lat: float, lon: float, how: str) -> None:
    st.session_state.pin = {"lat": round(float(lat), 6), "lon": round(float(lon), 6), "how": how}
    st.session_state.pop("check", None)          # an old result belongs to the old pin
    if not how.startswith("listing"):            # a new spot on the map is no longer the listed property,
        st.session_state.unlink_property = True  # so a damage result must not be saved to it


def on_property_change() -> None:
    """Choosing a listed property moves the pin to its stored coordinates."""
    row = st.session_state.get("check_property")
    if isinstance(row, dict) and pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
        set_pin(row["latitude"], row["longitude"], f"listing #{row['id']}")


def build_map(pin: dict | None) -> folium.Map:
    center, zoom = ((pin["lat"], pin["lon"]), 17) if pin else (JAKARTA, 11)
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(SATELLITE_TILES, attr="Esri, Maxar, Earthstar Geographics", name="Satellite").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map", show=False).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    if pin:
        folium.Marker([pin["lat"], pin["lon"]], tooltip="House location",
                      icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(m)
    return m


def show_flood(res: dict) -> None:
    if res["status"] == "ok":
        level = res["hazard_class"].split()[0]
        st.metric("Flood hazard index (BNPB)", f"{res['value']:.2f}", help=floodrisk.CLASS_SOURCE)
        msg = f"{res['hazard_class'].capitalize()} flood hazard"
        {"high": st.error, "medium": st.warning}.get(level, st.success)(
            {"high": f"⚠ {msg}: this area is flood-prone.",
             "medium": f"⚠ {msg}: check the street's flood history before buying."}.get(level, f"{msg}."))
        st.caption("Value at this one point, not the whole district. " + floodrisk.CLASS_SOURCE + ".")
    elif res["status"] == "no_data":
        st.info("Outside BNPB's mapped flood-hazard area. This is not proof of zero risk.")
    else:
        st.warning(f"Flood data unavailable: {res.get('note', 'lookup failed')}")


st.set_page_config(page_title="Property Due-Diligence Advisor", layout="wide")
st.title("Property Due-Diligence & Investment Advisor")
conn = get_conn()
props = pd.read_sql_query("SELECT * FROM properties ORDER BY id", conn)
records = props.to_dict("records")

if (props["source"] == "synthetic_demo").any():
    st.warning("Some properties are SYNTHETIC DEMO DATA (invented titles and prices). Load real listings with "
               "db.load_properties_csv before relying on any ranking.")

# ---------------------------------------------------------------- 1. check a property
st.header("1. Check a property")
st.caption("Find the house on the map, upload a top-down satellite crop of that ONE building, then press Check. "
           "You get the area, the damage estimate and the BNPB flood hazard together. Screening signals, not an inspection.")

s1, s2 = st.columns([3, 2])
with s1.form("address_search", clear_on_submit=False, border=False):
    a1, a2 = st.columns([4, 1], vertical_alignment="bottom")
    address = a1.text_input("Search an address or area", placeholder="e.g. Jl. Pluit Raya, Jakarta Utara")
    if a2.form_submit_button("Find", width="stretch") and address.strip():
        hit = floodrisk.geocode(address)
        if hit["status"] == "ok":
            set_pin(hit["lat"], hit["lon"], "address search")
        else:
            st.warning(hit.get("note", "address not found"))
if st.session_state.pop("unlink_property", False):
    st.session_state.check_property = NO_PROPERTY
s2.selectbox("Or start from a listed property", [NO_PROPERTY] + records, key="check_property",
             format_func=lambda r: r if isinstance(r, str) else property_label(r), on_change=on_property_change)

pin = st.session_state.get("pin")
out = st_folium(build_map(pin), key="house_map", height=420, use_container_width=True, returned_objects=["last_clicked"])
click = (out or {}).get("last_clicked")
if click and (click["lat"], click["lng"]) != st.session_state.get("last_click"):
    st.session_state.last_click = (click["lat"], click["lng"])
    set_pin(click["lat"], click["lng"], "map click")
    st.rerun()
st.caption("Click the map to drop the pin on the house. Satellite view helps find the right roof; "
           + (f"pin at {pin['lat']:.5f}, {pin['lon']:.5f} (from {pin['how']})." if pin else "no pin yet."))

p1, p2 = st.columns([3, 2], vertical_alignment="bottom")
photo = p1.file_uploader("Top-down photo of the building (optional, for the damage check)", type=["jpg", "jpeg", "png"])
run = p2.button("Check this property", type="primary", width="stretch", disabled=not (pin or photo))

if run:
    chosen = st.session_state.get("check_property")
    listed = chosen if isinstance(chosen, dict) else None
    result = {"pin": pin, "photo": photo.getvalue() if photo else None, "photo_name": photo.name if photo else None}
    with st.spinner("Checking location, damage and flood hazard..."):
        if pin:
            result["place"] = floodrisk.reverse_geocode(pin["lat"], pin["lon"])
            result["flood"] = floodrisk.flood_index(pin["lat"], pin["lon"])
        if photo:
            try:
                result["damage"] = classify_image(API_URL, photo.getvalue(), photo.name)
                if listed:
                    db.save_prediction(conn, int(listed["id"]), result["damage"], photo.name)
                    result["saved_to"] = int(listed["id"])
            except RuntimeError as e:
                result["damage_error"] = str(e)
    st.session_state.check = result

res = st.session_state.get("check")
if res:
    place, flood, dmg = res.get("place"), res.get("flood"), res.get("damage")
    area = place["area"] if place and place["status"] == "ok" else "location not named"
    bits = []
    if dmg:
        bits.append("likely damaged" if dmg["p_damaged"] >= DAMAGED_AT else "no damage detected")
    if flood and flood["status"] == "ok":
        bits.append(f"{flood['hazard_class'].split()[0]} flood hazard")
    st.subheader(f"{area}: {', '.join(bits)}" if bits else area)

    c_photo, c_loc, c_dmg, c_flood = st.columns([1, 1.2, 1.2, 1.4], border=True)
    with c_photo:
        st.markdown("**Photo**")
        if res["photo"]:
            st.image(res["photo"])
        else:
            st.caption("No photo uploaded, so no damage check.")
    with c_loc:
        st.markdown("**Location**")
        if res["pin"]:
            st.write(area)
            st.caption(f"{res['pin']['lat']:.5f}, {res['pin']['lon']:.5f} · from {res['pin']['how']}")
            if place and place["status"] != "ok":
                st.caption(place.get("note", ""))
        else:
            st.caption("No pin on the map, so no location or flood check.")
    with c_dmg:
        st.markdown("**Damage (CNN)**")
        if dmg:
            st.metric("Chance of damage", f"{dmg['p_damaged']:.0%}")
            (st.error if dmg["p_damaged"] >= DAMAGED_AT else st.success)(
                f"{dmg['label']} ({dmg['confidence']:.0%} confidence)")
            if res.get("saved_to"):
                st.caption(f"Saved to property #{res['saved_to']}; the ranking below uses it.")
            with st.expander("All class probabilities"):
                st.bar_chart(pd.Series(dmg["probs"], name="probability"))
        elif res.get("damage_error"):
            st.error(res["damage_error"])
        else:
            st.caption("Upload a photo to estimate damage.")
    with c_flood:
        st.markdown("**Flood (BNPB InaRISK)**")
        if flood:
            show_flood(flood)
        else:
            st.caption("Drop a pin to check the flood hazard.")

st.divider()

# ---------------------------------------------------------------- 2. rank properties
st.header("2. Rank properties")
shortlist = pd.DataFrame()
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
avoid_flood = st.checkbox("Avoid flood-prone areas (adds the BNPB flood index to the score)", value=False)

if not props.empty:
    shortlist = rec.recommend(conn, budget, None if city == "(any)" else city,
                              None if district == "(any)" else district, top_n, query=wish,
                              flood_fn=floodrisk.flood_index if avoid_flood else None)
    shortlist["flood index"] = [flood_cell(la, lo) for la, lo in zip(shortlist["latitude"], shortlist["longitude"])]
    cols = ["id", "title", "city", "district", "price_idr", "condition_label", "flood index"]
    cols += (["text_part"] if wish.strip() else []) + ["score", "why"]
    show = shortlist[cols].rename(columns={"price_idr": "price (IDR)", "condition_label": "condition",
                                           "text_part": "text match"})
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"price (IDR)": st.column_config.NumberColumn(format="IDR %d"),
                                "text match": st.column_config.NumberColumn(format="%.2f"),
                                "score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")})
    st.caption("Flood index: BNPB InaRISK flood-hazard value (0 to 1) at the property's coordinates with its BNPB class; "
               "higher = more flood-prone. " + (f"It counts for {rec.W_FLOOD:.0%} of the score (flood safety = 1 - index). "
               if avoid_flood else "Tick 'Avoid flood-prone areas' to add it to the score. ") +
               "'no data' means outside the mapped area, not zero risk, and is scored neutral. "
               "Demo properties use the approximate district centre.")
    with st.expander("How the score works"):
        st.markdown(
            f"`score = {rec.W_CONDITION} x condition + {rec.W_BUDGET} x budget fit + {rec.W_LOCATION} x location`  \n"
            f"- **condition** = 1 - chance of damage from the CNN; not assessed = {rec.UNKNOWN_CONDITION} (neutral).  \n"
            f"- **budget fit** = 1 within budget, falling to 0 at {rec.OVER_BUDGET_TOLERANCE:.0%} over budget.  \n"
            f"- **location** = 1 same district, {rec.LOCATION_SAME_CITY} same city, 0 otherwise.  \n"
            f"With a description: `score = {rec.W_CONDITION_Q} x condition + {rec.W_BUDGET_Q} x budget fit + "
            f"{rec.W_LOCATION_Q} x location + {rec.W_TEXT_Q} x text match`, where **text match** is the TF-IDF "
            "cosine similarity between your words and each listing (best match = 1).  \n"
            f"Avoid flood-prone areas: `score = {1 - rec.W_FLOOD:.2f} x score above + {rec.W_FLOOD} x flood safety`, "
            f"where **flood safety** = 1 - BNPB flood index (no data = {rec.FLOOD_UNKNOWN}, neutral).  \n"
            "These weights are design choices, not fitted values.")

    st.subheader("Similar properties (content-based)")
    base = st.selectbox("More like this property", records, format_func=property_label, key="similar_base")
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

st.divider()

# ---------------------------------------------------------------- 3. ask the advisor
st.header("3. Ask the advisor")
st.caption(f"Gemini key: {'loaded' if os.getenv('GOOGLE_API_KEY') else 'MISSING'}  |  "
           f"Tavily key: {'loaded' if os.getenv('TAVILY_API_KEY') else 'missing (falls back to DuckDuckGo)'}  |  "
           f"model: {os.getenv('GEMINI_MODEL', agent.DEFAULT_MODEL)}")
if not os.getenv("GOOGLE_API_KEY"):
    st.info("Chat is off: set GOOGLE_API_KEY (and TAVILY_API_KEY for live price search) in the .env file and restart.")
else:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    # inside a container the chat input renders inline under the history instead of pinned to the window bottom
    with st.container():
        box = st.container(height=520, border=True)
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
                    reply = agent.ask(ag, st.session_state.chat, q)
                    text = reply["answer"]
                    if reply["sources"]:
                        text += "\n\nSources:\n" + "\n".join(f"- {u}" for u in reply["sources"])
                except Exception as e:  # network, quota, bad key: show it, keep the app alive
                    text = f"The advisor hit an error: {e}"
                st.markdown(text)
            st.session_state.chat += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]
