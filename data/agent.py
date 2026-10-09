#!/usr/bin/env python3
"""
agent.py -- the advisor chatbot: a LangChain tool-calling agent (Gemini 2.5 Flash)
with four tools, replacing the earlier vector-DB/RAG plan (Rizky's Checkpoint 1 feedback).

    search_prices(query)         live web search for Indonesian property prices (Tavily, DuckDuckGo fallback)
    get_condition(property_id)   the CNN damage result stored in SQLite for one property
    get_flood_risk(property_id)  BNPB InaRISK flood-hazard index at the property's coordinates
    find_similar(property_id)    content-based recommender: listings most like this one (TF-IDF + numbers)

Keys come from the environment, never from code:
    GOOGLE_API_KEY   Gemini (Google AI Studio)
    TAVILY_API_KEY   Tavily search
    GEMINI_MODEL     optional, default gemini-2.5-flash
"""
from __future__ import annotations

import json
import os

from langchain.agents import create_agent
from langchain_core.tools import tool

import content
import db
import floodrisk

DEFAULT_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a property due-diligence assistant for buyers in Indonesia.

Rules:
- Put a currency label on every money figure (IDR or USD). Never write a bare number for a price.
- For prices or market questions, call search_prices first. Quote only what the search results say,
  and cite the source URL next to each figure. If the results do not contain a price, say so. Never guess a price.
- Treat sale prices and rental prices separately. Answer with sale (asking) prices unless the user asks about renting.
  If you mention a rental price, label it as rent and give the period (per month or per year); never mix it into a sale range.
  Match the bedroom count and the area the user asked for, and say what each price refers to (for example "3-bedroom house, asking price").
- For a property's condition, call get_condition with its id. Report p_damaged (the chance of damage) first, then the label.
  If the label says no-damage but p_damaged is 0.5 or higher, say the result is uncertain and lean on p_damaged.
  Say this is a screening signal from a satellite-image model, not a building inspection.
- If get_condition says the property was not assessed, tell the user to classify a photo first. Do not guess.
- For flood questions, call get_flood_risk with the property id. Report the raw index exactly as returned and say it
  comes from BNPB InaRISK. Do not invent a class such as low, medium or high, and do not say a property is safe.
  If the status is no_data, say the point is outside the mapped hazard area and that this is not proof of zero risk.
  If the property is a synthetic demo, say its coordinates are only the approximate district centre.
- For "similar to" or "more like" questions, call find_similar with the property id and give the similarity and
  shared keywords for each result. Do not name the tools in your answer.
- Keep answers short and in English. End with one concrete next step.

{shortlist}"""


# ----------------------------------------------------------------------------
# web search (separate function so tests can replace it)
# ----------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Return [{title, url, snippet}]. Tavily first; DuckDuckGo if Tavily fails or has no key."""
    try:
        from langchain_tavily import TavilySearch
        raw = TavilySearch(max_results=max_results, country="indonesia").invoke({"query": query})
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": (r.get("content") or "")[:500]}
                for r in raw.get("results", [])]
    except Exception as tavily_error:
        try:
            from ddgs import DDGS
            hits = DDGS().text(query, region="id-id", max_results=max_results)
            return [{"title": h.get("title", ""), "url": h.get("href", ""), "snippet": (h.get("body") or "")[:500]}
                    for h in hits]
        except Exception as ddg_error:
            raise RuntimeError(f"web search unavailable (tavily: {tavily_error}; duckduckgo: {ddg_error})")


def make_tools(conn):
    """The four tools, bound to one database connection."""

    @tool
    def search_prices(query: str) -> str:
        """Search the web for Indonesian property prices or market information.
        Include the city or district and the property type in the query, for example
        'rumah 3 kamar tidur Duren Sawit Jakarta Timur harga'. Returns titles, URLs and snippets."""
        try:
            hits = web_search(query)
        except Exception as e:
            return f"SEARCH_ERROR: {e}"
        if not hits:
            return "No results found."
        return "\n\n".join(f"[{i + 1}] {h['title']}\n{h['url']}\n{h['snippet']}" for i, h in enumerate(hits))

    @tool
    def get_condition(property_id: int) -> str:
        """Get the stored damage-classifier result for one property by its numeric id.
        Returns label (no-damage, minor-damage, major-damage, destroyed), confidence and p_damaged."""
        row = db.get_condition(conn, int(property_id))
        if row is None:
            return json.dumps({"property_id": property_id, "status": "not assessed",
                               "note": "no photo has been classified for this property yet"})
        return json.dumps({"property_id": property_id, "status": "assessed", **row})

    @tool
    def get_flood_risk(property_id: int) -> str:
        """Get the BNPB InaRISK flood-hazard index (a raw number from 0 to 1) at one property's coordinates.
        Use for flood questions. Returns status ok, no_data or error."""
        loc = db.get_location(conn, int(property_id))
        if loc is None:
            return json.dumps({"property_id": property_id, "status": "unknown property"})
        if loc["latitude"] is None or loc["longitude"] is None:
            return json.dumps({"property_id": property_id, "status": "no coordinates",
                               "note": "this property has no latitude/longitude, so flood risk cannot be looked up"})
        res = floodrisk.flood_index(float(loc["latitude"]), float(loc["longitude"]))
        out = {"property_id": property_id, "district": loc["district"], **res}
        if loc["source"] == "synthetic_demo":
            out["location_note"] = "demo property: coordinates are the approximate district centre, not an address"
        return json.dumps(out)

    @tool
    def find_similar(property_id: int) -> str:
        """Find the listings most similar to one property (content-based: TF-IDF on the listing text plus
        closeness on price, size and bedrooms). Returns up to 3 properties with similarity 0 to 1."""
        sim = content.similar(conn, int(property_id), 3)
        if sim.empty:
            return json.dumps({"property_id": property_id, "status": "unknown property or nothing to compare"})
        rows = [{"id": int(r.id), "title": r.title, "city": r.city, "district": r.district,
                 "price_idr": int(r.price_idr), "similarity": round(float(r.similarity), 2),
                 "shared_keywords": r.shared_terms} for r in sim.itertuples()]
        return json.dumps({"property_id": property_id, "similar": rows})

    return [search_prices, get_condition, get_flood_risk, find_similar]


def shortlist_text(df) -> str:
    """Turn the recommender's table into a short block of context for the system prompt."""
    if df is None or len(df) == 0:
        return ""
    lines = [f"- id {int(r.id)}: {r.title}, {r.city}, price IDR {int(r.price_idr):,}" for r in df.itertuples()]
    return "The user's current shortlist (use these ids with get_condition):\n" + "\n".join(lines)


def build_agent(conn, shortlist: str = "", model_name: str | None = None):
    """Create the agent. Needs GOOGLE_API_KEY in the environment."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
                                 temperature=0.2, max_retries=2, timeout=60)
    return create_agent(llm, make_tools(conn), system_prompt=SYSTEM_PROMPT.format(shortlist=shortlist))


def message_text(msg) -> str:
    """Gemini may return a string or a list of content blocks; always give back plain text."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def ask(agent, history: list[dict], question: str) -> dict:
    """Run one turn. history is [{'role': 'user'|'assistant', 'content': str}, ...].
    Returns {'answer': str, 'tools_used': [names], 'sources': [urls]}."""
    messages = history + [{"role": "user", "content": question}]
    result = agent.invoke({"messages": messages})
    out = result["messages"]
    new = out[len(messages):]  # only what this turn added
    tools_used = [tc["name"] for m in new if getattr(m, "tool_calls", None) for tc in m.tool_calls]
    sources = []
    for m in new:
        if getattr(m, "type", "") == "tool" and getattr(m, "name", "") == "search_prices":
            sources += [ln for ln in message_text(m).splitlines() if ln.startswith("http")]
    return {"answer": message_text(out[-1]), "tools_used": tools_used, "sources": list(dict.fromkeys(sources))}
