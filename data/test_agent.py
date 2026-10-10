#!/usr/bin/env python3
"""
test_agent.py -- tests the agent WIRING with a scripted fake model and a stubbed web search.
No Gemini or Tavily key is used, so this proves the tools, the plumbing and the error handling,
NOT the quality of Gemini's answers. Run:  python3 test_agent.py
"""
import json
import tempfile
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.agents import create_agent

import agent
import floodrisk
import db
import recommender as rec
import seed_demo


class ScriptedModel(BaseChatModel):
    """Plays back a fixed list of AIMessages, one per model call."""
    script: list
    i: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        msg = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


conn = db.connect(Path(tempfile.mkdtemp()) / "t.db")
seed_demo.seed(conn)
db.save_prediction(conn, 1, {"label": "minor-damage", "confidence": 0.81, "p_damaged": 0.7, "damaged": True}, "a.jpg")
tools = agent.make_tools(conn)

# --- tools on their own ------------------------------------------------------------------
search_tool, cond_tool, flood_tool, similar_tool, area_tool = tools
floodrisk._geocode_fetch = lambda place: [{"lat": "-6.16", "lon": "106.90", "display_name": "Kelapa Gading, Jakarta Utara"}]
floodrisk._fetch = lambda la, lo: {"value": "0.81"}
floodrisk.clear_cache()
ar = json.loads(area_tool.invoke({"place": "Kelapa Gading, Jakarta"}))
assert ar["status"] == "ok" and ar["value"] == 0.81 and "Kelapa Gading" in ar["resolved_to"] and "one point" in ar["note_point"]
floodrisk.clear_cache()
sm = json.loads(similar_tool.invoke({"property_id": 3}))
assert len(sm["similar"]) == 3 and all(0 <= r["similarity"] <= 1 for r in sm["similar"]) and 3 not in [r["id"] for r in sm["similar"]]
assert json.loads(similar_tool.invoke({"property_id": 9999}))["status"].startswith("unknown")
assert json.loads(cond_tool.invoke({"property_id": 1}))["label"] == "minor-damage"
nc = json.loads(cond_tool.invoke({"property_id": 2}))
assert nc["status"] == "not assessed" and "label" not in nc  # no made-up label for an unclassified property

floodrisk._fetch = lambda lat, lon: {"value": "0.703704"}
floodrisk.clear_cache()
fr = json.loads(flood_tool.invoke({"property_id": 1}))
assert fr["status"] == "ok" and fr["value"] == 0.704 and "location_note" in fr, fr
assert json.loads(flood_tool.invoke({"property_id": 9999}))["status"] == "unknown property"
conn.execute("UPDATE properties SET latitude = NULL, longitude = NULL WHERE id = 2")
nf = json.loads(flood_tool.invoke({"property_id": 2}))
assert nf["status"] == "no coordinates" and "value" not in nf  # never a made-up number

agent.web_search = lambda q, max_results=5: [{"title": "Sample page", "url": "https://example.com/p1", "snippet": "price text"}]
assert "https://example.com/p1" in search_tool.invoke({"query": "rumah Duren Sawit"})

def boom(q, max_results=5):
    raise RuntimeError("no key")
agent.web_search = boom
assert search_tool.invoke({"query": "x"}).startswith("SEARCH_ERROR")  # error is returned to the model, not raised

agent.web_search = lambda q, max_results=5: [{"title": "Sample page", "url": "https://example.com/p1", "snippet": "price text"}]

# --- full loop: model calls both tools, then answers ---------------------------------------
fake = ScriptedModel(script=[
    call("get_condition", {"property_id": 1}, "c1"),
    call("search_prices", {"query": "rumah Duren Sawit harga"}, "c2"),
    AIMessage(content=[{"type": "text", "text": "Demo house A is minor-damage (81% conf). See https://example.com/p1."}]),
])
ag = create_agent(fake, tools, system_prompt=agent.SYSTEM_PROMPT.format(shortlist=""))
res = agent.ask(ag, [], "Is Demo house A in good shape and what do similar houses cost?")
assert res["tools_used"] == ["get_condition", "search_prices"], res
assert res["sources"] == ["https://example.com/p1"], res
assert "minor-damage" in res["answer"]            # list-of-blocks content is flattened to text

# --- history is passed through and only the new turn is inspected -----------------------------
fake2 = ScriptedModel(script=[AIMessage(content="Noted.")])
ag2 = create_agent(fake2, tools, system_prompt="x")
res2 = agent.ask(ag2, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], "thanks")
assert res2["answer"] == "Noted." and res2["tools_used"] == []

# --- shortlist context -----------------------------------------------------------------------
top = rec.recommend(conn, 2_000_000_000, "Jakarta", top_n=2)
txt = agent.shortlist_text(top)
assert "IDR" in txt and f"id {int(top.id[0])}" in txt
assert agent.shortlist_text(None) == ""

print("all agent wiring checks passed")
