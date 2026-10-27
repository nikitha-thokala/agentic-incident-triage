"""
Tier 2 AI triage: a tool-using agent that investigates the incidents Tier 0 and Tier 1 could not decide.
The agent (LLM) calls read-only tools on an MCP server, then routes the incident or escalates it to
L2 with evidence-backed work notes.

  python -m unknown_incidents.tier2_triage                 # run the test tickets through Tier 0 → Tier 1 → Tier 2
  python -m unknown_incidents.tier2_triage "your ticket"   # triage one ticket
"""
import asyncio
import sys
from os import getenv
from pathlib import Path
from typing import Literal

import openai
from agents import Agent, AgentsException, MaxTurnsExceeded, ModelSettings, Runner
from agents.mcp import MCPServerStdio, create_static_tool_filter
from pydantic import BaseModel, Field

from unknown_incidents.tier0_triage import tier0_triage
from unknown_incidents.tier1_triage import CLASSES, UNKNOWN, tier1_triage_async, usage_of

LLM_MODEL = getenv("TIER2_MODEL", "gpt-4o")  # a mid-tier model, e.g. gpt-6.1-sol if your key has it
CONFIDENCE_GATE = 0.85  # route only at or above this; tune with an eval
MAX_TURNS = 8           # caps tool calls and cost per incident
REPO_ROOT = Path(__file__).resolve().parent.parent  # the MCP server runs as a module from here
ALLOWED_TOOLS = ["search_similar_incidents", "get_incident", "search_knowledge_base", "run_store_diagnostic"]


class Tier2Decision(BaseModel):
    action: Literal["ROUTE", "ESCALATE"] = Field(description="ROUTE only when the evidence clearly supports one class")
    incident_class: str = Field(description=f"Exactly one of the allowed classes, or '{UNKNOWN}'")
    confidence: float = Field(description="Probability from 0.0 to 1.0 that the class is correct")
    evidence: list[str] = Field(description="Facts from tool results, each naming its source, e.g. 'KB0102: ...'")
    work_notes: str = Field(description="Notes for the L2 engineer: what was checked, what was found, next step")


INSTRUCTIONS = (
    "You are a Tier 2 investigator for store IT incidents that cheaper triage tiers could not route. "
    "Investigate with the tools: read the incident record if a number is given, look for similar resolved "
    "incidents, read the relevant knowledge-base article, and run the store diagnostics it calls for. "
    "Use only facts from tool results and never invent data. The tools are read-only: you never fix anything. "
    f"Choose ROUTE with one allowed class only when the evidence clearly supports it; otherwise ESCALATE with "
    f"class '{UNKNOWN}' or your best class. Always write work notes an L2 engineer can act on. "
    "Be calibrated: 0.9 or above only when diagnostics and past incidents agree."
)


def build_prompt(text: str, prior: dict | None) -> str:
    lines = [f"Allowed classes: {', '.join(CLASSES)}, {UNKNOWN}", "", f"New incident:\n{text}"]
    if prior:
        lines += ["", "Earlier tiers escalated it:"]
        if prior.get("tier0_reason"):
            lines.append(f"- Tier 0 (retrieval): {prior['tier0_reason']}")
        lines.append(f"- Tier {prior['tier']}: {prior['reason']}")
        if prior.get("llm_reasoning"):
            lines.append(f"- Tier 1 reasoning: {prior['llm_reasoning']}")
    return "\n".join(lines)


def _result(action: str, cls, confidence: float, reason: str, decision: Tier2Decision | None = None,
            tools_used: list[str] = ()):
    return {
        "tier": 2, "action": action, "class": cls, "confidence": round(confidence, 3), "reason": reason,
        "evidence": decision.evidence if decision else [],
        "work_notes": decision.work_notes if decision else reason,
        "tools_used": list(tools_used),
    }


async def tier2_triage(text: str, prior: dict | None = None) -> dict:
    mcp_server = MCPServerStdio(
        name="incident-tools",
        params={"command": sys.executable, "args": ["-m", "unknown_incidents.tier2_mcp_server"], "cwd": str(REPO_ROOT)},
        cache_tools_list=True,
        client_session_timeout_seconds=60,  # the first similar-incident search loads the embedding model
        tool_filter=create_static_tool_filter(allowed_tool_names=ALLOWED_TOOLS),
    )
    async with mcp_server:
        investigator = Agent(
            name="Tier 2 Incident Investigator",
            instructions=INSTRUCTIONS,
            model=LLM_MODEL,
            model_settings=ModelSettings(temperature=0),
            mcp_servers=[mcp_server],
            output_type=Tier2Decision,  # structured output: the SDK returns a validated Tier2Decision
        )
        try:
            run = await Runner.run(investigator, build_prompt(text, prior), max_turns=MAX_TURNS)
        except MaxTurnsExceeded:
            return _result("ESCALATE", None, 0.0, f"agent did not finish within {MAX_TURNS} turns")
        except (AgentsException, openai.OpenAIError) as e:
            return _result("ESCALATE", None, 0.0, f"agent failed: {e}")

    trace = tool_trace(run.new_items)
    return {**_decide(run.final_output, [t["tool"] for t in trace]), "usage": usage_of(run, LLM_MODEL),
            "tool_trace": trace}


def _field(raw, name):
    return raw.get(name) if isinstance(raw, dict) else getattr(raw, name, None)


def tool_trace(items, max_output_chars: int = 2000) -> list[dict]:
    """Each tool call the agent made, with its arguments and what the tool returned (for audit and the LLM judge)."""
    outputs = {_field(i.raw_item, "call_id"): i.output for i in items if i.type == "tool_call_output_item"}
    return [{"tool": _field(i.raw_item, "name") or "?",
             "arguments": _field(i.raw_item, "arguments") or "",
             "output": str(outputs.get(_field(i.raw_item, "call_id"), ""))[:max_output_chars]}
            for i in items if i.type == "tool_call_item"]


def _decide(decision: Tier2Decision, tools_used: list[str]) -> dict:
    cls = decision.incident_class.strip()
    confidence = min(max(decision.confidence, 0.0), 1.0)
    if decision.action == "ESCALATE" or cls == UNKNOWN:
        return _result("ESCALATE", None, confidence, "agent escalated to L2 with work notes", decision, tools_used)
    if cls not in CLASSES:  # the model must never invent a class
        return _result("ESCALATE", None, confidence, f"agent answered an unknown class '{cls}'", decision, tools_used)
    if confidence < CONFIDENCE_GATE:
        return _result("ESCALATE", None, confidence,
                       f"suggested '{cls}' at {confidence:.2f}, below the {CONFIDENCE_GATE} gate", decision, tools_used)
    return _result("ROUTE", cls, confidence, f"agent classified with confidence {confidence:.2f}", decision, tools_used)


async def triage_async(text: str, max_tier: int = 2) -> dict:
    """Tier 0 → Tier 1 → Tier 2. Each tier runs only when the cheaper one escalates.
    max_tier stops the cascade early (the eval uses it to run Tier 0 alone at no LLM cost).
    usage_trail holds the token counts of every LLM tier that ran for this ticket."""
    t0 = tier0_triage(text)
    if t0["action"] == "ROUTE" or max_tier == 0:
        return {**t0, "usage_trail": []}
    t1 = await tier1_triage_async(text)
    t1["tier0_reason"] = t0["reason"]
    t1["usage_trail"] = [t1["usage"]] if "usage" in t1 else []
    if t1["action"] == "ROUTE" or max_tier == 1:
        return t1
    t2 = await tier2_triage(text, prior=t1)
    t2["tier1_reason"] = t1["reason"]
    t2["usage_trail"] = t1["usage_trail"] + ([t2["usage"]] if "usage" in t2 else [])
    return t2


def triage(text: str) -> dict:
    return asyncio.run(triage_async(text))


async def _main(tickets: list[str]):
    # one event loop for every ticket: the OpenAI client's connections belong to the loop that opened them
    for t in tickets:
        _print(t, await triage_async(t))


def _print(text: str, r: dict):
    print(f"\n{text}\n  → Tier {r['tier']}: {r['action']} {r['class'] or ''} | {r['reason']}")
    if r.get("tier1_reason"):
        print(f"     (Tier 1 escalated: {r['tier1_reason']})")
    if r.get("tools_used"):
        print(f"     tools: {', '.join(r['tools_used'])}")
    for e in r.get("evidence", []):
        print(f"     • {e}")
    if r["tier"] == 2:
        print(f"     work notes: {r['work_notes']}")


if __name__ == "__main__":
    tickets = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "Alert BATCH01 triggered again at store 4532",                     # Tier 0 routes
        "INC9001: Store 2111 registers can't scan any items since 7am",  # no class fits: Tier 2 escalates with notes
        "INC9002: Item 55512 price missing at store 4532 after setup",   # ambiguous: Tier 2 investigates
    ]
    asyncio.run(_main(tickets))
