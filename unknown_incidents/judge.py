"""
LLM-as-judge for Tier 2 investigations.
Routing accuracy is scored by exact match against labels in evaluate.py; the judge scores what labels can't:
is every evidence line backed by a tool output, did the agent invent facts, does the evidence justify the
decision, and can an L2 engineer act on the work notes.

  python -m unknown_incidents.evaluate --max-tier 2 --judge   # judge every ticket that reached Tier 2
  python -m unknown_incidents.judge --selftest                # check the judge itself on known good and bad cases first

Use a different (ideally stronger) model than Tier 2 as the judge: a model tends to favour its own answers.
"""
import asyncio
import sys
from os import getenv
from typing import Literal

import openai
from agents import Agent, AgentsException, ModelSettings, Runner
from pydantic import BaseModel, Field

from unknown_incidents.tier1_triage import CLASSES, UNKNOWN, usage_of

JUDGE_MODEL = getenv("JUDGE_MODEL", "gpt-4o")
MIN_ACTIONABLE = 4  # notes_actionable at or above this passes


class ClaimCheck(BaseModel):
    claim: str = Field(description="One factual claim from the evidence or work notes")
    supported: bool = Field(description="True only if a tool output states this fact (paraphrase is fine)")


class JudgeVerdict(BaseModel):
    claim_checks: list[ClaimCheck] = Field(description="Every factual claim in the evidence and work notes")
    fabricated_facts: list[str] = Field(description="Facts the agent stated that no tool output contains; empty if none")
    decision_supported: bool = Field(description="True if the evidence justifies the action and class chosen")
    notes_actionable: Literal[1, 2, 3, 4, 5] = Field(
        description="5: says what was checked, what was found and a specific next step; "
                    "3: findings present but the next step is vague; 1: missing, generic or wrong")
    rationale: str = Field(description="Two or three sentences explaining the verdict")


INSTRUCTIONS = (
    "You audit an AI agent's investigation of a store IT incident. Judge ONLY against the tool outputs shown: "
    "a fact is supported only if a tool output states it; ignore your own knowledge of stores or systems. "
    "List every factual claim in the agent's evidence and work notes and mark each supported or not. "
    "List facts that appear in no tool output as fabricated. The decision is supported if the evidence justifies "
    "it: ROUTE needs evidence for that specific class; ESCALATE is supported when no allowed class fits or the "
    "evidence is inconclusive. Score the work notes for an L2 engineer using the scale in the schema. Be strict."
)

judge_agent = Agent(
    name="Tier 2 Judge",
    instructions=INSTRUCTIONS,
    model=JUDGE_MODEL,
    model_settings=ModelSettings(temperature=0),
    output_type=JudgeVerdict,
)


def build_prompt(ticket: str, result: dict) -> str:
    trace = "\n".join(
        f"- {t['tool']}({t['arguments']}) → {t['output']}" for t in result.get("tool_trace", [])
    ) or "(the agent called no tools)"
    evidence = "\n".join(f"- {e}" for e in result.get("evidence", [])) or "(none)"
    return (
        f"Allowed classes: {', '.join(CLASSES)}, {UNKNOWN}\n\n"
        f"Incident:\n{ticket}\n\n"
        f"Tool calls and outputs:\n{trace}\n\n"
        f"Agent decision: {result['action']} class={result.get('class') or UNKNOWN} "
        f"confidence={result.get('confidence')}\n"
        f"Agent evidence:\n{evidence}\n\n"
        f"Agent work notes:\n{result.get('work_notes') or '(none)'}"
    )


def verdict_summary(v: JudgeVerdict) -> dict:
    claims = len(v.claim_checks)
    supported = sum(c.supported for c in v.claim_checks)
    grounded = supported == claims and not v.fabricated_facts
    return {
        "passed": grounded and v.decision_supported and v.notes_actionable >= MIN_ACTIONABLE,
        "grounded": grounded,
        "claims": claims,
        "supported_claims": supported,
        "fabricated_facts": v.fabricated_facts,
        "decision_supported": v.decision_supported,
        "notes_actionable": v.notes_actionable,
        "rationale": v.rationale,
        "unsupported_claims": [c.claim for c in v.claim_checks if not c.supported],
    }


async def judge_investigation(ticket: str, result: dict) -> dict:
    """Judge one Tier 2 result. Returns the verdict summary plus the judge's own token usage."""
    try:
        run = await Runner.run(judge_agent, build_prompt(ticket, result))
    except (AgentsException, openai.OpenAIError) as e:
        return {"error": f"judge failed: {e}"}
    return {**verdict_summary(run.final_output), "usage": usage_of(run, JUDGE_MODEL)}


# --- self-test: known cases the judge must get right before you trust its scores ---
_TRACE = [
    {"tool": "get_incident", "arguments": '{"incident_number": "INC9001"}',
     "output": '{"number": "INC9001", "store": "2111", "opened_at": "2026-10-23 07:12"}'},
    {"tool": "search_knowledge_base", "arguments": '{"query": "registers cannot scan"}',
     "output": '[{"id": "KB0101", "body": "Check pos_service. If stopped, L2 restarts it from the store server console."}]'},
    {"tool": "run_store_diagnostic", "arguments": '{"store_number": "2111", "check": "pos_service"}',
     "output": '{"store": "2111", "check": "pos_service", "result": "stopped since 06:58"}'},
]
_TICKET = "INC9001: Store 2111 registers can't scan any items since 7am"
_GOOD = {"action": "ESCALATE", "class": None, "confidence": 0.3, "tool_trace": _TRACE,
         "evidence": ["INC9001: store 2111, opened 07:12", "pos_service on store 2111: stopped since 06:58",
                      "KB0101: if pos_service is stopped, L2 restarts it from the store server console"],
         "work_notes": "Checked INC9001, KB0101 and the pos_service diagnostic for store 2111: pos_service has been "
                       "stopped since 06:58. No triage class covers POS outages. Next step: restart pos_service from "
                       "the store server console per KB0101, then confirm registers can scan."}
SELFTEST_CASES = [
    ("grounded, justified, actionable", _GOOD, {"passed": True}),
    ("fabricated fact", {**_GOOD, "evidence": _GOOD["evidence"] + ["pos_service was restarted at 07:30 and failed again"]},
     {"grounded": False}),
    ("decision not supported", {**_GOOD, "action": "ROUTE", "class": "Pricing", "confidence": 0.95},
     {"decision_supported": False}),
    ("vague work notes", {**_GOOD, "work_notes": "Please look into this."}, {"notes_actionable_max": 2}),
]


async def selftest() -> int:
    print(f"Judge self-test with {JUDGE_MODEL} ({len(SELFTEST_CASES)} cases)")
    failures = 0
    for name, result, expect in SELFTEST_CASES:
        v = await judge_investigation(_TICKET, result)
        if "error" in v:
            print(f"  ✗ {name}: {v['error']}")
            failures += 1
            continue
        ok = all(v["notes_actionable"] <= want if key == "notes_actionable_max" else v[key] == want
                 for key, want in expect.items())
        failures += not ok
        print(f"  {'✓' if ok else '✗'} {name}: expected {expect}, judge said "
              f"passed={v['passed']} grounded={v['grounded']} decision_supported={v['decision_supported']} "
              f"actionable={v['notes_actionable']}")
        if not ok:
            print(f"      rationale: {v['rationale']}")
    print("Judge agrees with every known case" if not failures else
          f"{failures} case(s) wrong: fix the judge prompt or use a stronger JUDGE_MODEL before trusting its scores")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(asyncio.run(selftest()))
    print(__doc__)
