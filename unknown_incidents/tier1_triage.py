"""
Tier 1 AI triage: one LLM call with retrieved past incidents as context (augmented generation).
Runs only when Tier 0 escalates. Routes when the LLM is confident, otherwise escalates to Tier 2.

  python -m unknown_incidents.tier1_triage                 # run the test tickets through Tier 0 → Tier 1
  python -m unknown_incidents.tier1_triage "your ticket"   # triage one ticket
"""
import sys
from os import getenv

import openai
from agents import Agent, AgentsException, ModelSettings, Runner
from pydantic import BaseModel, Field

from unknown_incidents.tier0_triage import NAMESPACE, PAST_INCIDENTS, embed, get_index, tier0_triage

LLM_MODEL = getenv("OPENAI_MODEL", "gpt-4o")
CONFIDENCE_GATE = 0.80  # route only at or above this; tune with an eval
CONTEXT_K = 5           # past incidents given to the LLM as context
UNKNOWN = "unknown"
CLASSES = sorted({i["class"] for i in PAST_INCIDENTS})  # in production: the routable class list


class Tier1Decision(BaseModel):
    incident_class: str = Field(description=f"Exactly one of the allowed classes, or '{UNKNOWN}'")
    confidence: float = Field(description="Probability from 0.0 to 1.0 that the class is correct")
    reasoning: str = Field(description="One or two sentences citing the past incidents used")


classifier = Agent(
    name="Tier 1 Incident Classifier",
    model=LLM_MODEL,
    model_settings=ModelSettings(temperature=0),
    output_type=Tier1Decision,  # structured output: the SDK returns a validated Tier1Decision
    instructions=(
        "You classify store IT incidents. Choose exactly one class from the allowed list, "
        f"or '{UNKNOWN}' if none clearly fits. Base your answer only on the new incident and the "
        "similar past incidents provided, and never invent facts. Be calibrated: use 0.9 or above "
        "only when the evidence clearly supports the class, and below 0.5 when you are guessing."
    ),
)


def retrieve_context(text: str, k: int = CONTEXT_K) -> list[dict]:
    """Top-k past incidents with no threshold: weaker matches are still useful context for the LLM."""
    res = get_index().query(vector=embed([text])[0], top_k=k, include_metadata=True,
                            namespace=NAMESPACE)
    return [
        {"id": m["id"], "similarity": round(m["score"], 3), "text": m["metadata"]["text"],
         "class": m["metadata"]["class"], "resolution": m["metadata"]["resolution"]}
        for m in res["matches"]
    ]


def build_prompt(text: str, context: list[dict]) -> str:
    examples = "\n".join(
        f'- {c["id"]} (similarity {c["similarity"]}): "{c["text"]}" '
        f'→ class: {c["class"]}; resolution: {c["resolution"]}'
        for c in context
    ) or "(no similar incidents found)"
    return (
        f"Allowed classes: {', '.join(CLASSES)}, {UNKNOWN}\n\n"
        f"New incident:\n{text}\n\n"
        f"Similar past incidents (most similar first):\n{examples}"
    )


def _result(action: str, cls, confidence: float, reason: str, context: list[dict], reasoning: str = ""):
    return {
        "tier": 1, "action": action, "class": cls, "confidence": round(confidence, 3),
        "reason": reason, "llm_reasoning": reasoning,
        "evidence": [{"id": c["id"], "class": c["class"], "similarity": c["similarity"]} for c in context],
    }


def usage_of(run, model: str) -> dict:
    """Token counts for one run, so the eval can price each tier."""
    u = run.context_wrapper.usage
    return {"model": model, "requests": u.requests, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens}


def tier1_triage(text: str) -> dict:
    context = retrieve_context(text)
    try:
        run = Runner.run_sync(classifier, build_prompt(text, context))
    except (AgentsException, openai.OpenAIError) as e:
        return _result("ESCALATE", None, 0.0, f"LLM call failed: {e}", context)
    return {**_decide(run.final_output, context), "usage": usage_of(run, LLM_MODEL)}


async def tier1_triage_async(text: str) -> dict:
    """Same as tier1_triage, for callers already running an event loop (Tier 2)."""
    context = retrieve_context(text)
    try:
        run = await Runner.run(classifier, build_prompt(text, context))
    except (AgentsException, openai.OpenAIError) as e:
        return _result("ESCALATE", None, 0.0, f"LLM call failed: {e}", context)
    return {**_decide(run.final_output, context), "usage": usage_of(run, LLM_MODEL)}


def _decide(decision: Tier1Decision, context: list[dict]) -> dict:
    cls = decision.incident_class.strip()
    confidence = min(max(decision.confidence, 0.0), 1.0)
    if cls == UNKNOWN:
        return _result("ESCALATE", None, confidence, "LLM could not classify it", context,
                       decision.reasoning)
    if cls not in CLASSES:  # the model must never invent a class
        return _result("ESCALATE", None, confidence, f"LLM answered an unknown class '{cls}'",
                       context, decision.reasoning)
    if confidence < CONFIDENCE_GATE:
        return _result("ESCALATE", None, confidence,
                       f"suggested '{cls}' at {confidence:.2f}, below the {CONFIDENCE_GATE} gate",
                       context, decision.reasoning)
    return _result("ROUTE", cls, confidence, f"LLM classified with confidence {confidence:.2f}",
                   context, decision.reasoning)


def triage(text: str) -> dict:
    """Tier 0 first (no LLM). Tier 1 only when Tier 0 escalates."""
    t0 = tier0_triage(text)
    if t0["action"] == "ROUTE":
        return t0
    t1 = tier1_triage(text)
    t1["tier0_reason"] = t0["reason"]
    return t1


def _print(text: str, r: dict):
    print(f"\n{text}\n  → Tier {r['tier']}: {r['action']} {r['class'] or ''} | {r['reason']}")
    if r.get("tier0_reason"):
        print(f"     (Tier 0 escalated: {r['tier0_reason']})")
    if r.get("llm_reasoning"):
        print(f"     LLM: {r['llm_reasoning']}")
    for e in r["evidence"]:
        print(f"     {e}")


if __name__ == "__main__":
    tickets = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "Alert BATCH01 triggered again at store 4532",            # Tier 0 routes (alert code)
        "Cancel the advance shipping notice 00000007777",       # reworded: Tier 0 misses, Tier 1 routes
        "Store receipt printer is jammed",                      # no fitting class: Tier 1 escalates
        "Item price missing after new item setup",              # ambiguous: Tier 1 decides or escalates
    ]
    for t in tickets:
        _print(t, triage(t))
