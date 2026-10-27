"""Quality checks for the L2 research notes.

1. grounding_check: free and deterministic. Every finding must cite a source the agent actually touched, and
   every number in a finding must appear in some tool output. Catches most fabrication at zero cost.
2. llm_judge: LLM-as-judge for what rules can't see (is the cause supported, are the next steps actionable).
   Runs only in evaluation, on a cheap model, with cached verdicts, so it is not a per-ticket production cost.
"""
from __future__ import annotations

import json
import re

from .llm import LLM, LLMRequest
from .models import JudgeVerdict, ResearchNotes, Ticket


def grounding_check(notes: ResearchNotes, trace: list[dict]) -> dict:
    outputs = " ".join(c["output"] for c in trace)
    tools = {c["tool"] for c in trace}
    unsupported = []
    for f in notes.findings:
        source_ok = f.source in tools or f.source in outputs
        numbers_ok = all(n in outputs for n in re.findall(r"\d+(?:\.\d+)?", f.fact))
        if not (source_ok and numbers_ok):
            unsupported.append(f.fact)
    return {"grounded": not unsupported, "unsupported_claims": unsupported,
            "has_next_steps": bool(notes.recommended_next_steps)}


JUDGE_SYSTEM = (
    "You audit AI research notes written for an L2 support engineer. Judge only against the tool outputs shown; "
    "a claim is supported only if a tool output states it. Ignore your own knowledge. Score actionable 1-5: "
    "5 = a specific next step the engineer can do now, 3 = findings but a vague next step, 1 = missing or generic. "
    "Be strict."
)


def llm_judge(ticket: Ticket, notes: ResearchNotes, trace: list[dict], llm: LLM, model: str) -> JudgeVerdict:
    calls = "\n".join(f"- {c['tool']}({json.dumps(c['arguments'])}) -> {c['output']}" for c in trace) or "(none)"
    prompt = f"Ticket:\n{ticket.clean_text}\n\nTool calls and outputs:\n{calls}\n\nNotes:\n{notes.model_dump_json(indent=1)}"
    resp = llm.call(LLMRequest(task="judge", model=model, system=JUDGE_SYSTEM,
                               input=[{"role": "user", "content": prompt}], output_model=JudgeVerdict,
                               max_output_tokens=400, cacheable=True))
    return resp.parsed
