"""Data shapes shared by every stage. LLM outputs are Pydantic models, so a malformed answer fails validation."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Action = Literal["auto_resolve", "reassign", "assign_l2", "duplicate"]
UNKNOWN = "UNKNOWN"


class Ticket(BaseModel):
    number: str
    opened_at: str = ""
    caller: str = ""
    short_description: str = ""
    description: str = ""
    # filled by the transform stage
    clean_text: str = ""
    entities: dict[str, str] = Field(default_factory=dict)
    redactions: int = 0
    injection_flags: list[str] = Field(default_factory=list)


# ---- LLM structured outputs ----

class Classification(BaseModel):
    """Tier 1 answer. The model picks a cluster tag; it never picks the action."""
    cluster_tag: str = Field(description="Exactly one tag from the catalogue, or UNKNOWN")
    confidence: float = Field(description="Probability 0.0-1.0 that the tag is right. Be calibrated.")
    reasoning: str = Field(description="One sentence citing the ticket words or similar incidents used")


class Finding(BaseModel):
    fact: str = Field(description="A fact taken from a tool result")
    source: str = Field(description="Where it came from: a KB id, incident number, or tool name, e.g. 'KB0101'")


class ResearchNotes(BaseModel):
    """Tier 2 answer: AI research notes attached to the ticket for the L2 engineer."""
    suspected_cause: str = Field(description="Most likely cause, or 'unknown' if the evidence does not say")
    findings: list[Finding] = Field(description="Facts found with tools, each with its source")
    recommended_next_steps: list[str] = Field(description="Concrete steps for the L2 engineer, most useful first")
    suggested_tag: str = Field(description="Best cluster tag from the catalogue, or UNKNOWN")
    confidence: float = Field(description="Probability 0.0-1.0 that suspected_cause is right")


class JudgeVerdict(BaseModel):
    grounded: bool = Field(description="True if every finding is supported by a tool output shown")
    unsupported_claims: list[str] = Field(description="Claims no tool output supports; empty if none")
    actionable: int = Field(description="1-5: 5 means a specific next step an engineer can do now")
    rationale: str = Field(description="One or two sentences")


# ---- pipeline outputs ----

class Decision(BaseModel):
    number: str
    action: Action
    cluster_tag: str
    confidence: float
    assignment_group: str
    decided_by: str                     # e.g. "tier0:rules", "tier1:gpt-4o-mini", "policy"
    reason: str
    tier_path: list[str] = Field(default_factory=list)
    runbook_result: str | None = None
    duplicate_of: str | None = None
    research_notes: ResearchNotes | None = None
    tool_trace: list[dict] = Field(default_factory=list)
    cost_usd: float = 0.0
    llm_calls: int = 0
    latency_ms: int = 0
