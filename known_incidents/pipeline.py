"""The triage pipeline: extract → transform → classify (Tier 0 → Tier 1) → route by policy → research (Tier 2) → load.

Cost is spent only where it buys something: rules and retrieval are free, the cheap model sees only what
rules could not decide, the strong model only what the cheap one was unsure about, and the research agent
only tickets that go to an engineer. Any failure or budget cap falls back to "assign to L2", never to a guess.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from .config import DATA_DIR, Catalogue, Settings
from .etl import group_duplicates, read_jsonl, transform
from .llm import LLM, Budget, BudgetExceeded, make_provider
from .models import Classification, Decision, ResearchNotes, Ticket, UNKNOWN
from .policy import route
from .research_agent import research
from .retriever import Retriever
from .tier0 import tier0
from .tier1 import tier1
from .tools import Toolbox


class Pipeline:
    def __init__(self, settings: Settings | None = None, provider=None, data_dir: Path = DATA_DIR):
        self.settings = settings or Settings()
        self.catalogue = Catalogue()
        self.history = Retriever(read_jsonl(data_dir / "resolved_incidents.jsonl"))
        self.toolbox = Toolbox(data_dir)
        self.budget = Budget(self.settings.max_run_usd, self.settings.max_ticket_usd)
        provider = provider or make_provider(self.settings.provider)
        if getattr(provider, "free", False):  # label offline runs honestly in traces and summaries
            self.settings.cheap_model, self.settings.strong_model = "offline-small", "offline-large"
        cache = None if getattr(provider, "free", False) else self.settings.cache_dir
        self.llm = LLM(provider, self.budget, cache)

    def run(self, tickets: list[Ticket]) -> tuple[list[Decision], dict]:
        start = time.perf_counter()
        cleaned = [transform(t) for t in tickets]
        duplicates = group_duplicates(cleaned)
        self.toolbox.open_tickets = [{"number": t.number, "store": t.entities.get("store"), "opened_at": t.opened_at,
                                      "short_description": t.short_description} for t in cleaned]
        decisions: dict[str, Decision] = {}
        for t in cleaned:
            if t.number not in duplicates:
                decisions[t.number] = self.triage(t)
        for child, parent in duplicates.items():
            p = decisions[parent]
            decisions[child] = Decision(
                number=child, action="duplicate", cluster_tag=p.cluster_tag, confidence=p.confidence,
                assignment_group=p.assignment_group, decided_by="storm grouping", duplicate_of=parent,
                reason=f"same issue as {parent}: linked as a child, no LLM call", tier_path=["dedupe"])
        ordered = [decisions[t.number] for t in tickets]
        return ordered, self.summary(ordered, cleaned, time.perf_counter() - start)

    def triage(self, ticket: Ticket) -> Decision:
        start = time.perf_counter()
        self.budget.start_ticket()
        t0, why = tier0(ticket, self.catalogue, self.history)
        if t0 and t0.confidence >= self.catalogue.min_confidence(t0.cluster_tag):
            cls, path, decided_by = t0, [f"tier0:{why} {t0.cluster_tag} {t0.confidence:.2f}"], "tier0"
        else:
            # Tier 0 had no answer, or one too weak for the action its tag would trigger: ask the LLM
            path = [f"tier0: {'tentative ' + t0.cluster_tag if t0 else 'escalate'} ({why})"]
            try:
                cls, steps = tier1(ticket, self.catalogue, self.history, self.llm, self.settings)
                path += steps
                decided_by = steps[-1].split()[0]
                if t0 and t0.cluster_tag == cls.cluster_tag:
                    # two independent signals agree: combine them (noisy-OR)
                    combined = 1 - (1 - t0.confidence) * (1 - cls.confidence)
                    cls = cls.model_copy(update={"confidence": combined})
                    path.append(f"tier0+tier1 agree {cls.cluster_tag} {combined:.2f}")
                    decided_by = "tier0+" + decided_by
            except Exception as e:  # budget cap, API error or malformed output: fail safe
                cls = t0 or Classification(cluster_tag=UNKNOWN, confidence=0.0, reasoning=str(e))
                path.append(f"tier1: skipped ({_short(e)})")
                decided_by = "tier0" if t0 else "fallback"

        r = route(ticket, cls, self.catalogue, self.toolbox, execute=self.settings.execute_runbooks)
        path.append(f"policy:{r.action}")
        d = Decision(number=ticket.number, action=r.action, cluster_tag=cls.cluster_tag,
                     confidence=round(cls.confidence, 3), assignment_group=r.assignment_group,
                     decided_by=decided_by, reason=r.reason, runbook_result=r.runbook_result, tier_path=path)

        if r.action == "assign_l2" and cls.cluster_tag == UNKNOWN and self.settings.handoff_unknown:
            d.tier_path.append("handoff: unknown_incidents cascade")  # triage.py investigates it there instead
        elif r.action == "assign_l2":
            try:
                d.research_notes, d.tool_trace = research(ticket, cls.cluster_tag, r.reason, self.catalogue,
                                                          self.toolbox, self.llm, self.settings)
                d.tier_path.append(f"tier2:research ({len(d.tool_trace)} tool calls)")
            except Exception as e:
                d.research_notes = ResearchNotes(
                    suspected_cause="unknown", findings=[], suggested_tag=cls.cluster_tag, confidence=0.0,
                    recommended_next_steps=[f"AI research was skipped ({_short(e)}); triage manually."])
                d.tier_path.append(f"tier2: skipped ({_short(e)})")

        d.cost_usd = round(self.budget.ticket_usd, 6)
        d.llm_calls = self.budget.ticket_calls
        d.latency_ms = round((time.perf_counter() - start) * 1000)
        return d

    def summary(self, decisions: list[Decision], tickets: list[Ticket], seconds: float) -> dict:
        n = len(decisions)
        actions = Counter(d.action for d in decisions)
        return {
            "tickets": n,
            "actions": dict(actions),
            "decided_by": dict(Counter(d.decided_by.split(":")[0] for d in decisions)),
            "touchless_rate": round((actions["auto_resolve"] + actions["reassign"] + actions["duplicate"]) / n, 3)
            if n else 0,
            "tickets_with_no_llm_call": sum(d.llm_calls == 0 for d in decisions),
            "pii_redactions": sum(t.redactions for t in tickets),
            "injection_flagged": [t.number for t in tickets if t.injection_flags],
            "dry_run": not self.settings.execute_runbooks,
            "provider": self.settings.provider,
            "seconds": round(seconds, 2),
            **self.budget.summary(),
        }


def _short(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:160] if not isinstance(e, BudgetExceeded) else str(e)
