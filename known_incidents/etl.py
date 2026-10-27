"""Extract tickets, transform them (redact, extract entities, flag prompt injection, group duplicates), load results.

Everything here is deterministic and free: no LLM call happens before a ticket is cleaned.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

from .models import Decision, Ticket

# ---------------------------------------------------------------- extract

def extract(path: Path) -> list[Ticket]:
    """Read a ServiceNow-style export (.jsonl or .csv). Eval labels, if present, never reach the pipeline."""
    if path.suffix == ".csv":
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [Ticket(**{k: v for k, v in r.items() if k in Ticket.model_fields}) for r in rows]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- transform

PII = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")),
]
ENTITIES = {
    "store": re.compile(r"\bstore\s*#?\s*(\d{4})\b", re.I),
    "item": re.compile(r"\bitem\s*#?\s*(\d{4,6})\b", re.I),
    "asn": re.compile(r"\b(\d{11})\b"),
    "alert_code": re.compile(r"\b(?!INC)([A-Z]{2,6}\d{2,5})\b"),
}
INJECTION = [
    re.compile(r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)", re.I),
    re.compile(r"disregard (all |any )?(previous|prior|your) (instructions|rules)", re.I),
    re.compile(r"\b(system prompt|developer message|you are now)\b", re.I),
    re.compile(r"\bmark (this|the) ticket (as )?(resolved|closed)\b", re.I),
]


def redact(text: str) -> tuple[str, int]:
    count = 0
    for label, pattern in PII:
        text, n = pattern.subn(f"[{label}]", text)
        count += n
    return text, count


def transform(ticket: Ticket) -> Ticket:
    raw = f"{ticket.short_description}. {ticket.description}".strip(". ")
    clean, n = redact(" ".join(raw.split()))
    entities = {name: m.group(1) for name, rx in ENTITIES.items() if (m := rx.search(clean))}
    flags = [rx.pattern for rx in INJECTION if rx.search(clean)]
    return ticket.model_copy(update={"clean_text": clean, "entities": entities, "redactions": n,
                                     "injection_flags": flags})


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def group_duplicates(tickets: list[Ticket], min_jaccard: float = 0.8, window_hours: float = 2) -> dict[str, str]:
    """Incident storms: the same issue reported several times. Returns {duplicate number: parent number}.
    Only the parent is triaged, so a storm of 50 tickets costs one LLM call instead of 50."""
    parents: list[Ticket] = []
    links: dict[str, str] = {}
    for t in sorted(tickets, key=lambda t: t.opened_at):
        for p in parents:
            a, b = _words(t.description), _words(p.description)
            close = abs(_hours(t.opened_at) - _hours(p.opened_at)) <= window_hours
            if t.entities == p.entities and close and len(a & b) / len(a | b) >= min_jaccard:
                links[t.number] = p.number
                break
        else:
            parents.append(t)
    return links


def _hours(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts).timestamp() / 3600
    except ValueError:
        return 0.0


# ---------------------------------------------------------------- load

def load(decisions: list[Decision], out_dir: Path, summary: dict) -> None:
    """Write decisions (one JSON per line), a markdown note per L2 ticket, and the run summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "decisions.jsonl", "w") as f:
        for d in decisions:
            f.write(d.model_dump_json(exclude_none=True) + "\n")
    notes_dir = out_dir / "l2_notes"
    notes_dir.mkdir(exist_ok=True)
    for d in decisions:
        if d.research_notes:
            (notes_dir / f"{d.number}.md").write_text(render_notes(d))
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


def render_notes(d: Decision) -> str:
    """The work note an L2 engineer sees on the ticket."""
    n = d.research_notes
    lines = [f"# {d.number}: AI research notes", "",
             f"Assigned to **{d.assignment_group}** ({d.reason}).", "",
             f"**Suspected cause** (confidence {n.confidence:.0%}): {n.suspected_cause}", "",
             "## What the agent checked"]
    lines += [f"- {f.fact} _(source: {f.source})_" for f in n.findings] or ["- Nothing conclusive was found."]
    lines += ["", "## Suggested next steps"]
    lines += [f"{i}. {s}" for i, s in enumerate(n.recommended_next_steps, 1)]
    lines += ["", f"_Suggested cluster tag: {n.suggested_tag}. Generated by AI from read-only checks; verify before acting._"]
    return "\n".join(lines) + "\n"
