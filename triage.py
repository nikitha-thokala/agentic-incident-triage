"""Triage a batch of tickets end to end: known incidents first, unknown incidents handed to the investigation cascade.

  python triage.py                       # known path offline ($0); unknown path runs when its keys are set
  python triage.py --unknown off         # known path only, its own research agent writes the L2 notes
  python triage.py --provider openai --max-unknown 10

Known path (known_incidents/): rules, retrieval and a cheap LLM map the ticket to a cluster tag; policy auto-resolves,
reassigns, or assigns to L2. Unknown path (unknown_incidents/): a ticket no tag fits goes through the Tier 0 → 1 → 2
cascade (Pinecone retrieval, LLM classifier, MCP tool agent), which routes it or escalates with work notes.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from known_incidents.config import DATA_DIR, Settings
from known_incidents.etl import extract, load, transform
from known_incidents.llm import Usage
from known_incidents.models import UNKNOWN
from known_incidents.pipeline import Pipeline

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

UNKNOWN_PATH_KEYS = ("OPENAI_API_KEY", "PINECONE_API_KEY")


def unknown_path_ready() -> tuple[bool, str]:
    missing = [k for k in UNKNOWN_PATH_KEYS if not os.getenv(k)]
    if missing:
        return False, f"missing {', '.join(missing)}"
    try:
        import agents, fastembed, pinecone  # noqa: F401, E401
    except ImportError as e:
        return False, f"{e.name} not installed (pip install -r requirements-unknown.txt)"
    return True, "ready"


async def investigate_unknown(texts: list[str]) -> list[dict]:
    """Run the cascade on each ticket in ONE event loop (the OpenAI client's connections belong to that loop)."""
    from unknown_incidents.tier2_triage import triage_async  # heavy imports: only when this path runs
    return [await triage_async(t) for t in texts]


def usage_cost(trail: list[dict]) -> float:
    return sum(Usage(u["model"], u["input_tokens"], 0, u["output_tokens"]).cost_usd for u in trail)


def run(tickets, settings: Settings, investigate=None, max_unknown: int = 20) -> tuple[list[dict], dict, list]:
    """investigate: async fn(list[str]) -> list[dict] for unknown tickets, or None to keep them on the known path."""
    settings.handoff_unknown = investigate is not None
    decisions, summary = Pipeline(settings).run(tickets)
    clean = {t.number: transform(t).clean_text for t in tickets}  # PII already redacted before the cascade sees it
    records = [{"number": d.number, "path": "known", "action": d.action, "tag_or_class": d.cluster_tag,
                "assigned_to": d.assignment_group, "reason": d.reason, "known": d.model_dump(exclude_none=True)}
               for d in decisions]

    unknown = [r for r in records if r["action"] == "assign_l2" and r["tag_or_class"] == UNKNOWN][:max_unknown]
    results = asyncio.run(investigate([clean[r["number"]] for r in unknown])) if investigate and unknown else []
    for rec, res in zip(unknown, results):
        rec["path"] = "unknown"
        rec["unknown"] = res
        if res["action"] == "ROUTE":
            rec.update(action="route", tag_or_class=res["class"], reason=res["reason"])
        else:
            rec.update(action="assign_l2", reason=res["reason"])

    summary["unknown_path"] = {
        "investigated": len(results),
        "routed": sum(r["action"] == "ROUTE" for r in results),
        "escalated_with_notes": sum(r["action"] == "ESCALATE" for r in results),
        "cost_usd": round(sum(usage_cost(r.get("usage_trail", [])) for r in results), 6),
        "skipped_over_cap": max(0, sum(d.action == "assign_l2" and d.cluster_tag == UNKNOWN
                                       for d in decisions) - len(unknown)) if investigate else 0,
    }
    return records, summary, decisions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickets", type=Path, default=DATA_DIR / "tickets.jsonl")
    ap.add_argument("--provider", choices=["offline", "openai"], help="known path LLM; overrides LLM_PROVIDER")
    ap.add_argument("--unknown", choices=["auto", "on", "off"], default="auto",
                    help="auto: use the cascade when its keys and packages are present")
    ap.add_argument("--max-unknown", type=int, default=20, help="cap on tickets sent to the cascade per run")
    ap.add_argument("--out", type=Path, default=Path("output"))
    args = ap.parse_args(argv)

    settings = Settings()
    if args.provider:
        settings.provider = args.provider
    ready, why = unknown_path_ready()
    if args.unknown == "on" and not ready:
        ap.error(f"unknown path not available: {why}")
    use_unknown = args.unknown == "on" or (args.unknown == "auto" and ready)

    records, summary, decisions = run(extract(args.tickets), settings,
                                      investigate_unknown if use_unknown else None, args.max_unknown)
    load(decisions, args.out, summary)
    with open(args.out / "triage.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"{'ticket':<8} {'path':<8} {'action':<12} {'tag / class':<26} assigned to / reason")
    for r in records:
        print(f"{r['number']:<8} {r['path']:<8} {r['action']:<12} {r['tag_or_class']:<26} "
              f"{r['assigned_to'] if r['path'] == 'known' else r['reason']}")
    u = summary["unknown_path"]
    print(f"\nknown path: {summary['actions']}, ${summary['total_cost_usd']:.4f}")
    off = f"off ({why if not ready else 'turned off'}), " if not use_unknown else ""
    print(f"unknown path: {off}"
          f"{u['investigated']} investigated, {u['routed']} routed, {u['escalated_with_notes']} escalated, "
          f"${u['cost_usd']:.4f}")
    print(f"Wrote {args.out}/triage.jsonl, decisions.jsonl, l2_notes/, run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
