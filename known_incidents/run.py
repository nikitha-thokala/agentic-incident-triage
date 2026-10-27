"""Triage a batch of tickets.

  python -m known_incidents.run                                  # offline: no API key, $0
  python -m known_incidents.run --provider openai                # real models (set OPENAI_API_KEY in .env)
  python -m known_incidents.run --tickets my_export.csv --max-run-usd 0.10
"""
import argparse
from pathlib import Path

from known_incidents.config import DATA_DIR, Settings
from known_incidents.etl import extract, load
from known_incidents.pipeline import Pipeline

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickets", type=Path, default=DATA_DIR / "tickets.jsonl", help=".jsonl or .csv export")
    ap.add_argument("--provider", choices=["offline", "openai"], help="overrides LLM_PROVIDER")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--max-run-usd", type=float, help="overrides MAX_RUN_USD")
    ap.add_argument("--execute", action="store_true", help="run auto-resolve runbooks (default: dry run)")
    args = ap.parse_args(argv)

    settings = Settings()
    if args.provider:
        settings.provider = args.provider
    if args.max_run_usd is not None:
        settings.max_run_usd = args.max_run_usd
    settings.execute_runbooks = args.execute

    decisions, summary = Pipeline(settings).run(extract(args.tickets))
    load(decisions, args.out, summary)

    print(f"{'ticket':<8} {'action':<12} {'cluster tag':<18} {'conf':>4}  {'decided by':<26} {'cost':>8}  assigned to")
    for d in decisions:
        print(f"{d.number:<8} {d.action:<12} {d.cluster_tag:<18} {d.confidence:>4.2f}  {d.decided_by:<26} "
              f"${d.cost_usd:>7.5f}  {d.assignment_group}")
        detail = d.runbook_result or (f"duplicate of {d.duplicate_of}" if d.duplicate_of else d.reason)
        print(f"{'':<8} └ {detail}")
    print(f"\n{summary['tickets']} tickets: {summary['actions']}")
    print(f"touchless {summary['touchless_rate']:.0%}, {summary['tickets_with_no_llm_call']} with no LLM call, "
          f"{summary['llm_calls']} LLM calls ({summary['cache_hits']} cache hits), "
          f"total ${summary['total_cost_usd']:.4f} of ${summary['max_run_usd']:.2f} budget")
    print(f"PII redactions {summary['pii_redactions']}, injection flagged {summary['injection_flagged']}"
          f"{', runbooks in DRY RUN' if summary['dry_run'] else ''}")
    print(f"Wrote {args.out}/decisions.jsonl, {args.out}/l2_notes/*.md, {args.out}/run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
