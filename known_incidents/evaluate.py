"""Evaluate the pipeline against labelled tickets. Run it before and after every prompt, threshold or model change.

  python -m known_incidents.evaluate                             # offline, $0
  python -m known_incidents.evaluate --provider openai --judge   # real models, plus an LLM judge on the L2 research notes
  python -m known_incidents.evaluate --max-wrong-auto 0          # CI gate: exit 1 if anything was auto-resolved by mistake

Labels live in the ticket file ({"label": {"tag": ..., "action": ...}}, tag null = no tag fits) and are never
shown to the pipeline.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from known_incidents.config import DATA_DIR, Settings
from known_incidents.etl import extract, read_jsonl, transform
from known_incidents.judge import grounding_check, llm_judge
from known_incidents.models import UNKNOWN
from known_incidents.pipeline import Pipeline

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def score(decisions, labels: dict) -> dict:
    rows = []
    for d in decisions:
        want = labels[d.number]
        rows.append({
            "number": d.number,
            "tag_ok": d.cluster_tag == (want["tag"] or UNKNOWN),
            "action_ok": d.action == want["action"],
            # the costly error: closing a ticket that a human should have seen
            "wrong_auto_resolve": d.action == "auto_resolve" and want["action"] != "auto_resolve",
            "expected": want, "got": {"tag": d.cluster_tag, "action": d.action}, "path": d.tier_path,
        })
    n = len(rows)
    return {
        "tickets": n,
        "tag_accuracy": sum(r["tag_ok"] for r in rows) / n,
        "action_accuracy": sum(r["action_ok"] for r in rows) / n,
        "wrong_auto_resolves": sum(r["wrong_auto_resolve"] for r in rows),
        "missed_auto_resolves": sum(r["expected"]["action"] == "auto_resolve" and r["got"]["action"] != "auto_resolve"
                                    for r in rows),
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickets", type=Path, default=DATA_DIR / "tickets.jsonl")
    ap.add_argument("--provider", choices=["offline", "openai"])
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge on L2 notes (needs a real provider)")
    ap.add_argument("--max-wrong-auto", type=int, help="exit 1 if wrong auto-resolves exceed this")
    args = ap.parse_args(argv)

    settings = Settings()
    if args.provider:
        settings.provider = args.provider
    labels = {r["number"]: r["label"] for r in read_jsonl(args.tickets)}
    tickets = extract(args.tickets)
    pipeline = Pipeline(settings)
    decisions, summary = pipeline.run(tickets)
    report = score(decisions, labels)

    l2 = [d for d in decisions if d.research_notes and d.tool_trace]
    grounding = {d.number: grounding_check(d.research_notes, d.tool_trace) for d in l2}
    report["notes_grounded_rate"] = sum(g["grounded"] for g in grounding.values()) / len(l2) if l2 else None
    report["grounding"] = grounding
    if args.judge and settings.provider != "offline":
        by_number = {t.number: transform(t) for t in tickets}
        before = pipeline.budget.run_usd
        verdicts = {d.number: llm_judge(by_number[d.number], d.research_notes, d.tool_trace, pipeline.llm,
                                        settings.judge_model).model_dump() for d in l2}
        report["judge"] = {"model": settings.judge_model, "verdicts": verdicts,
                           "grounded_rate": sum(v["grounded"] for v in verdicts.values()) / len(verdicts),
                           "actionable_mean": sum(v["actionable"] for v in verdicts.values()) / len(verdicts),
                           "judge_cost_usd": round(pipeline.budget.run_usd - before, 6)}

    print(f"{'ticket':<8} {'expected':<32} {'got':<32} path")
    for r in report["rows"]:
        mark = "✓" if r["tag_ok"] and r["action_ok"] else ("✗✗" if r["wrong_auto_resolve"] else "✗")
        exp = f"{r['expected']['tag'] or UNKNOWN}/{r['expected']['action']}"
        got = f"{r['got']['tag']}/{r['got']['action']}"
        print(f"{r['number']:<8} {exp:<32} {got:<32} {mark}  {' → '.join(r['path'])}")
    print(f"\ntag accuracy {report['tag_accuracy']:.0%}, action accuracy {report['action_accuracy']:.0%}, "
          f"wrong auto-resolves {report['wrong_auto_resolves']} (keep at 0), "
          f"missed auto-resolves {report['missed_auto_resolves']}")
    if report["notes_grounded_rate"] is not None:
        print(f"L2 notes grounded (every finding cites a tool output): {report['notes_grounded_rate']:.0%}")
        for number, g in grounding.items():
            for claim in g["unsupported_claims"]:
                print(f"  {number} unsupported: {claim}")
    if "judge" in report:
        j = report["judge"]
        print(f"LLM judge ({j['model']}): grounded {j['grounded_rate']:.0%}, actionable {j['actionable_mean']:.1f}/5, "
              f"cost ${j['judge_cost_usd']:.4f}")
    print(f"pipeline cost ${summary['total_cost_usd']:.4f}, {summary['llm_calls']} LLM calls")

    out = Path(__file__).resolve().parent / "eval_results"  # kept with this package, committed as a baseline
    out.mkdir(exist_ok=True)
    path = out / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"summary": summary, **report}, indent=2, default=str))
    print(f"Saved {path}")

    if args.max_wrong_auto is not None and report["wrong_auto_resolves"] > args.max_wrong_auto:
        print(f"FAIL: {report['wrong_auto_resolves']} wrong auto-resolves")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
