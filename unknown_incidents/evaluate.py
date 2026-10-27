"""
Offline evaluation harness for AI triage (Tier 0 → Tier 1 → Tier 2).
Replays labeled tickets from eval_set.jsonl through the cascade and scores it: how many tickets were
routed, how many routes were right, wrong routes, correct escalations, cost and latency, per tier.

  python -m unknown_incidents.evaluate                  # Tier 0 only: no LLM calls, free
  python -m unknown_incidents.evaluate --max-tier 1     # Tier 0 → Tier 1 (asks before calling OpenAI)
  python -m unknown_incidents.evaluate --max-tier 2     # full cascade, Tier 2 agent included
  python -m unknown_incidents.evaluate --sweep          # Tier 0: try several similarity thresholds (free)
  python -m unknown_incidents.evaluate --max-tier 2 --judge   # also score Tier 2 evidence and work notes with an LLM judge
  python -m unknown_incidents.evaluate --max-tier 2 --yes --max-wrong-route 0   # no prompt; exit 1 on any wrong route

Results are saved to eval_results/<timestamp>.json so you can compare runs before and after a change.

eval_set.jsonl: one ticket per line, {"id", "text", "expected_class"}; expected_class null means no class
fits and the right answer is ESCALATE. Keep these tickets out of the Pinecone index, or a ticket finds itself.
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from unknown_incidents import judge
from unknown_incidents import tier0_triage as tier0
from unknown_incidents import tier1_triage as tier1
from unknown_incidents import tier2_triage as tier2

HERE = Path(__file__).parent
EVAL_SET = HERE / "eval_set.jsonl"
RESULTS_DIR = HERE / "eval_results"

# USD per 1M tokens (input, output). List prices from third-party sites: check openai.com/api/pricing.
PRICES = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
}
SWEEP_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def load_eval_set(path: Path = EVAL_SET) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def outcome(expected, r: dict) -> str:
    if r["action"] == "ROUTE":
        return "correct_route" if r["class"] == expected else "wrong_route"
    return "correct_escalation" if expected is None else "missed_route"  # missed: lost automation, not an error


def cost_of(usage_trail: list[dict]) -> float | None:
    total = 0.0
    for u in usage_trail:
        if u["model"] not in PRICES:
            return None  # unknown price: report tokens only
        p_in, p_out = PRICES[u["model"]]
        total += (u["input_tokens"] * p_in + u["output_tokens"] * p_out) / 1_000_000
    return total


async def run_eval(tickets: list[dict], max_tier: int, use_judge: bool = False) -> list[dict]:
    rows = []
    for t in tickets:
        start = time.perf_counter()
        r = await tier2.triage_async(t["text"], max_tier=max_tier)
        trail = r.get("usage_trail", [])
        rows.append({
            "id": t["id"], "text": t["text"], "expected": t["expected_class"],
            "tier": r["tier"], "action": r["action"], "predicted": r["class"],
            "confidence": r["confidence"], "reason": r["reason"],
            "outcome": outcome(t["expected_class"], r),
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "llm_requests": sum(u["requests"] for u in trail),
            "input_tokens": sum(u["input_tokens"] for u in trail),
            "output_tokens": sum(u["output_tokens"] for u in trail),
            "cost_usd": cost_of(trail),
            # Tier 2 grounding checks: it must investigate and leave notes an engineer can act on
            "tools_used": r.get("tools_used", []),
            "evidence_count": len(r["evidence"]) if r["tier"] == 2 else None,
            "has_work_notes": bool(r.get("work_notes")) if r["tier"] == 2 else None,
            "evidence": r["evidence"] if r["tier"] == 2 else None,
            "work_notes": r.get("work_notes") if r["tier"] == 2 else None,
            "tool_trace": r.get("tool_trace", []),
        })
        line = f"  {t['id']}: tier {r['tier']} {r['action']:<8} {r['class'] or '-':<26} {rows[-1]['outcome']}"
        # the judge only scores real investigations: a Tier 2 run that crashed has nothing to judge
        if use_judge and r["tier"] == 2 and (r.get("tool_trace") or r["evidence"]):
            rows[-1]["judge"] = await judge.judge_investigation(t["text"], r)
            v = rows[-1]["judge"]
            line += "  judge: " + ("error" if "error" in v else "pass" if v["passed"] else "FAIL")
        print(line)
    return rows


def score(rows: list[dict]) -> dict:
    n = len(rows)
    count = lambda o: sum(r["outcome"] == o for r in rows)  # noqa: E731
    routed = count("correct_route") + count("wrong_route")
    no_class = sum(r["expected"] is None for r in rows)
    costs = [r["cost_usd"] for r in rows]
    known_cost = None if None in costs else sum(costs)
    latencies = sorted(r["latency_ms"] for r in rows)
    metrics = {
        "tickets": n,
        "auto_route_rate": routed / n if n else 0.0,                           # how much work the AI takes off L2
        "route_precision": count("correct_route") / routed if routed else None,  # of what it routed, how much was right
        "wrong_route_rate": count("wrong_route") / n if n else 0.0,            # the costly error
        "correct_escalation_rate": count("correct_escalation") / no_class if no_class else None,
        "missed_route_rate": count("missed_route") / n if n else 0.0,          # routable tickets sent to L2
        "llm_requests": sum(r["llm_requests"] for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "total_cost_usd": known_cost,
        "cost_per_correct_route_usd": (known_cost / count("correct_route")
                                       if known_cost is not None and count("correct_route") else None),
        "latency_ms_mean": round(sum(latencies) / n) if n else 0,
        "latency_ms_max": latencies[-1] if latencies else 0,
        "by_tier": {},
    }
    for tier in (0, 1, 2):
        tr = [r for r in rows if r["tier"] == tier]
        if not tr:
            continue
        t_routed = [r for r in tr if r["action"] == "ROUTE"]
        metrics["by_tier"][tier] = {
            "ended_here": len(tr),
            "routed": len(t_routed),
            "correct_routes": sum(r["outcome"] == "correct_route" for r in t_routed),
            "wrong_routes": sum(r["outcome"] == "wrong_route" for r in t_routed),
            "escalated": len(tr) - len(t_routed),
        }
    t2 = [r for r in rows if r["tier"] == 2]
    if t2:
        metrics["tier2_grounding"] = {
            "investigated_with_tools": sum(bool(r["tools_used"]) for r in t2) / len(t2),
            "cited_evidence": sum(bool(r["evidence_count"]) for r in t2) / len(t2),
            "left_work_notes": sum(bool(r["has_work_notes"]) for r in t2) / len(t2),
        }
    judged = [r["judge"] for r in rows if "judge" in r and "error" not in r["judge"]]
    if any("judge" in r for r in rows):
        claims = sum(v["claims"] for v in judged)
        judge_usage = [v["usage"] for v in judged]
        metrics["judge"] = {
            "model": judge.JUDGE_MODEL,
            "judged": len(judged),
            "judge_errors": sum("judge" in r and "error" in r["judge"] for r in rows),
            "pass_rate": sum(v["passed"] for v in judged) / len(judged) if judged else None,
            "claim_support_rate": sum(v["supported_claims"] for v in judged) / claims if claims else None,
            "fabrication_rate": sum(bool(v["fabricated_facts"]) for v in judged) / len(judged) if judged else None,
            "decision_supported_rate": sum(v["decision_supported"] for v in judged) / len(judged) if judged else None,
            "notes_actionable_mean": sum(v["notes_actionable"] for v in judged) / len(judged) if judged else None,
            "judge_cost_usd": cost_of(judge_usage),  # reported apart: it's eval cost, not triage cost
        }
    return metrics


def sweep(tickets: list[dict]) -> list[dict]:
    """Tier 0 only: how coverage and precision move with the similarity threshold. No LLM calls."""
    original = tier0.SIMILARITY_THRESHOLD
    table = []
    try:
        for threshold in SWEEP_THRESHOLDS:
            tier0.SIMILARITY_THRESHOLD = threshold  # tier0_triage reads it on every call
            rows = [{"expected": t["expected_class"], "outcome": outcome(t["expected_class"], tier0.tier0_triage(t["text"]))}
                    for t in tickets]
            routed = [r for r in rows if r["outcome"] in ("correct_route", "wrong_route")]
            correct = sum(r["outcome"] == "correct_route" for r in routed)
            table.append({"threshold": threshold, "routed": len(routed), "correct": correct,
                          "wrong": len(routed) - correct,
                          "precision": correct / len(routed) if routed else None})
    finally:
        tier0.SIMILARITY_THRESHOLD = original
    return table


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def print_report(metrics: dict, rows: list[dict]):
    print(f"\nTickets: {metrics['tickets']}")
    print(f"  auto-routed           {_pct(metrics['auto_route_rate'])}")
    print(f"  route precision       {_pct(metrics['route_precision'])}   (of routed tickets, share with the right class)")
    print(f"  wrong routes          {_pct(metrics['wrong_route_rate'])}   ← keep this at or near 0")
    print(f"  correct escalations   {_pct(metrics['correct_escalation_rate'])}   (no-class tickets sent to L2)")
    print(f"  missed routes         {_pct(metrics['missed_route_rate'])}   (routable tickets sent to L2)")
    cost = metrics["total_cost_usd"]
    print(f"  LLM requests {metrics['llm_requests']}, tokens in/out {metrics['input_tokens']}/{metrics['output_tokens']}, "
          f"cost {'n/a (model not in PRICES)' if cost is None else f'${cost:.4f}'}")
    if metrics["cost_per_correct_route_usd"] is not None:
        print(f"  cost per correct route ${metrics['cost_per_correct_route_usd']:.4f}")
    print(f"  latency mean {metrics['latency_ms_mean']} ms, max {metrics['latency_ms_max']} ms")
    print("\nBy tier where each ticket ended:")
    for tier, m in metrics["by_tier"].items():
        print(f"  Tier {tier}: {m['ended_here']} ended, {m['routed']} routed "
              f"({m['correct_routes']} right, {m['wrong_routes']} wrong), {m['escalated']} escalated")
    if "tier2_grounding" in metrics:
        g = metrics["tier2_grounding"]
        print(f"  Tier 2: used tools {_pct(g['investigated_with_tools'])}, cited evidence {_pct(g['cited_evidence'])}, "
              f"left work notes {_pct(g['left_work_notes'])}")
    if "judge" in metrics:
        j = metrics["judge"]
        cost = "n/a" if j["judge_cost_usd"] is None else f"${j['judge_cost_usd']:.4f}"
        print(f"\nLLM judge ({j['model']}) on {j['judged']} Tier 2 investigations"
              + (f", {j['judge_errors']} judge error(s)" if j["judge_errors"] else "") + f", judge cost {cost}:")
        print(f"  passed all checks     {_pct(j['pass_rate'])}")
        print(f"  claims supported      {_pct(j['claim_support_rate'])}   (evidence backed by a tool output)")
        print(f"  with fabricated facts {_pct(j['fabrication_rate'])}   ← keep this at 0")
        print(f"  decision supported    {_pct(j['decision_supported_rate'])}")
        mean = j["notes_actionable_mean"]
        print(f"  work notes actionable {'n/a' if mean is None else f'{mean:.1f} / 5'}")
        for r in rows:
            v = r.get("judge")
            if v and "error" not in v and not v["passed"]:
                print(f"  {r['id']} failed: {v['rationale']}")
                for f in dict.fromkeys(v["fabricated_facts"] + v["unsupported_claims"]):  # same fact can be in both
                    print(f"      unsupported: {f}")
    errors = [r for r in rows if r["outcome"] in ("wrong_route", "missed_route")]
    if errors:
        print("\nTo review:")
        for r in errors:
            print(f"  {r['id']} {r['outcome']}: expected {r['expected']!r}, got {r['predicted']!r} "
                  f"(tier {r['tier']}: {r['reason']})")


def config_snapshot(max_tier: int) -> dict:
    return {
        "max_tier": max_tier,
        "tier0": {"similarity_threshold": tier0.SIMILARITY_THRESHOLD, "strong_match": tier0.STRONG_MATCH,
                  "min_agreeing": tier0.MIN_AGREEING, "embedding_model": tier0.MODEL_NAME},
        "tier1": {"model": tier1.LLM_MODEL, "confidence_gate": tier1.CONFIDENCE_GATE},
        "tier2": {"model": tier2.LLM_MODEL, "confidence_gate": tier2.CONFIDENCE_GATE, "max_turns": tier2.MAX_TURNS},
        "judge_model": judge.JUDGE_MODEL,
    }


def save(payload: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline evaluation of AI triage")
    ap.add_argument("--max-tier", type=int, choices=(0, 1, 2), default=0)
    ap.add_argument("--sweep", action="store_true", help="Tier 0 similarity-threshold sweep (no LLM calls)")
    ap.add_argument("--judge", action="store_true", help="score Tier 2 investigations with an LLM judge (needs --max-tier 2)")
    ap.add_argument("--yes", action="store_true", help="don't ask before calling OpenAI")
    ap.add_argument("--max-wrong-route", type=float, default=None,
                    help="exit 1 if the wrong-route rate is above this (e.g. 0 in CI)")
    args = ap.parse_args(argv)
    tickets = load_eval_set()

    if args.sweep:
        print(f"Tier 0 threshold sweep over {len(tickets)} tickets (current threshold {tier0.SIMILARITY_THRESHOLD})")
        table = sweep(tickets)
        print(f"\n  {'threshold':>9}  {'routed':>6}  {'correct':>7}  {'wrong':>5}  precision")
        for row in table:
            print(f"  {row['threshold']:>9.2f}  {row['routed']:>6}  {row['correct']:>7}  {row['wrong']:>5}  {_pct(row['precision'])}")
        print(f"\nSaved {save({'sweep': table, 'config': config_snapshot(0)})}")
        return 0

    if args.judge and args.max_tier != 2:
        ap.error("--judge scores Tier 2 investigations: add --max-tier 2")

    if args.max_tier > 0 and not args.yes:
        models = ({tier1.LLM_MODEL} | ({tier2.LLM_MODEL} if args.max_tier == 2 else set())
                  | ({judge.JUDGE_MODEL} if args.judge else set()))
        answer = input(f"This calls OpenAI ({', '.join(sorted(models))}) for up to {len(tickets)} tickets"
                       f"{' with Tier 2 tool loops' if args.max_tier == 2 else ''}. Continue? [y/N] ")
        if answer.strip().lower() != "y":
            print("Stopped. Run without --max-tier for the free Tier 0 eval.")
            return 0

    print(f"Evaluating {len(tickets)} tickets through Tier 0{''.join(f' → {t}' for t in range(1, args.max_tier + 1))}")
    if args.judge and judge.JUDGE_MODEL == tier2.LLM_MODEL:
        print(f"Note: the judge and Tier 2 both use {judge.JUDGE_MODEL}; a model tends to favour its own answers. "
              f"Set JUDGE_MODEL to a different, ideally stronger, model.")
    rows = asyncio.run(run_eval(tickets, args.max_tier, use_judge=args.judge))
    metrics = score(rows)
    print_report(metrics, rows)
    print(f"\nSaved {save({'config': config_snapshot(args.max_tier), 'metrics': metrics, 'rows': rows})}")

    if args.max_wrong_route is not None and metrics["wrong_route_rate"] > args.max_wrong_route:
        print(f"FAIL: wrong-route rate {metrics['wrong_route_rate']:.0%} is above {args.max_wrong_route:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
