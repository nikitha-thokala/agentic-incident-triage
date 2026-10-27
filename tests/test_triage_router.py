"""The link between the two paths, tested with a fake cascade: no keys, no Pinecone, no network."""
from known_incidents.config import DATA_DIR, Settings
from known_incidents.etl import extract
from triage import run

TICKETS = DATA_DIR / "tickets.jsonl"


def fake_cascade(seen: list):
    async def investigate(texts):
        seen.extend(texts)
        return [{"tier": 2, "action": "ESCALATE", "class": None, "confidence": 0.3, "reason": "agent escalated",
                 "work_notes": "checked things", "evidence": [], "usage_trail": [
                     {"model": "gpt-4o-mini", "requests": 1, "input_tokens": 1000, "output_tokens": 100}]}
                if "kiosk" in t else
                {"tier": 1, "action": "ROUTE", "class": "Item Setup", "confidence": 0.9, "reason": "LLM classified",
                 "usage_trail": []}
                for t in texts]
    return investigate


def test_only_unknown_tickets_reach_the_cascade_and_known_ones_are_untouched():
    seen = []
    records, summary, decisions = run(extract(TICKETS), Settings(provider="offline"), fake_cascade(seen))
    by = {r["number"]: r for r in records}
    assert {r["number"] for r in records if r["path"] == "unknown"} == {"INC1010", "INC1014"}
    assert len(seen) == 2
    assert by["INC1010"]["action"] == "assign_l2" and by["INC1010"]["unknown"]["work_notes"]
    assert by["INC1014"]["action"] == "route" and by["INC1014"]["tag_or_class"] == "Item Setup"
    assert by["INC1001"]["path"] == "known" and by["INC1001"]["action"] == "auto_resolve"
    # handed-off tickets skip the known path's own research agent, so nothing is paid for twice
    assert not any(d.research_notes for d in decisions if d.number in ("INC1010", "INC1014"))
    assert summary["unknown_path"]["investigated"] == 2 and summary["unknown_path"]["cost_usd"] > 0


def test_cascade_cap_and_off_switch():
    _, summary, _ = run(extract(TICKETS), Settings(provider="offline"), fake_cascade([]), max_unknown=1)
    assert summary["unknown_path"]["investigated"] == 1 and summary["unknown_path"]["skipped_over_cap"] == 1

    records, _, decisions = run(extract(TICKETS), Settings(provider="offline"), None)
    assert all(r["path"] == "known" for r in records)
    assert all(d.research_notes for d in decisions if d.action == "assign_l2")
