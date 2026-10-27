"""Routing policy: the model proposes a cluster tag, this deterministic code decides what happens.

An auto-resolve needs all of: an auto-resolve tag, high confidence, the entities its runbook needs, a read-only
precondition check that passes, and no sign of prompt injection. Anything short of that goes to an L2 engineer,
never silently closed. Runbooks run as a dry run unless execution is switched on.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Catalogue
from .models import Classification, Ticket, UNKNOWN
from .tools import Toolbox


@dataclass
class Routing:
    action: str
    assignment_group: str
    reason: str
    runbook_result: str | None = None


# ---- runbooks: (precondition, action). The precondition only reads; the action is what auto-resolve does.

def _void_asn(tb: Toolbox, e: dict) -> tuple[bool, str]:
    rec = tb.tool_lookup_asn(e["asn"])
    if "error" in rec:
        return False, rec["error"]
    if rec["status"] == "received":
        return False, f"ASN {e['asn']} was received on {rec.get('received_at', '?')}; the receipt must be reversed first"
    return True, f"set ASN {e['asn']} ({rec['vendor']}) to status 9 (voided)"


def _resend_item_file(tb: Toolbox, e: dict) -> tuple[bool, str]:
    out = tb.tool_run_store_diagnostic(e["store"], "item_file_age_hours")
    if "error" in out:
        return False, out["error"]
    if out["result"] <= 24:
        return False, f"store {e['store']} item file is only {out['result']}h old, so a re-send would not help"
    return True, f"re-sent the item file to store {e['store']} (last file was {out['result']}h old)"


def _restart_batch_job(tb: Toolbox, e: dict) -> tuple[bool, str]:
    out = tb.tool_run_store_diagnostic(e["store"], "batch_jobs")
    if "error" in out:
        return False, out["error"]
    status = out["result"].get(e["alert_code"], "not found")
    if "hung" not in status:
        return False, f"{e['alert_code']} on store {e['store']} is '{status}', not hung"
    return True, f"restarted {e['alert_code']} on store {e['store']} (was {status})"


RUNBOOKS = {"void_asn": _void_asn, "resend_item_file": _resend_item_file, "restart_batch_job": _restart_batch_job}


def route(ticket: Ticket, cls: Classification, catalogue: Catalogue, toolbox: Toolbox,
          execute: bool = False) -> Routing:
    policy = catalogue.tags.get(cls.cluster_tag)
    if cls.cluster_tag == UNKNOWN or policy is None:
        return Routing("assign_l2", catalogue.default_l2_group, "no cluster tag fits")
    group = policy["assignment_group"]
    # a low-confidence reassign stays with our own L2; other tags go to the L2 engineers of the owning team
    l2_group = catalogue.default_l2_group if policy["action"] == "reassign" else group
    l2 = lambda why: Routing("assign_l2", l2_group, why)  # noqa: E731

    if ticket.injection_flags:
        return l2("possible prompt injection in the ticket text, needs a human")

    if policy["action"] == "auto_resolve":
        if cls.confidence < catalogue.auto_resolve_min_confidence:
            return l2(f"{cls.cluster_tag} at {cls.confidence:.2f}, below the "
                      f"{catalogue.auto_resolve_min_confidence} auto-resolve bar")
        missing = [e for e in policy["required_entities"] if e not in ticket.entities]
        if missing:
            return l2(f"{cls.cluster_tag} but the ticket has no {', '.join(missing)}")
        ok, detail = RUNBOOKS[policy["runbook"]](toolbox, ticket.entities)
        if not ok:
            return l2(f"{cls.cluster_tag} runbook precondition failed: {detail}")
        result = detail if execute else f"DRY RUN: would have {detail}"
        return Routing("auto_resolve", group, f"{cls.cluster_tag} via runbook {policy['runbook']}", result)

    if policy["action"] == "reassign":
        if cls.confidence < catalogue.reassign_min_confidence:
            return l2(f"{cls.cluster_tag} at {cls.confidence:.2f}, below the reassign bar")
        return Routing("reassign", group, f"{cls.cluster_tag} belongs to {group}")

    return Routing("assign_l2", group, f"{cls.cluster_tag} needs an engineer")
