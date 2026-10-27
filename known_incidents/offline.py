"""Offline provider: a deterministic, rule-based stand-in for the LLM so the whole pipeline runs with no API key
and no cost. It speaks the same request/response protocol as a real model, including multi-turn tool calls,
so the agent loop, budget, tracing and policy code paths are all exercised. Its answers are heuristics, not AI.
"""
from __future__ import annotations

import json
import re

from .llm import LLMRequest, LLMResponse, ToolCall, Usage
from .models import Classification, Finding, ResearchNotes, UNKNOWN
from .retriever import tokens


class OfflineProvider:
    free = True

    def __init__(self):
        self._turn = 0

    def complete(self, req: LLMRequest) -> LLMResponse:
        self._turn += 1
        if req.task == "classify":
            parsed, calls = self._classify(req.hints), []
        elif req.task == "research":
            calls = self._next_tools(req.hints) if req.allow_tools else []
            parsed = None if calls else self._notes(req.hints)
        else:
            raise NotImplementedError(f"offline provider has no '{req.task}' task")
        tokens_in = (len(req.system) + len(json.dumps(req.input))) // 4
        tokens_out = len(parsed.model_dump_json()) // 4 if parsed else 20
        return LLMResponse(parsed, calls, f"offline-{self._turn}", Usage(req.model, tokens_in, 0, tokens_out))

    # ---- classification: word overlap with each tag's description and keywords, plus similar-incident votes
    def _classify(self, h: dict) -> Classification:
        words = set(tokens(h["text"])) - {"store", "stores"}  # in nearly every ticket: says nothing
        scores = {}
        for name, t in h["tags"].items():
            vocab = set(tokens(t["description"] + " " + " ".join(t["keywords"])))
            scores[name] = len(words & vocab) + sum(d["tag"] == name for d in h["similar"]) * 0.5
        best = max(scores, key=scores.get)
        if scores[best] < 2.5:
            return Classification(cluster_tag=UNKNOWN, confidence=0.85, reasoning="no tag shares enough words")
        return Classification(cluster_tag=best, confidence=min(0.6 + scores[best] / 20, 0.85),
                              reasoning=f"word overlap score {scores[best]:.1f}")

    # ---- research: plan tool calls in two rounds (independent lookups in parallel, then the KB's checks)
    def _next_tools(self, h: dict) -> list[ToolCall]:
        t, e = h["ticket"], h["ticket"]["entities"]
        done = {(c["tool"], json.dumps(c["arguments"], sort_keys=True)) for c in h["trace"]}
        plan = [("search_knowledge_base", {"query": t["clean_text"]}),
                ("search_similar_incidents", {"query": t["clean_text"]})]
        if "store" in e:
            plan.append(("list_open_tickets", {"store": e["store"]}))
        if "asn" in e:
            plan.append(("lookup_asn", {"asn": e["asn"]}))
        if "item" in e and "store" in e and "price" in t["clean_text"].lower():
            plan.append(("lookup_item_price", {"item": e["item"], "store": e["store"]}))
        kb = self._outputs(h, "search_knowledge_base")
        if kb and "store" in e:
            articles = kb[0] if isinstance(kb[0], list) else []
            for check in (articles[0]["checks"] if articles else ["ping"]):
                plan.append(("run_store_diagnostic", {"store": e["store"], "check": check}))
        todo = [(n, a) for n, a in plan if (n, json.dumps(a, sort_keys=True)) not in done]
        return [ToolCall(f"call_{self._turn}_{i}", n, a) for i, (n, a) in enumerate(todo)]

    @staticmethod
    def _outputs(h: dict, tool: str) -> list:
        return [json.loads(c["output"]) for c in h["trace"] if c["tool"] == tool]

    def _notes(self, h: dict) -> ResearchNotes:
        findings: list[Finding] = []
        causes: list[str] = []
        steps: list[str] = []
        suggested = h["guess"]
        for c in h["trace"]:
            out = json.loads(c["output"])
            if isinstance(out, dict) and "error" in out:
                findings.append(Finding(fact=out["error"], source=c["tool"]))
            elif c["tool"] == "search_knowledge_base" and out:
                a = out[0]
                findings.append(Finding(fact=f"Runbook '{a['title']}': {a['body']}", source=a["id"]))
                steps += [f"{s.strip().rstrip('.')} ({a['id']})." for s in re.split(r"(?<=\.)\s+", a["body"])
                          if s and not s.startswith(("Run ", "Look up"))]  # checks the agent already ran
            elif c["tool"] == "search_similar_incidents" and out and out[0]["similarity"] >= 0.4:
                top = out[0]
                findings.append(Finding(fact=f"Similar incident '{top['text']}' was resolved by: {top['resolution']}",
                                        source=top["number"]))
                if suggested == UNKNOWN:
                    suggested = top["tag"]
            elif c["tool"] == "run_store_diagnostic":
                findings.append(Finding(fact=f"{out['check']} on store {out['store']}: {out['result']}",
                                        source="run_store_diagnostic"))
                result = json.dumps(out["result"])
                if any(w in result for w in ("stopped", "hung", "unreachable")):
                    causes.append(f"{out['check']} on store {out['store']} is {out['result']}")
                if out["check"] == "item_file_age_hours" and out["result"] <= 24:
                    causes.append(f"the item file is fresh ({out['result']}h), so the item was probably "
                                  "left out of it: check the item setup record")
            elif c["tool"] == "list_open_tickets":
                others = [o for o in out if o["number"] != h["ticket"]["number"]]
                if others:
                    listed = ", ".join(f"{o['number']} '{o['short_description']}'" for o in others)
                    findings.append(Finding(fact=f"Other open tickets for this store: {listed}",
                                            source="list_open_tickets"))
            elif c["tool"] == "lookup_asn":
                findings.append(Finding(fact=f"ASN {out['asn']} status {out['status']}"
                                        + (f", received {out['received_at']}" if "received_at" in out else ""),
                                        source="lookup_asn"))
                if out["status"] == "received":
                    causes.append(f"ASN {out['asn']} was already received, so it cannot be voided directly")
                    steps.insert(0, "Reverse the receipt for the ASN, then void it (KB0105).")
            elif c["tool"] == "lookup_item_price":
                findings.append(Finding(fact=f"Item {out['item']}: pricing system {out['pricing_system']}, "
                                        f"register {out.get('register')}, shelf label {out.get('shelf_label')}",
                                        source="lookup_item_price"))
                if out.get("shelf_label") not in (None, out["pricing_system"]):
                    causes.append("the shelf label is out of date: the pricing system and register agree")
        if not steps and causes:
            steps = [f"Start from the finding: {causes[0]}. Check whether one of the other open tickets for this "
                     "store already covers it before debugging further."]
        if not steps:
            steps = ["No runbook matches this issue: call the store for the device model and error message.",
                     "After resolving, write a KB article so the next ticket like this can be triaged automatically."]
        return ResearchNotes(suspected_cause="; ".join(causes) or "unknown", findings=findings,
                             recommended_next_steps=steps[:5], suggested_tag=suggested,
                             confidence=0.8 if causes else 0.3)
