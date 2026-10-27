"""Read-only tools the Tier 2 research agent can call. Sample data stands in for ServiceNow, the knowledge base,
store diagnostics, the ASN system and the pricing system.

Safety by construction: every tool only reads, arguments are validated, the diagnostic checks are an allowlist,
and outputs are truncated so one chatty tool can't blow up the token bill.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import DATA_DIR
from .etl import read_jsonl
from .retriever import Retriever

MAX_OUTPUT_CHARS = 1500
CHECKS = ["ping", "pos_service", "item_file_age_hours", "batch_jobs"]


def _fn(name: str, description: str, properties: dict) -> dict:
    return {"type": "function", "name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "properties": properties, "required": list(properties),
                           "additionalProperties": False}}


SPECS = [
    _fn("search_similar_incidents", "Find resolved past incidents similar to the text, with tag and resolution.",
        {"query": {"type": "string", "description": "Incident text or keywords"}}),
    _fn("search_knowledge_base", "Search runbook articles. Each result lists the diagnostic checks it calls for.",
        {"query": {"type": "string", "description": "Keywords, e.g. 'registers cannot scan'"}}),
    _fn("run_store_diagnostic", "Run one read-only diagnostic check on a store.",
        {"store": {"type": "string", "description": "4-digit store number"},
         "check": {"type": "string", "enum": CHECKS}}),
    _fn("list_open_tickets", "List other open tickets for the same store: a shared root cause shows up here.",
        {"store": {"type": "string", "description": "4-digit store number"}}),
    _fn("lookup_asn", "Get an advance shipping notice (ASN) record: status, vendor, received date.",
        {"asn": {"type": "string", "description": "11-digit ASN number"}}),
    _fn("lookup_item_price", "Get an item's price in the pricing system and at a store (register and shelf label).",
        {"item": {"type": "string"}, "store": {"type": "string", "description": "4-digit store number"}}),
]


class Toolbox:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.history = Retriever(read_jsonl(data_dir / "resolved_incidents.jsonl"))
        self.kb = read_jsonl(data_dir / "kb_articles.jsonl")
        self.kb_index = Retriever([{**a, "_text": f"{a['title']} {a['title']} {a['body']}"} for a in self.kb],
                                  text_key="_text")  # title counted twice: it says what the article is for
        self.systems = json.loads((data_dir / "systems.json").read_text())
        self.open_tickets: list[dict] = []  # set by the pipeline to the batch being triaged

    def call(self, name: str, args: dict) -> str:
        """Run a tool and return its output as JSON text. Errors are returned to the model, not raised."""
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            out: object = {"error": f"unknown tool '{name}'"}
        else:
            try:
                out = fn(**args)
            except TypeError as e:
                out = {"error": f"bad arguments: {e}"}
        text = json.dumps(out)
        return text if len(text) <= MAX_OUTPUT_CHARS else text[:MAX_OUTPUT_CHARS] + '..."(truncated)"'

    def tool_search_similar_incidents(self, query: str) -> list[dict]:
        return [{"number": d["number"], "similarity": round(s, 2), "text": d["text"], "tag": d["tag"],
                 "resolution": d["resolution"]} for s, d in self.history.search(query, top_k=3)]

    def tool_search_knowledge_base(self, query: str) -> list[dict]:
        return [{k: v for k, v in d.items() if k != "_text"}
                for s, d in self.kb_index.search(query, top_k=2) if s >= 0.25]

    def tool_run_store_diagnostic(self, store: str, check: str) -> dict:
        if check not in CHECKS:
            return {"error": f"check '{check}' is not allowed; use one of {CHECKS}"}
        s = self.systems["stores"].get(store.strip())
        if s is None:
            return {"error": f"unknown store {store}"}
        return {"store": store, "check": check, "result": s[check]}

    def tool_list_open_tickets(self, store: str) -> list[dict]:
        return [t for t in self.open_tickets if t["store"] == store.strip()]

    def tool_lookup_asn(self, asn: str) -> dict:
        rec = self.systems["asns"].get(asn.strip())
        return {"asn": asn, **rec} if rec else {"error": f"ASN {asn} not found"}

    def tool_lookup_item_price(self, item: str, store: str) -> dict:
        rec = self.systems["prices"].get(item.strip())
        if not rec:
            return {"error": f"item {item} not found in the pricing system"}
        return {"item": item, "store": store, "pricing_system": rec["pricing_system"],
                **rec["stores"].get(store.strip(), {"error": f"no store prices for {store}"})}
