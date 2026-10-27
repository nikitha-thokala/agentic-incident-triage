"""
Tier 2 MCP tool server: read-only tools the Tier 2 agent uses to investigate an incident.
The agent starts this file itself over stdio; you don't run it by hand.

  Tools: search_similar_incidents, get_incident, search_knowledge_base, run_store_diagnostic
  Sample data stands in for ServiceNow, the knowledge base and store diagnostics.
"""
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer(
    name="incident-tools",
    instructions="Read-only tools for investigating store IT incidents. No tool changes anything.",
)
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False)

# --- sample data (in production: ServiceNow API, KB search, store diagnostics service) ---
INCIDENTS = {
    "INC9001": {"number": "INC9001", "store": "2111", "opened_at": "2026-10-23 07:12",
                "short_description": "Registers can't scan any items since this morning",
                "assignment_group": "Store Systems - Item and Receiving", "state": "New",
                "work_notes": []},
    "INC9002": {"number": "INC9002", "store": "4532", "opened_at": "2026-10-23 08:40",
                "short_description": "Item 55512 price missing after new item setup",
                "assignment_group": "Store Systems - Item and Receiving", "state": "New",
                "work_notes": ["Store manager says the item was created yesterday."]},
}
KB_ARTICLES = [
    {"id": "KB0101", "title": "POS service down: registers cannot scan",
     "body": "Check pos_service. If stopped, L2 restarts it from the store server console."},
    {"id": "KB0102", "title": "New item missing at store or price missing after setup",
     "body": "Check item_file_age. If older than 24h, the item file did not reach the store: re-send it."},
    {"id": "KB0103", "title": "Shelf label shows wrong price",
     "body": "Confirm the price in the pricing system, then push a price update to the store."},
]
STORE_DIAGNOSTICS = {
    "2111": {"ping": "reachable", "pos_service": "stopped since 06:58", "item_file_age": "3h"},
    "4532": {"ping": "reachable", "pos_service": "running", "item_file_age": "31h"},
}
CHECKS = ("ping", "pos_service", "item_file_age")  # allowlist: the agent can't run anything else


@server.tool(annotations=READ_ONLY)
def search_similar_incidents(query: str, top_k: int = 5) -> list[dict]:
    """Find resolved past incidents similar to the query, with their class and resolution.

    Args:
        query: The incident text or keywords, e.g. "item price missing after setup".
        top_k: How many past incidents to return (1-10).
    """
    from unknown_incidents.tier0_triage import NAMESPACE, embed, get_index  # loaded on first use: keeps startup fast

    res = get_index().query(vector=embed([query])[0], top_k=max(1, min(top_k, 10)),
                            include_metadata=True, namespace=NAMESPACE)
    return [{"id": m["id"], "similarity": round(m["score"], 3), "text": m["metadata"]["text"],
             "class": m["metadata"]["class"], "resolution": m["metadata"]["resolution"]}
            for m in res["matches"]]


@server.tool(annotations=READ_ONLY)
def get_incident(incident_number: str) -> dict:
    """Get the ServiceNow record for an incident: store, opened time, group, state, work notes.

    Args:
        incident_number: The incident number, e.g. "INC9001".
    """
    return INCIDENTS.get(incident_number.strip().upper(), {"error": f"{incident_number} not found"})


@server.tool(annotations=READ_ONLY)
def search_knowledge_base(query: str) -> list[dict]:
    """Search knowledge-base articles (runbooks) by keywords.

    Args:
        query: Keywords, e.g. "registers cannot scan".
    """
    words = {w for w in query.lower().split() if len(w) > 2}
    scored = [(len(words & set(f"{a['title']} {a['body']}".lower().split())), a) for a in KB_ARTICLES]
    return [a for score, a in sorted(scored, key=lambda s: -s[0]) if score > 0][:3]


@server.tool(annotations=READ_ONLY)
def run_store_diagnostic(store_number: str, check: Literal["ping", "pos_service", "item_file_age"]) -> dict:
    """Run one allowlisted, read-only diagnostic check on a store.

    Args:
        store_number: The 4-digit store number, e.g. "2111".
        check: ping (network), pos_service (register service status) or item_file_age (hours since the last item file).
    """
    if check not in CHECKS:
        return {"error": f"check '{check}' is not allowed; use one of {list(CHECKS)}"}
    store = STORE_DIAGNOSTICS.get(store_number.strip())
    if store is None:
        return {"error": f"unknown store {store_number}"}
    return {"store": store_number, "check": check, "result": store[check]}


if __name__ == "__main__":
    server.run("stdio")
