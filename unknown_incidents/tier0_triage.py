"""
Tier 0 AI triage: Pinecone + local open-source embeddings, no LLM.
  python -m unknown_incidents.tier0_triage --load   # one time: create the index and upload past incidents
  python -m unknown_incidents.tier0_triage          # run the test tickets
"""
import re
import sys
from collections import Counter
from os import getenv

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from fastembed import TextEmbedding

load_dotenv()

INDEX_NAME = "incident-triage"
NAMESPACE = "resolved"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # open-source, 384 dimensions, runs locally
DIMENSION = 384

SIMILARITY_THRESHOLD = 0.80  # starting points: tune with an eval
STRONG_MATCH = 0.93
MIN_AGREEING = 2
DUPLICATE_SIM = 0.98
TOP_K = 5

ALERT_CODE = re.compile(r"\b(?!INC)[A-Z]{2,6}\d{2,5}\b")  # e.g. BATCH01; ignores INC ticket numbers

model = TextEmbedding(MODEL_NAME)  # downloaded once, then runs offline
pc = Pinecone(api_key=getenv("PINECONE_API_KEY"))

PAST_INCIDENTS = [
    {"id": "INC101", "text": "Please void ASN 00000001234, merchandise returned to vendor",
     "class": "ASN Needs Voided", "resolution": "ASN set to status 9"},
    {"id": "INC102", "text": "ASN 00000005678 needs to be voided, vendor never shipped",
     "class": "ASN Needs Voided", "resolution": "ASN set to status 9"},
    {"id": "INC103", "text": "Please void ASN 00000001234, merchandise returned to vendor",
     "class": "ASN Needs Voided", "resolution": "Duplicate of INC101"},
    {"id": "INC201", "text": "New item 55512 not showing in store system after setup",
     "class": "Item Setup", "resolution": "Re-sent item file to store"},
    {"id": "INC202", "text": "Item created yesterday is missing from the store item file",
     "class": "Item Setup", "resolution": "Re-sent item file to store"},
    {"id": "INC301", "text": "BATCH01 process hung alert on store server",
     "class": "BATCH01 Process Hung Alert", "resolution": "Restarted BATCH01 job"},
    {"id": "INC302", "text": "Alert BATCH01: batch process hung for over 30 minutes",
     "class": "BATCH01 Process Hung Alert", "resolution": "Restarted BATCH01 job"},
    {"id": "INC401", "text": "Shelf label shows wrong price for item 7781",
     "class": "Pricing", "resolution": "Pushed price update to store"},
]


def embed(texts: list[str]) -> list[list[float]]:
    vecs = np.array(list(model.embed(texts)))   
    return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).tolist()  # 

def get_index():
    return pc.Index(INDEX_NAME)


def load_incidents():
    """Create the index (if needed) and upsert past incidents. Safe to rerun: upsert by id."""
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    vectors = [
        {"id": i["id"], "values": v,
         "metadata": {"text": i["text"], "class": i["class"], "resolution": i["resolution"],
                      "alert_codes": sorted(set(ALERT_CODE.findall(i["text"])))}}
        for i, v in zip(PAST_INCIDENTS, embed([i["text"] for i in PAST_INCIDENTS]))
    ]
    get_index().upsert(vectors=vectors, namespace=NAMESPACE)
    print(f"Upserted {len(vectors)} incidents into '{INDEX_NAME}'")


def _route(cls, confidence, reason, evidence):
    return {"tier": 0, "action": "ROUTE", "class": cls, "confidence": round(confidence, 3),
            "reason": reason, "evidence": evidence}


def _escalate(reason, evidence=()):
    return {"tier": 0, "action": "ESCALATE", "class": None, "confidence": 0.0,
            "reason": reason, "evidence": list(evidence)}


def tier0_triage(text: str) -> dict:
    index = get_index()
    query_vec = embed([text])[0]

    # Rule 1: exact alert code, using a Pinecone metadata filter
    codes = sorted(set(ALERT_CODE.findall(text)))
    if codes:
        res = index.query(vector=query_vec, top_k=20, include_metadata=True, namespace=NAMESPACE,
                          filter={"alert_codes": {"$in": codes}})
        classes = {m["metadata"]["class"] for m in res["matches"]}
        if len(classes) == 1:
            return _route(classes.pop(), 1.0, f"exact alert code {codes}",
                          [{"id": m["id"], "match": "alert code"} for m in res["matches"]])

    # Rule 2: semantic search
    res = index.query(vector=query_vec, top_k=TOP_K, include_metadata=True,
                      include_values=True, namespace=NAMESPACE)
    matches = [m for m in res["matches"] if m["score"] >= SIMILARITY_THRESHOLD]
    if not matches:
        best = res["matches"][0] if res["matches"] else None
        return _escalate(f"no match above {SIMILARITY_THRESHOLD}"
                         + (f" (best {best['id']} at {best['score']:.3f})" if best else ""))

    # near-duplicate collapse: copies of one ticket count as one vote
    kept = []
    for m in matches:
        v = np.array(m["values"])
        if not any(v @ np.array(k["values"]) >= DUPLICATE_SIM for k in kept):
            kept.append(m)
    evidence = [{"id": m["id"], "class": m["metadata"]["class"], "similarity": round(m["score"], 3)}
                for m in kept]

    # Rule 3: agreement check
    votes = Counter(m["metadata"]["class"] for m in kept)
    if len(votes) > 1:
        return _escalate(f"matches disagree: {dict(votes)}", evidence)
    cls, n = votes.most_common(1)[0]
    best_score = kept[0]["score"]
    if n >= MIN_AGREEING or best_score >= STRONG_MATCH:
        return _route(cls, best_score, f"{n} distinct matching incident(s) agree", evidence)
    return _escalate(f"only {n} match at {best_score:.3f}, below the strong-match bar", evidence)


if __name__ == "__main__":
    if "--load" in sys.argv:
        load_incidents()
        sys.exit()
    for t in [
        "Alert BATCH01 triggered again at store 4532",           # alert code → route
        "Can you void ASN 00000009999? vendor didn't ship it",  # agreeing matches → route
        "Store receipt printer is jammed",                      # no match → escalate
        "Item price missing after new item setup",              # disagree → escalate
    ]:
        r = tier0_triage(t)
        print(f"\n{t}\n  → {r['action']} {r['class'] or ''} | {r['reason']}")
        for e in r["evidence"]:
            print(f"     {e}")