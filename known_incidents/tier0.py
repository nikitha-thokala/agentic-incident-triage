"""Tier 0: classify with rules and similar past incidents. No LLM, so it costs nothing and runs in microseconds.
It answers only when its signals agree; anything unclear goes up to Tier 1."""
from __future__ import annotations

import re
from collections import Counter

from .config import Catalogue
from .models import Classification, Ticket
from .retriever import Retriever

SIMILARITY = 0.45     # a past incident counts as similar at or above this; tune with evaluate.py
STRONG_SIMILARITY = 0.75


def keyword_tags(text: str, catalogue: Catalogue) -> set[str]:
    low = text.lower()
    return {name for name, t in catalogue.tags.items()
            if any(re.search(rf"\b{re.escape(k)}\b", low) for k in t["keywords"])}


def knn_vote(text: str, retriever: Retriever) -> tuple[str | None, float, list[str]]:
    """Majority class of similar resolved incidents. Returns (tag, confidence, evidence ids)."""
    hits = [(s, d) for s, d in retriever.search(text) if s >= SIMILARITY]
    if not hits:
        return None, 0.0, []
    votes = Counter(d["tag"] for _, d in hits)
    ids = [d["number"] for _, d in hits]
    if len(votes) > 1:
        return None, 0.0, ids                      # similar incidents disagree
    tag, n = votes.most_common(1)[0]
    if n >= 2:
        return tag, 0.90, ids
    return (tag, 0.80, ids) if hits[0][0] >= STRONG_SIMILARITY else (None, 0.0, ids)


def tier0(ticket: Ticket, catalogue: Catalogue, retriever: Retriever) -> tuple[Classification | None, str]:
    """Returns (classification, reason). classification is None when Tier 0 cannot decide."""
    code = ticket.entities.get("alert_code", "").lower()
    if code:
        tags = [n for n, t in catalogue.tags.items() if code in t["keywords"]]
        if len(tags) == 1:
            return Classification(cluster_tag=tags[0], confidence=0.99,
                                  reasoning=f"exact alert code {code.upper()}"), "alert code"

    kw = keyword_tags(ticket.clean_text, catalogue)
    knn_tag, knn_conf, ids = knn_vote(ticket.clean_text, retriever)
    kw_tag = next(iter(kw)) if len(kw) == 1 else None

    if kw_tag and knn_tag == kw_tag:
        return Classification(cluster_tag=kw_tag, confidence=0.95,
                              reasoning=f"keywords and similar incidents {ids} agree"), "keywords + similar incidents"
    if len(kw) > 1 or (kw_tag and knn_tag and kw_tag != knn_tag):
        return None, f"signals disagree: keywords {sorted(kw)}, similar incidents {knn_tag}"
    if kw_tag:
        return Classification(cluster_tag=kw_tag, confidence=0.85, reasoning="catalogue keywords"), "keywords"
    if knn_tag:
        return Classification(cluster_tag=knn_tag, confidence=knn_conf,
                              reasoning=f"similar incidents {ids}"), "similar incidents"
    return None, "no rule or similar incident matched"
