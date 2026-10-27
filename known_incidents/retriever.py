"""Find similar resolved incidents with TF-IDF cosine similarity. Pure Python: no vector DB, no API, no cost.

Swap in embeddings (e.g. fastembed + Pinecone, as in a production setup) behind the same `search` method
when the history grows past a few thousand tickets or wording varies a lot.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOP = {"the", "a", "an", "is", "at", "to", "for", "of", "and", "in", "on", "it", "please", "my", "this", "that",
        "be", "has", "was", "with", "from", "by", "any", "all", "since", "after", "are", "i", "me", "not", "no",
        "t", "s", "won", "can", "don", "keeps", "every", "few"}


def tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z]+|\d+", text.lower()) if w not in STOP]
    return [w for w in words if not w.isdigit()]  # ids (ASN, item, store numbers) say nothing about the issue


class Retriever:
    def __init__(self, docs: list[dict], text_key: str = "text"):
        self.docs = docs
        tfs = [Counter(tokens(d[text_key])) for d in docs]
        df = Counter(w for tf in tfs for w in tf)
        self.idf = {w: math.log((1 + len(docs)) / (1 + n)) + 1 for w, n in df.items()}
        self.vectors = [self._vec(tf) for tf in tfs]

    def _vec(self, tf: Counter) -> dict[str, float]:
        v = {w: c * self.idf.get(w, 0.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def search(self, text: str, top_k: int = 5) -> list[tuple[float, dict]]:
        q = self._vec(Counter(tokens(text)))
        scored = [(sum(q[w] * v.get(w, 0.0) for w in q), d) for v, d in zip(self.vectors, self.docs)]
        return sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])[:top_k]
