"""Hybrid retrieval: BM25 (FTS5) + cosine over stored embeddings, fused with RRF.

Reciprocal Rank Fusion is used because it needs no score normalisation between
two incomparable scales (bm25 is negative-unbounded, cosine is [-1, 1]).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from . import vectors
from .embed import EmbeddingClient


@dataclass
class Hit:
    fact_id: int
    content: str
    category: str
    tags: str
    score: float
    bm25_rank: int | None = None
    vec_rank: int | None = None
    vec_score: float | None = None
    entities: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "fact_id": self.fact_id, "content": self.content,
            "category": self.category, "tags": self.tags,
            "score": round(self.score, 5),
            "bm25_rank": self.bm25_rank, "vec_rank": self.vec_rank,
            "vec_score": None if self.vec_score is None else round(self.vec_score, 4),
            "entities": self.entities,
        }


def rrf(rank: int | None, k: int = 60) -> float:
    return 0.0 if rank is None else 1.0 / (k + rank)


class HybridSearch:
    def __init__(self, store, embedder: EmbeddingClient | None = None,
                 model: str | None = None, rrf_k: int = 60):
        self.store = store
        self.embedder = embedder
        self.model = model or (embedder.model if embedder else "")
        self.k = rrf_k
        self._cache: dict[str, list[float]] = {}

    # --------------------------------------------------------------- vectors
    def _query_vec(self, query: str) -> list[float] | None:
        if not self.embedder or not self.model:
            return None
        if query in self._cache:
            return self._cache[query]
        try:
            vec = self.embedder.embed_one(query)
        except Exception:  # noqa: BLE001 - degrade to lexical
            return None
        self._cache[query] = vec
        return vec

    def _vector_ranks(self, qvec: Sequence[float], limit: int) -> list[tuple[int, float]]:
        qa = vectors.unpack(vectors.pack(list(qvec)))
        scored = [(fid, vectors.cosine(qa, vec))
                  for fid, vec in self.store.embeddings(self.model)]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 8, *, category: str | None = None,
               use_vectors: bool = True, log: bool = True) -> list[Hit]:
        started = time.perf_counter()
        query = (query or "").strip()
        if not query:
            return []

        pool = max(k * 6, 40)
        lex = self.store.fts_search(query, pool)                 # [(id, bm25)]
        allowed: set[int] = set()
        if category:
            allowed = {f["fact_id"] for f in self.store.list_facts(
                limit=1_000_000, category=category)}
            lex = [x for x in lex if x[0] in allowed]
        lex_rank = {fid: i for i, (fid, _) in enumerate(lex, 1)}

        vec_rank: dict[int, int] = {}
        vec_score: dict[int, float] = {}
        qvec = self._query_vec(query) if use_vectors else None
        if qvec is not None:
            for i, (fid, score) in enumerate(self._vector_ranks(qvec, pool), 1):
                if category and fid not in allowed:
                    continue
                vec_rank[fid] = i
                vec_score[fid] = score

        candidates = set(lex_rank) | set(vec_rank)
        hits: list[Hit] = []
        for fid in candidates:
            row = self.store.get(fid)
            if row is None:
                continue
            score = rrf(lex_rank.get(fid), self.k) + rrf(vec_rank.get(fid), self.k)
            hits.append(Hit(
                fact_id=fid, content=row["content"], category=row["category"] or "",
                tags=row["tags"] or "", score=score,
                bm25_rank=lex_rank.get(fid), vec_rank=vec_rank.get(fid),
                vec_score=vec_score.get(fid),
                entities=[e["name"] for e in self.store.fact_entities(fid)],
            ))
        hits.sort(key=lambda h: (-h.score, h.fact_id))
        hits = hits[:k]

        if log:
            ms = (time.perf_counter() - started) * 1000
            self.store.metric("search", "latency_ms", ms, vector=bool(qvec), k=k)
            self.store.metric("search", "hit", 1.0 if hits else 0.0, q=query[:120])
            self.store.db.commit()
        return hits
