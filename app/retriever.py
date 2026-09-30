"""Hybrid retrieval over the supplied corpus.

sparse : BM25 over stemmed content terms
dense  : sentence-transformers embeddings (bge-small, CPU), cosine similarity
fusion : Reciprocal Rank Fusion (RRF, k=60)
rerank : cheap lexical-coverage rerank (no extra model) + near-duplicate removal
"""
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Chunk
from .text import content_terms, jaccard

log = logging.getLogger(__name__)
RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float
    coverage: float
    sources: dict = field(default_factory=dict)  # {"sparse": rank, "dense": rank}

    def brief(self):
        return {"id": self.chunk.chunk_id, "score": round(self.score, 4), "coverage": round(self.coverage, 2)}


class DenseIndex:
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, chunks, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")
        self.emb = self.model.encode([c.index_text for c in chunks], normalize_embeddings=True,
                                     batch_size=32, show_progress_bar=False)

    def search(self, query, n):
        q = self.model.encode([self.QUERY_PREFIX + query], normalize_embeddings=True)[0]
        sims = self.emb @ q
        order = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in order]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], mode: str = "hybrid", embed_model: str = "BAAI/bge-small-en-v1.5"):
        self.chunks = chunks
        self.terms = [content_terms(c.index_text) for c in chunks]
        self.bm25 = BM25Okapi(self.terms)
        n = len(chunks)
        df = Counter(t for ts in self.terms for t in set(ts))
        self.idf = {t: math.log((n + 1) / (d + 0.5)) for t, d in df.items()}
        self.vocab = set(self.idf)
        self.mode = mode
        self.dense = None
        if mode in ("hybrid", "dense"):
            try:
                t0 = time.time()
                self.dense = DenseIndex(chunks, embed_model)
                log.info("dense index built in %.1fs", time.time() - t0)
            except Exception as e:  # model not downloadable, no torch, ...
                log.warning("Dense retrieval unavailable (%s); falling back to sparse-only.", e)
                self.mode = "sparse"
        self._cache = {}

    # ---- helpers used by controller / decomposer ----
    def informative(self, term: str) -> bool:
        return term in self.vocab or term.isdigit()

    def coverage(self, query: str, chunk: Chunk) -> float:
        q = [t for t in content_terms(query) if not t.isdigit()]
        if not q:
            return 0.0
        cterms = set(content_terms(chunk.index_text))
        return sum(1 for t in q if t in cterms) / len(q)

    # ---- main search ----
    def _sparse(self, query, n):
        q = content_terms(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        order = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def search(self, query: str, k: int = 5, prefer_docs: frozenset = frozenset()) -> list[Hit]:
        """prefer_docs: documents already used in this session; they get a small boost so a late
        detail refines the current topic instead of drifting to a look-alike document."""
        key = (query.lower().strip(), k, prefer_docs)
        if key in self._cache:
            return self._cache[key]
        pool = max(20, 4 * k)
        lists = {}
        if self.mode in ("hybrid", "sparse"):
            lists["sparse"] = self._sparse(query, pool)
        if self.mode in ("hybrid", "dense") and self.dense:
            lists["dense"] = self.dense.search(query, pool)
        fused, src = {}, {}
        for name, lst in lists.items():
            for rank, (i, _) in enumerate(lst, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
                src.setdefault(i, {})[name] = rank
        if not fused:
            self._cache[key] = []
            return []
        top = max(fused.values())
        hits = []
        for i, s in fused.items():
            cov = self.coverage(query, self.chunks[i])
            boost = 0.03 if self.chunks[i].doc_id in prefer_docs else 0.0
            hits.append(Hit(self.chunks[i], 0.7 * (s / top) + 0.3 * cov + boost, cov, src[i]))
        hits.sort(key=lambda h: -h.score)
        hits = dedupe(hits, self.terms, self.chunks)[:k]
        self._cache[key] = hits
        return hits


def dedupe(hits: list[Hit], all_terms, chunks) -> list[Hit]:
    idx = {c.chunk_id: i for i, c in enumerate(chunks)}
    kept, seen = [], set()
    for h in hits:
        if h.chunk.chunk_id in seen:
            continue
        t = all_terms[idx[h.chunk.chunk_id]]
        if any(jaccard(t, all_terms[idx[k.chunk.chunk_id]]) > 0.85 for k in kept):
            continue
        seen.add(h.chunk.chunk_id)
        kept.append(h)
    return kept


def rrf_merge(result_lists: list[list[Hit]], cap: int, per_list_min: int = 2) -> list[Hit]:
    """Fuse evidence across sub-queries: guarantee each sub-query's best hits, then RRF order."""
    chosen, seen = [], set()
    for lst in result_lists:
        for h in lst[:per_list_min]:
            if h.chunk.chunk_id not in seen:
                seen.add(h.chunk.chunk_id)
                chosen.append(h)
    n_guaranteed = len(chosen)
    scores = {}
    for lst in result_lists:
        for r, h in enumerate(lst, start=1):
            scores[h.chunk.chunk_id] = scores.get(h.chunk.chunk_id, 0) + 1 / (RRF_K + r)
            if h.chunk.chunk_id not in seen:
                seen.add(h.chunk.chunk_id)
                chosen.append(h)
    rest = sorted(chosen[n_guaranteed:], key=lambda h: -scores[h.chunk.chunk_id])
    return (chosen[:n_guaranteed] + rest)[:cap]
