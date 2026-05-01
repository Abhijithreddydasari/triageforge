"""Hybrid BM25 + dense retriever with Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from schema import Chunk

RRF_K = 60
BM25_CANDIDATES = 20
DENSE_CANDIDATES = 20


def _reciprocal_rank_fusion(
    rankings: List[List[int]], k: int = RRF_K
) -> List[int]:
    """Fuse multiple ranked lists into one using RRF."""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def retrieve(
    query: str,
    chunks: List[Chunk],
    bm25: BM25Okapi,
    faiss_index: faiss.IndexFlatIP,
    embed_model: SentenceTransformer,
    company: Optional[str] = None,
    k: int = 6,
) -> List[Chunk]:
    """Hybrid retrieval: BM25 top-N + dense top-N, fused with RRF.

    If company is provided, results are filtered to that sub-corpus.
    Returns top-k chunks with rrf_score set.
    """
    if company:
        company_lower = company.strip().lower()
    else:
        company_lower = None

    eligible_indices = []
    if company_lower:
        for i, c in enumerate(chunks):
            if c.company.lower() == company_lower:
                eligible_indices.append(i)
    else:
        eligible_indices = list(range(len(chunks)))

    if not eligible_indices:
        return []

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    eligible_set = set(eligible_indices)
    bm25_filtered = [
        (i, bm25_scores[i])
        for i in eligible_indices
    ]
    bm25_filtered.sort(key=lambda x: x[1], reverse=True)
    bm25_top = [idx for idx, _ in bm25_filtered[:BM25_CANDIDATES]]

    query_emb = embed_model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)

    search_k = min(len(chunks), DENSE_CANDIDATES * 5)
    scores, indices = faiss_index.search(query_emb, search_k)

    dense_top = []
    for idx in indices[0]:
        if idx in eligible_set:
            dense_top.append(int(idx))
            if len(dense_top) >= DENSE_CANDIDATES:
                break

    fused = _reciprocal_rank_fusion([bm25_top, dense_top])

    results = []
    rrf_scores: Dict[int, float] = {}
    for ranking in [bm25_top, dense_top]:
        for rank, doc_id in enumerate(ranking):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    for idx in fused[:k]:
        chunk = chunks[idx]
        chunk_copy = Chunk(
            id=chunk.id,
            text=chunk.text,
            source_path=chunk.source_path,
            company=chunk.company,
            area_path=chunk.area_path,
            rrf_score=rrf_scores.get(idx, 0.0),
        )
        results.append(chunk_copy)

    return results
