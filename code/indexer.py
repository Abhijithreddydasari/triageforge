"""One-time corpus indexing: chunk markdown files, build BM25 + FAISS indices.

Indices are cached to data/index/ keyed by a SHA-256 content hash of the
corpus file list + sizes. Rebuilds only when the corpus changes.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from schema import Chunk
from taxonomy import build_taxonomy, save_taxonomy

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 80

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _corpus_hash(data_dir: Path) -> str:
    """SHA-256 over sorted (path, size) pairs — detects any corpus change."""
    h = hashlib.sha256()
    for f in sorted(data_dir.rglob("*.md")):
        h.update(f"{f.relative_to(data_dir)}:{f.stat().st_size}".encode())
    return h.hexdigest()[:16]


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text)


def _heading_aware_chunk(
    text: str, source_path: str, company: str, area_path: str
) -> List[Chunk]:
    """Split on ## headings, then subdivide large sections."""
    sections = re.split(r"\n(?=##\s)", text)
    chunks: List[Chunk] = []

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        words = section.split()
        if len(words) <= CHUNK_TARGET_TOKENS:
            chunks.append(
                Chunk(
                    id=f"{source_path}::{i}",
                    text=section,
                    source_path=source_path,
                    company=company,
                    area_path=area_path,
                )
            )
        else:
            start = 0
            sub_idx = 0
            while start < len(words):
                end = min(start + CHUNK_TARGET_TOKENS, len(words))
                chunk_text = " ".join(words[start:end])
                chunks.append(
                    Chunk(
                        id=f"{source_path}::{i}.{sub_idx}",
                        text=chunk_text,
                        source_path=source_path,
                        company=company,
                        area_path=area_path,
                    )
                )
                start = end - CHUNK_OVERLAP_TOKENS if end < len(words) else end
                sub_idx += 1

    return chunks


def _load_corpus(data_dir: Path) -> List[Chunk]:
    all_chunks: List[Chunk] = []

    for company_dir in sorted(data_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        company = company_dir.name

        for md_file in sorted(company_dir.rglob("*.md")):
            rel = md_file.relative_to(company_dir)
            area_path = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else "_root"
            source_path = str(md_file.relative_to(data_dir.parent))

            raw = md_file.read_text(encoding="utf-8", errors="replace")
            text = _strip_frontmatter(raw)
            if len(text.strip()) < 20:
                continue

            chunks = _heading_aware_chunk(text, source_path, company, area_path)
            all_chunks.extend(chunks)

    return all_chunks


def build_index(
    data_dir: Path, index_dir: Path, force: bool = False
) -> Tuple[List[Chunk], BM25Okapi, faiss.IndexFlatIP, SentenceTransformer]:
    """Build or load cached BM25 + FAISS indices.

    Returns (chunks, bm25, faiss_index, embed_model).
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    corpus_hash = _corpus_hash(data_dir)
    hash_file = index_dir / "corpus_hash.txt"

    cached = (
        not force
        and hash_file.exists()
        and hash_file.read_text().strip() == corpus_hash
        and (index_dir / "chunks.pkl").exists()
        and (index_dir / "bm25.pkl").exists()
        and (index_dir / "faiss.index").exists()
    )

    embed_model = SentenceTransformer(EMBED_MODEL)

    if cached:
        print(f"[indexer] Cache hit ({corpus_hash}), loading indices...")
        with open(index_dir / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open(index_dir / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        taxonomy = build_taxonomy(data_dir)
        save_taxonomy(taxonomy, index_dir / "taxonomy.json")
        return chunks, bm25, faiss_index, embed_model

    print(f"[indexer] Building index for corpus hash {corpus_hash}...")
    chunks = _load_corpus(data_dir)
    print(f"[indexer] {len(chunks)} chunks from {data_dir}")

    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    print("[indexer] Encoding embeddings (this may take a minute)...")
    texts = [c.text for c in chunks]
    embeddings = embed_model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True, batch_size=64
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))
    hash_file.write_text(corpus_hash)

    taxonomy = build_taxonomy(data_dir)
    save_taxonomy(taxonomy, index_dir / "taxonomy.json")

    print(f"[indexer] Done. {len(chunks)} chunks indexed, saved to {index_dir}")
    return chunks, bm25, faiss_index, embed_model
