"""
pipeline/retriever.py
=====================
Retrieves relevant ICA 1872 statute sections for each classified clause.

Three-layer retrieval strategy:
  1. Statute map  — guaranteed sections per clause_type (the KG layer).
                    Ensures S27 always appears for NonCompete even when
                    semantic search misses it (key finding from PRD).
  2. FAISS        — dense semantic search via InLegalBERT embeddings.
  3. BM25         — sparse lexical search (built on-the-fly if index absent).

Results are merged with Reciprocal Rank Fusion (RRF) and deduplicated.

Output adds to each clause dict:
    {
        ...classifier fields...,
        "retrieved_sections": [
            {
                "section_id":  "ICA_S27",
                "title":       "Agreement in restraint of trade, void",
                "text":        "...",
                "score":       0.91,
                "source":      "statute_map" | "faiss" | "bm25",
            },
            ...
        ]
    }

Usage:
    from pipeline.retriever import retrieve_statutes
    enriched = retrieve_statutes(classified_clauses)
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Statute map  (KG layer — guaranteed lookups per clause type)
# ---------------------------------------------------------------------------

_STATUTE_MAP: dict[str, list[str]] = {
    "NonCompete":      ["ICA_S27"],
    "ForceMajeure":    ["ICA_S56"],
    "Termination":     ["ICA_S73", "ICA_S74"],
    "Indemnification": ["ICA_S124", "ICA_S125"],
    "Arbitration":     ["ICA_S28"],
    "Confidentiality": ["ICA_S27"],
    "LiabilityCap":    ["ICA_S73", "ICA_S74"],
    "IPAssignment":    ["ICA_S10", "ICA_S11"],
    "PaymentTerms":    ["ICA_S55"],
    "GoverningLaw":    [],
    "Jurisdiction":    [],
    "Renewal":         ["ICA_S62"],
}

# RRF constant — balances rank vs. score contributions
_RRF_K = 60
_TOP_K = 5  # sections returned per clause


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_registry(kb_dir: Path) -> dict[str, dict]:
    path = kb_dir / "chunk_registry.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_faiss(kb_dir: Path):
    import faiss
    return faiss.read_index(str(kb_dir / "faiss_index" / "ica_1872.index"))


def _load_model(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModel
    warnings.filterwarnings("ignore")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl, device


def _embed_query(text: str, tok, mdl, device) -> "np.ndarray":
    import torch
    import numpy as np
    inputs = tok(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        out = mdl(**inputs)
    emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype("float32")
    norm = math.sqrt(float(emb @ emb)) + 1e-8
    return (emb / norm).reshape(1, -1)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _build_bm25(registry: dict[str, dict]):
    """Build an in-memory BM25 index from the chunk registry."""
    section_ids = list(registry.keys())
    corpus = [_tokenise(registry[sid]["text"]) for sid in section_ids]

    # BM25 parameters
    k1, b = 1.5, 0.75
    N = len(corpus)
    avg_dl = sum(len(d) for d in corpus) / max(N, 1)

    # IDF for each term
    from collections import Counter
    df: Counter = Counter()
    for doc in corpus:
        for term in set(doc):
            df[term] += 1

    def idf(term: str) -> float:
        return math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)

    def score(query_terms: list[str], doc: list[str]) -> float:
        tf_counter = Counter(doc)
        dl = len(doc)
        s = 0.0
        for term in query_terms:
            if term not in tf_counter:
                continue
            tf = tf_counter[term]
            s += idf(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        return s

    return section_ids, corpus, score


def _rrf_merge(
    ranked_lists: list[list[str]],
    scores: dict[str, float] | None = None,
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion across multiple ranked lists of section_ids.
    Returns sorted (section_id, rrf_score) pairs.
    """
    rrf: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, sid in enumerate(ranked, start=1):
            rrf[sid] = rrf.get(sid, 0.0) + 1.0 / (k + rank)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Per-clause retrieval
# ---------------------------------------------------------------------------

def _retrieve_for_clause(
    clause_text: str,
    clause_type: str,
    registry: dict[str, dict],
    section_ids: list[str],
    faiss_index,
    bm25_fn,
    bm25_corpus: list[list[str]],
    tok,
    mdl,
    device,
    top_k: int,
) -> list[dict[str, Any]]:
    import numpy as np

    ranked_lists: list[list[str]] = []
    source_map: dict[str, str] = {}

    # Layer 1: statute map (guaranteed sections)
    guaranteed = _STATUTE_MAP.get(clause_type, [])
    # Filter to sections that actually exist in our registry
    guaranteed = [s for s in guaranteed if s in registry]
    if guaranteed:
        ranked_lists.append(guaranteed)
        for sid in guaranteed:
            source_map[sid] = "statute_map"

    # Layer 2: FAISS semantic search
    try:
        query_emb = _embed_query(clause_text, tok, mdl, device)
        k_faiss = min(top_k + len(guaranteed), faiss_index.ntotal)
        scores, indices = faiss_index.search(query_emb, k_faiss)
        faiss_ranked = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(section_ids):
                sid = section_ids[idx]
                faiss_ranked.append(sid)
                if sid not in source_map:
                    source_map[sid] = "faiss"
        if faiss_ranked:
            ranked_lists.append(faiss_ranked)
    except Exception:
        pass

    # Layer 3: BM25 lexical search
    try:
        query_terms = _tokenise(clause_text)
        bm25_scores = [
            (section_ids[i], bm25_fn(query_terms, bm25_corpus[i]))
            for i in range(len(section_ids))
        ]
        bm25_ranked = [sid for sid, _ in sorted(bm25_scores, key=lambda x: x[1], reverse=True)]
        for sid in bm25_ranked[:top_k]:
            if sid not in source_map:
                source_map[sid] = "bm25"
        if bm25_ranked:
            ranked_lists.append(bm25_ranked[:top_k * 2])
    except Exception:
        pass

    # Merge with RRF
    merged = _rrf_merge(ranked_lists)

    # Build result objects
    results = []
    for sid, rrf_score in merged[:top_k]:
        if sid not in registry:
            continue
        sec = registry[sid]
        results.append({
            "section_id": sid,
            "title": sec.get("title", ""),
            "text": sec.get("text", ""),
            "score": round(rrf_score, 4),
            "source": source_map.get(sid, "unknown"),
        })

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_statutes(
    clauses: list[dict],
    kb_dir: str | Path | None = None,
    model_name: str = "law-ai/InLegalBERT",
    top_k: int = _TOP_K,
) -> list[dict]:
    """
    Enrich each classified clause with retrieved ICA 1872 statute sections.

    Args:
        clauses:    Output from classifier.classify_clauses().
        kb_dir:     Path to knowledge_base/ directory. Auto-detected if omitted.
        model_name: HuggingFace model name for FAISS query embedding.
        top_k:      Max statute sections to return per clause.

    Returns:
        Same list with added key: retrieved_sections (list of section dicts).
    """
    if kb_dir is None:
        kb_dir = Path(__file__).resolve().parent.parent / "knowledge_base"
    kb_dir = Path(kb_dir)

    print("[retriever] Loading knowledge base artifacts...")
    registry = _load_registry(kb_dir)
    section_ids = list(registry.keys())

    faiss_index = _load_faiss(kb_dir)
    print(f"[retriever] FAISS index: {faiss_index.ntotal} vectors")

    print(f"[retriever] Loading model: {model_name}")
    tok, mdl, device = _load_model(model_name)

    print("[retriever] Building BM25 index...")
    bm25_section_ids, bm25_corpus, bm25_fn = _build_bm25(registry)

    results = []
    for clause in clauses:
        text = clause.get("clause_text", "")
        ctype = clause.get("clause_type", "Unknown")

        sections = _retrieve_for_clause(
            clause_text=text,
            clause_type=ctype,
            registry=registry,
            section_ids=section_ids,
            faiss_index=faiss_index,
            bm25_fn=bm25_fn,
            bm25_corpus=bm25_corpus,
            tok=tok,
            mdl=mdl,
            device=device,
            top_k=top_k,
        )

        enriched = dict(clause)
        enriched["retrieved_sections"] = sections
        results.append(enriched)

    print(f"[retriever] Done. Retrieved statutes for {len(results)} clauses.")
    return results


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pipeline.segmenter import segment_contract
    from pipeline.classifier import classify_clauses

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.retriever <contract.pdf|contract.txt>")
        sys.exit(1)

    clauses = segment_contract(path=sys.argv[1])
    classified = classify_clauses(clauses, use_embeddings=False)
    enriched = retrieve_statutes(classified)

    for c in enriched:
        print(f"\n[{c['clause_id']}] {c['clause_type']} ({c['risk_level']})")
        print(f"  Clause: {c['clause_text'][:80]}...")
        for s in c["retrieved_sections"]:
            print(f"  -> [{s['section_id']}] {s['title'][:60]}  score={s['score']}  via={s['source']}")
