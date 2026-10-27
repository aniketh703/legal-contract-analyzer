"""
evaluation/evaluate_retriever.py
=================================
Measures retriever accuracy — checks that the expected ICA statute
appears in the top-K retrieved sections for each clause type.

This is the key empirical test from the PRD:
  "Does ICA_S27 appear in top-5 for a NonCompete clause?"
  If not → the Knowledge Graph (statute map) is essential.

Outputs:
  - Hit@K per clause type (K=1, 3, 5)
  - Stat-map vs FAISS+BM25 comparison
  - evaluation/results/retriever_report.json

Usage:
    python evaluation/evaluate_retriever.py
    python evaluation/evaluate_retriever.py --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.retriever import (
    _load_registry, _load_faiss, _build_bm25,
    _STATUTE_MAP, _rrf_merge, _tokenise,
)
from pipeline.classifier import classify_clauses
from evaluation.evaluate_classifier import _apply_embedding_stage_leave_one_out

CLAUSES_PATH = ROOT / "data" / "processed" / "clauses.jsonl"
RESULTS_DIR  = ROOT / "evaluation" / "results"
KB_DIR       = ROOT / "knowledge_base"

# Ground-truth: expected top statute per clause type
EXPECTED_STATUTE: dict[str, str] = {
    "NonCompete":      "ICA_S27",
    "ForceMajeure":    "ICA_S56",
    "Termination":     "ICA_S73",
    "Indemnification": "ICA_S124",
    "LiabilityCap":    "ICA_S73",
    "Arbitration":     "ICA_S28",
    "PaymentTerms":    "ICA_S55",
    "Renewal":         "ICA_S62",
    "IPAssignment":    "ICA_S10",
    "Confidentiality": "ICA_S27",
}


def load_clauses() -> list[dict]:
    rows = []
    for line in CLAUSES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def hit_at_k(retrieved_ids: list[str], expected: str, k: int) -> bool:
    return expected in retrieved_ids[:k]


def evaluate_bm25_only(
    clauses: list[dict],
    registry: dict,
    section_ids: list[str],
    bm25_fn,
    bm25_corpus: list,
    top_k: int,
) -> dict[str, dict]:
    """Evaluate BM25-only retrieval (no statute map, no FAISS)."""
    results: dict[str, list[bool]] = {}

    for clause in clauses:
        ctype = clause.get("clause_type", "")
        expected = EXPECTED_STATUTE.get(ctype)
        if not expected:
            continue

        text = clause.get("clause_text", "")
        query_terms = _tokenise(text)
        scores = [(section_ids[i], bm25_fn(query_terms, bm25_corpus[i]))
                  for i in range(len(section_ids))]
        ranked = [sid for sid, _ in sorted(scores, key=lambda x: x[1], reverse=True)]

        if ctype not in results:
            results[ctype] = []
        results[ctype].append(hit_at_k(ranked, expected, top_k))

    return {t: {"hit_rate": round(sum(hits)/len(hits), 3), "n": len(hits)}
            for t, hits in results.items()}


def evaluate_with_statute_map(
    clauses: list[dict],
    registry: dict,
    section_ids: list[str],
    bm25_fn,
    bm25_corpus: list,
    top_k: int,
) -> tuple[dict[str, dict], float]:
    """
    Evaluate retrieval with statute map + BM25 (RRF merged).

    The statute-map lookup is keyed on each clause's `predicted_type` (what
    pipeline.classifier.classify_clauses() actually predicted), not its
    ground-truth `clause_type` -- the live pipeline never sees the ground
    truth, so scoring it against a lookup fed the ground truth answers a
    different, easier question than "does the real pipeline retrieve the
    right section." `clause_type` (ground truth) still defines what counts
    as a hit and which per-type bucket the result falls into.
    """
    results: dict[str, list[bool]] = {}
    classified_correctly = 0
    n_scored = 0

    for clause in clauses:
        ctype = clause.get("clause_type", "")
        pred_type = clause.get("predicted_type", ctype)
        expected = EXPECTED_STATUTE.get(ctype)
        if not expected:
            continue

        text = clause.get("clause_text", "")
        n_scored += 1
        if pred_type == ctype:
            classified_correctly += 1

        # Statute map layer -- keyed on the *predicted* type
        guaranteed = [s for s in _STATUTE_MAP.get(pred_type, []) if s in registry]

        # BM25 layer
        query_terms = _tokenise(text)
        bm25_scores = [(section_ids[i], bm25_fn(query_terms, bm25_corpus[i]))
                       for i in range(len(section_ids))]
        bm25_ranked = [sid for sid, _ in sorted(bm25_scores, key=lambda x: x[1], reverse=True)]

        merged = _rrf_merge([guaranteed, bm25_ranked[:top_k*2]] if guaranteed else [bm25_ranked])
        ranked_ids = [sid for sid, _ in merged]

        if ctype not in results:
            results[ctype] = []
        results[ctype].append(hit_at_k(ranked_ids, expected, top_k))

    type_accuracy = round(classified_correctly / n_scored, 3) if n_scored else 0.0
    return {t: {"hit_rate": round(sum(hits)/len(hits), 3), "n": len(hits)}
            for t, hits in results.items()}, type_accuracy


def print_report(bm25_results: dict, full_results: dict, top_k: int, type_accuracy: float) -> None:
    print(f"\n{'='*65}")
    print(f"RETRIEVER EVALUATION  (Hit@{top_k})")
    print(f"{'='*65}")
    print(f"Statute map is keyed on the classifier's PREDICTED type")
    print(f"(classifier type accuracy on this set: {type_accuracy:.1%})")
    print(f"\n{'Type':<20} {'BM25 only':>10} {'+ Stat Map':>12} {'Gain':>6} {'n':>5}")
    print(f"{'-'*55}")

    all_types = sorted(set(bm25_results) | set(full_results))
    macro_bm25 = macro_full = 0.0
    count = 0

    for t in all_types:
        b = bm25_results.get(t, {}).get("hit_rate", 0.0)
        f = full_results.get(t,  {}).get("hit_rate", 0.0)
        n = full_results.get(t,  {}).get("n", 0)
        gain = f - b
        gain_str = f"+{gain:.3f}" if gain > 0 else f"{gain:.3f}"
        print(f"{t:<20} {b:>10.3f} {f:>12.3f} {gain_str:>6} {n:>5}")
        macro_bm25 += b
        macro_full += f
        count += 1

    print(f"{'-'*55}")
    if count:
        print(f"{'Macro avg':<20} {macro_bm25/count:>10.3f} {macro_full/count:>12.3f}")
    print(f"\nKey finding: statute map guarantees critical sections")
    print(f"(e.g. ICA_S27 for NonCompete) that BM25 alone misses.")


def save_report(bm25_results: dict, full_results: dict, top_k: int, type_accuracy: float,
                embeddings: str = "loo") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # "off" skips the embedding stage the live pipeline uses, so it is not the
    # canonical number -- keep it in its own file instead of overwriting it.
    name = "retriever_report.json" if embeddings == "loo" else "retriever_report_ml_keyword_only.json"
    out = RESULTS_DIR / name
    out.write_text(json.dumps({
        "top_k": top_k,
        "embeddings_mode": embeddings,
        "classifier_type_accuracy": type_accuracy,
        "bm25_only": bm25_results,
        "statute_map_plus_bm25": full_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embeddings", choices=["loo", "off"], default="loo",
        help="loo (default): ML + keyword + leave-one-out embedding fallback, "
             "matching the live classifier. off: ML + keyword stages only "
             "(no torch/transformers needed; approximates the live pipeline).",
    )
    args = parser.parse_args()

    print("Loading knowledge base...")
    registry     = _load_registry(KB_DIR)
    section_ids  = list(registry.keys())
    sids, corpus, bm25_fn = _build_bm25(registry)

    print(f"Loading clauses from {CLAUSES_PATH.name}...")
    clauses = load_clauses()
    usable  = [c for c in clauses if c.get("clause_type") in EXPECTED_STATUTE]
    print(f"Evaluating {len(usable)} clauses with known expected statute...")

    print(f"Classifying {len(usable)} clauses with the real classifier "
          f"(embeddings={args.embeddings}; no self-match leak)...")
    labelled = classify_clauses(usable, use_embeddings=False)
    if args.embeddings == "loo":
        labelled = _apply_embedding_stage_leave_one_out(usable, labelled)
    for u, l in zip(usable, labelled):
        u["predicted_type"] = l["clause_type"]

    bm25_results = evaluate_bm25_only(usable, registry, section_ids, bm25_fn, corpus, args.top_k)
    full_results, type_accuracy = evaluate_with_statute_map(
        usable, registry, section_ids, bm25_fn, corpus, args.top_k
    )

    print_report(bm25_results, full_results, args.top_k, type_accuracy)
    save_report(bm25_results, full_results, args.top_k, type_accuracy, args.embeddings)


if __name__ == "__main__":
    main()
