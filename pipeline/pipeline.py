"""
pipeline/pipeline.py
====================
Wires all four stages into a single callable.

Usage:
    from pipeline.pipeline import run_pipeline
    html = run_pipeline("contract.pdf")
    html = run_pipeline(text="The parties agree...", contract_name="My Contract")
"""

from __future__ import annotations
from pathlib import Path

from pipeline.segmenter import segment_contract, load_contract_text
from pipeline.document_gate import assess_document, CONTRACT_GATE_MESSAGE
from pipeline.classifier import classify_clauses
from pipeline.retriever import retrieve_statutes, ARTIFACT_CACHE
from pipeline.generator import generate_report
from pipeline.export_analysis import write_analysis_artifacts

# Shared with retriever.ARTIFACT_CACHE — FAISS/BM25/registry loaded once per process
_retriever_cache: dict = ARTIFACT_CACHE


def run_pipeline(
    path: str | Path | None = None,
    text: str | None = None,
    contract_name: str | None = None,
    use_embeddings: bool = True,
    artifact_dir: str | Path | None = None,
    use_llm_summary: bool = False,
) -> tuple[str, list[dict]]:
    """
    Run the full pipeline on a contract file or raw text.

    Returns:
        (html_report, enriched_clauses)
    """
    if path is not None:
        path = Path(path)
        name = contract_name or path.name
    else:
        name = contract_name or "Contract"

    raw_text = load_contract_text(path=path, text=text)
    clauses = segment_contract(path=path, text=text)
    if not clauses:
        return "<p>No clauses could be extracted from this document.</p>", []

    is_contract, gate_message = assess_document(
        raw_text, segments=clauses, filename=name
    )
    if not is_contract:
        return gate_message or CONTRACT_GATE_MESSAGE, []

    classified = classify_clauses(clauses, use_embeddings=use_embeddings)
    enriched = retrieve_statutes(classified, cache=_retriever_cache)
    html = generate_report(enriched, contract_name=name, use_llm_summary=use_llm_summary)

    if artifact_dir is not None:
        write_analysis_artifacts(Path(artifact_dir), html, enriched, name)

    return html, enriched
