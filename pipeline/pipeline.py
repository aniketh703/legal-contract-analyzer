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

from pipeline.segmenter import segment_contract
from pipeline.classifier import classify_clauses
from pipeline.retriever import retrieve_statutes
from pipeline.generator import generate_report

# Retriever artifacts loaded once and reused across requests
_retriever_cache: dict = {}


def run_pipeline(
    path: str | Path | None = None,
    text: str | None = None,
    contract_name: str | None = None,
    use_embeddings: bool = True,
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

    clauses = segment_contract(path=path, text=text)
    if not clauses:
        return "<p>No clauses could be extracted from this document.</p>", []

    classified = classify_clauses(clauses, use_embeddings=use_embeddings)
    enriched = retrieve_statutes(classified)
    html = generate_report(enriched, contract_name=name)

    return html, enriched
