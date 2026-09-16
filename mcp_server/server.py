"""
mcp_server/server.py
=====================
MCP server exposing the Legal Contract Analyzer pipeline as tools for
MCP-compatible clients (Claude Desktop, Claude Code, etc.).

This is an interface layer only -- every tool below wraps an existing,
already-tested pipeline function directly. No classification, retrieval,
or risk-scoring logic is duplicated here.

Tools:
    analyze_contract(text, contract_name)  -- full clause-level risk
        analysis: segment, classify, retrieve statutes, return the same
        structured JSON payload /export/json produces.
    lookup_statute_section(section_id)     -- look up one section of the
        Indian Contract Act, 1872 from the indexed knowledge base.
    list_clause_types()                    -- the 47-type taxonomy and
        each type's default risk level, straight from models/risk_map.json.

Run:
    python -m mcp_server.server
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from pipeline.export_analysis import build_analysis_payload  # noqa: E402
from pipeline.pipeline import run_pipeline  # noqa: E402

mcp = MCPServer(
    name="legal-contract-analyzer",
    version="0.1.0",
    instructions=(
        "Analyze Indian commercial contracts for clause-level legal risk. "
        "Classifies each clause by type and risk level (HIGH/MEDIUM/LOW) and "
        "grounds the analysis in the relevant sections of the Indian Contract "
        "Act, 1872."
    ),
)

_CHUNK_REGISTRY_PATH = ROOT / "knowledge_base" / "chunk_registry.json"
_RISK_MAP_PATH = ROOT / "models" / "risk_map.json"

_chunk_registry_cache: dict | None = None
_risk_map_cache: dict | None = None


def _load_chunk_registry() -> dict:
    global _chunk_registry_cache
    if _chunk_registry_cache is None:
        _chunk_registry_cache = json.loads(_CHUNK_REGISTRY_PATH.read_text(encoding="utf-8"))
    return _chunk_registry_cache


def _load_risk_map() -> dict:
    global _risk_map_cache
    if _risk_map_cache is None:
        _risk_map_cache = json.loads(_RISK_MAP_PATH.read_text(encoding="utf-8"))
    return _risk_map_cache


@contextlib.contextmanager
def _stdout_to_stderr():
    """
    MCP's stdio transport uses stdout as an exclusive JSON-RPC channel --
    any stray print() corrupts it. pipeline/retriever.py logs progress
    (KB loading, BM25 index build, etc.) via bare print() rather than the
    logging module, which is fine for its CLI/Flask callers but breaks an
    MCP tool call outright ("Failed to parse JSONRPC message from server",
    confirmed while testing this server). Redirect stdout to stderr for
    the duration of any call into the pipeline, rather than editing
    retriever.py's logging behavior for every other caller.
    """
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = original_stdout


@mcp.tool()
def analyze_contract(text: str, contract_name: str = "Contract") -> dict:
    """
    Run the full Legal Contract Analyzer pipeline on raw contract text.

    Segments the text into clauses, classifies each by type and risk
    level, retrieves the Indian Contract Act 1872 sections that govern
    each clause, and returns a structured JSON analysis -- the identical
    payload the Flask app's /export/json route produces.

    Args:
        text: The full contract text (plain text, not a file path).
        contract_name: Optional display name for the contract.
    """
    with _stdout_to_stderr():
        html, clauses = run_pipeline(
            text=text, contract_name=contract_name, use_embeddings=True
        )
    if not clauses:
        return {
            "error": "No clauses could be extracted from this text.",
            "contract_name": contract_name,
        }
    return build_analysis_payload(clauses, contract_name)


@mcp.tool()
def lookup_statute_section(section_id: str) -> dict:
    """
    Look up one section of the Indian Contract Act, 1872 by section
    number. Accepts "73", "S73", or "ICA_S73" -- all resolve to the same
    section.

    Args:
        section_id: The ICA section number to look up.
    """
    registry = _load_chunk_registry()
    key = section_id.strip().upper()
    if not key.startswith("ICA_S"):
        key = key[1:] if key.startswith("S") else key
        key = f"ICA_S{key}"
    entry = registry.get(key)
    if entry is None:
        return {
            "error": f"No ICA section found for '{section_id}'.",
            "indexed_sections": len(registry),
        }
    return entry


@mcp.tool()
def list_clause_types() -> dict:
    """
    List every contract clause type the classifier recognizes, with its
    default risk level (HIGH / MEDIUM / LOW), from models/risk_map.json.
    """
    risk_map = _load_risk_map()
    return {"count": len(risk_map), "clause_types": risk_map}


if __name__ == "__main__":
    mcp.run(transport="stdio")
