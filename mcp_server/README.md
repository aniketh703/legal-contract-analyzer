# MCP server

Exposes the Legal Contract Analyzer pipeline as [MCP](https://modelcontextprotocol.io) tools, so any MCP-compatible agent (Claude Desktop, Claude Code, etc.) can call it directly instead of going through the Flask app's HTTP routes. This is an interface layer only — every tool wraps an existing, already-tested pipeline function; no classification, retrieval, or scoring logic is duplicated here.

## Tools

| Tool | Wraps | Returns |
|---|---|---|
| `analyze_contract(text, contract_name)` | `pipeline.pipeline.run_pipeline` + `pipeline.export_analysis.build_analysis_payload` | The same structured JSON `/export/json` produces: clause types, risk levels, cited ICA sections. |
| `lookup_statute_section(section_id)` | `knowledge_base/chunk_registry.json` | Full text of one Indian Contract Act, 1872 section (accepts `"73"`, `"S73"`, or `"ICA_S73"`). |
| `list_clause_types()` | `models/risk_map.json` | All 47 clause types the classifier recognizes, with default risk level. |

## Setup

```bash
python -m venv .venv-mcp
.venv-mcp/bin/pip install -r requirements.txt
```

## Run standalone (stdio)

```bash
python -m mcp_server.server
```

## Register with Claude Code / Claude Desktop

Add to `.mcp.json` (Claude Code) or `claude_desktop_config.json` (Claude Desktop):

```json
{
  "mcpServers": {
    "legal-contract-analyzer": {
      "command": "/absolute/path/to/legal-contract-analyzer/.venv-mcp/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/legal-contract-analyzer"
    }
  }
}
```

## Verified

All three tools were driven end-to-end with a real MCP client (`ClientSession` over `stdio_client`, spawning this server as a subprocess exactly as Claude Desktop/Code would) — see `example_output.txt` for the transcript.

One real bug found and fixed in the process: `pipeline/retriever.py` logs progress via bare `print()` (KB loading, BM25 index build, etc.), which is fine for its CLI/Flask callers but corrupts MCP's stdio transport outright, since stdout there is an exclusive JSON-RPC channel — any stray print breaks the protocol ("Failed to parse JSONRPC message from server"). Fixed by redirecting stdout to stderr for the duration of any pipeline call in `server.py`, rather than changing `retriever.py`'s logging behavior for every other caller.
