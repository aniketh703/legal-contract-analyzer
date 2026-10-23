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

Two real bugs found and fixed in the process:

1. `pipeline/retriever.py` logged progress via bare `print()` (KB loading, BM25 index build, etc.), which is fine for its CLI/Flask callers but corrupts MCP's stdio transport outright, since stdout there is an exclusive JSON-RPC channel — any stray print breaks the protocol ("Failed to parse JSONRPC message from server"). The first fix (redirecting `sys.stdout` to `sys.stderr` for the duration of a pipeline call, inside `server.py`) worked for a single call but was not thread-safe: the MCP SDK dispatches sync tool calls to separate OS threads, so two concurrent `analyze_contract` calls could race on the global `sys.stdout` swap, with one thread restoring real stdout mid-pipeline on another. Fixed at the root instead — `retriever.py`'s five `print()` call sites now write to `sys.stderr` directly, so `server.py` no longer needs to touch `sys.stdout` at all. Verified by driving three `analyze_contract` calls concurrently via `asyncio.gather` through a real client.
2. `analyze_contract` collapsed every "no clauses extracted" case into one generic error message, discarding the specific, more useful message `run_pipeline`'s document gate returns when the input isn't a contract at all (e.g. a resume or LinkedIn profile). Fixed by importing `CONTRACT_GATE_MESSAGE` from `pipeline.document_gate` and applying the same gate-message check `app/main.py`'s `/analyse` route already uses, so the tool surfaces the real rejection reason instead of a generic one.
