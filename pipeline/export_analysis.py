"""
pipeline/export_analysis.py
===========================
Serialize enriched pipeline output for JSON download.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.disclaimer import LEGAL_DISCLAIMER_TEXT


def build_analysis_payload(
    clauses: list[dict],
    contract_name: str,
) -> dict[str, Any]:
    """Build a JSON-serializable analysis document."""
    return {
        "contract_name": contract_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": LEGAL_DISCLAIMER_TEXT,
        "summary": {
            "total_clauses": len(clauses),
            "high_risk": sum(1 for c in clauses if c.get("risk_level") == "HIGH"),
            "medium_risk": sum(1 for c in clauses if c.get("risk_level") == "MEDIUM"),
            "low_risk": sum(1 for c in clauses if c.get("risk_level") == "LOW"),
        },
        "clauses": clauses,
    }


def write_analysis_artifacts(
    artifact_dir: Path,
    html: str,
    clauses: list[dict],
    contract_name: str,
) -> tuple[Path, Path]:
    """Write report.html and analysis.json to artifact_dir."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report_path = artifact_dir / "report.html"
    json_path = artifact_dir / "analysis.json"

    report_path.write_text(html, encoding="utf-8")
    payload = build_analysis_payload(clauses, contract_name)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path, json_path
