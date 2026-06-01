"""
tests/test_export_analysis.py
"""
import json
from pathlib import Path

from pipeline.export_analysis import build_analysis_payload, write_analysis_artifacts


class TestExportAnalysis:
    def test_build_payload_shape(self):
        clauses = [{
            "clause_id": "x1",
            "clause_type": "Termination",
            "risk_level": "HIGH",
            "clause_text": "Either party may terminate.",
            "retrieved_sections": [],
        }]
        payload = build_analysis_payload(clauses, "Test.pdf")
        assert payload["contract_name"] == "Test.pdf"
        assert payload["summary"]["total_clauses"] == 1
        assert payload["summary"]["high_risk"] == 1
        assert "disclaimer" in payload
        assert len(payload["clauses"]) == 1

    def test_write_artifacts(self, tmp_path):
        html = "<html><body>ok</body></html>"
        clauses = [{"clause_id": "a", "clause_type": "Arbitration", "risk_level": "MEDIUM"}]
        report_path, json_path = write_analysis_artifacts(
            tmp_path, html, clauses, "Demo"
        )
        assert report_path.exists()
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["contract_name"] == "Demo"
