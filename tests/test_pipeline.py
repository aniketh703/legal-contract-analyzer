"""
tests/test_pipeline.py
=======================
Integration test — runs the full pipeline end-to-end on a sample contract.
"""
import pytest
from pipeline.pipeline import run_pipeline
from pipeline.document_gate import CONTRACT_GATE_MESSAGE
from tests.test_document_gate import SAMPLE_RESUME, ANIKETH_CV_SNIPPET


SAMPLE = """
1. TERMINATION
1.1 Either party may terminate this Agreement upon thirty (30) days written notice
in the event of a material breach that remains uncured after such notice.

2. NON-COMPETE
2.1 The employee shall not engage in any competing business activities for a period
of two (2) years after the termination of this Agreement.

3. INDEMNIFICATION
3.1 The Service Provider shall indemnify and hold harmless the Client from any
claims, damages, losses or expenses arising from breach or negligence.

4. FORCE MAJEURE
4.1 Neither party shall be liable for delays caused by events beyond their
reasonable control including acts of God, war, or natural disasters.

5. ARBITRATION
5.1 All disputes shall be resolved by arbitration under the Arbitration and
Conciliation Act, 1996, seated at New Delhi.

6. GOVERNING LAW
6.1 This Agreement shall be governed by and construed in accordance with the
laws of India.
"""


class TestRunPipeline:
    @pytest.fixture(scope="class")
    def pipeline_output(self):
        """Run pipeline once and share result across tests in this class."""
        html, clauses = run_pipeline(
            text=SAMPLE,
            contract_name="Integration Test",
            use_embeddings=False,  # skip model load for speed
        )
        return html, clauses

    # --- HTML output ---

    def test_returns_html_string(self, pipeline_output):
        html, _ = pipeline_output
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_html_contains_contract_name(self, pipeline_output):
        html, _ = pipeline_output
        assert "Integration Test" in html

    def test_html_contains_summary(self, pipeline_output):
        html, _ = pipeline_output
        assert "Contract Summary" in html

    def test_html_contains_statute_ids(self, pipeline_output):
        html, _ = pipeline_output
        assert "ICA_S" in html

    # --- Clause list ---

    def test_returns_clause_list(self, pipeline_output):
        _, clauses = pipeline_output
        assert isinstance(clauses, list)
        assert len(clauses) >= 4

    def test_all_clauses_have_type(self, pipeline_output):
        _, clauses = pipeline_output
        for c in clauses:
            assert c.get("clause_type"), f"Missing clause_type in {c['clause_id']}"

    def test_all_clauses_have_risk(self, pipeline_output):
        _, clauses = pipeline_output
        valid = {"HIGH", "MEDIUM", "LOW"}
        for c in clauses:
            assert c.get("risk_level") in valid

    def test_all_clauses_have_retrieved_sections(self, pipeline_output):
        _, clauses = pipeline_output
        for c in clauses:
            assert "retrieved_sections" in c
            assert isinstance(c["retrieved_sections"], list)

    def test_noncompete_is_high_risk(self, pipeline_output):
        _, clauses = pipeline_output
        nc = [c for c in clauses if c["clause_type"] == "NonCompete"]
        assert len(nc) >= 1
        assert all(c["risk_level"] == "HIGH" for c in nc)

    def test_noncompete_retrieves_s27(self, pipeline_output):
        _, clauses = pipeline_output
        nc = [c for c in clauses if c["clause_type"] == "NonCompete"]
        assert len(nc) >= 1
        section_ids = [s["section_id"] for s in nc[0]["retrieved_sections"]]
        assert "ICA_S27" in section_ids

    def test_force_majeure_retrieves_s56(self, pipeline_output):
        _, clauses = pipeline_output
        fm = [c for c in clauses if c["clause_type"] == "ForceMajeure"]
        assert len(fm) >= 1
        section_ids = [s["section_id"] for s in fm[0]["retrieved_sections"]]
        assert "ICA_S56" in section_ids

    # --- Edge cases ---

    def test_empty_text_returns_empty_clauses(self):
        html, clauses = run_pipeline(text="   ", use_embeddings=False)
        assert clauses == []

    def test_resume_blocked_before_enrichment(self):
        html, clauses = run_pipeline(
            text=SAMPLE_RESUME,
            contract_name="Profile.pdf",
            use_embeddings=False,
        )
        assert clauses == []
        assert html == CONTRACT_GATE_MESSAGE

    def test_aniketh_cv_blocked_not_classified(self):
        """Full CV text must not produce classified clause segments."""
        html, clauses = run_pipeline(
            text=ANIKETH_CV_SNIPPET,
            contract_name="Aniketh_Vustepalle_CV.pdf",
            use_embeddings=False,
        )
        assert clauses == []
        assert html == CONTRACT_GATE_MESSAGE

    def test_no_path_no_text_raises(self):
        with pytest.raises((ValueError, TypeError)):
            run_pipeline()

    def test_artifact_dir_writes_report_and_json(self, tmp_path):
        html, clauses = run_pipeline(
            text=SAMPLE,
            contract_name="Artifact Test",
            use_embeddings=False,
            artifact_dir=tmp_path,
        )
        assert (tmp_path / "report.html").exists()
        assert (tmp_path / "analysis.json").exists()
        assert "Artifact Test" in (tmp_path / "analysis.json").read_text(encoding="utf-8")
