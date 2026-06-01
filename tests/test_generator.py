"""
tests/test_generator.py
========================
Unit tests for pipeline/generator.py
"""
import pytest
from pipeline.generator import (
    generate_report,
    _build_explanation,
    _clause_card_html,
    _summary_html,
)

SAMPLE_CLAUSES = [
    {
        "clause_id": "test_c00",
        "clause_text": "Either party may terminate this Agreement upon 30 days written notice.",
        "heading": "1.1 Termination",
        "clause_type": "Termination",
        "risk_level": "MEDIUM",
        "confidence": 0.8,
        "method": "keyword",
        "retrieved_sections": [
            {"section_id": "ICA_S73", "title": "Compensation for breach", "text": "...", "score": 0.9, "source": "statute_map"},
        ],
    },
    {
        "clause_id": "test_c01",
        "clause_text": "The employee shall not engage in competing business for two years after termination.",
        "heading": "4.1 Non-Compete",
        "clause_type": "NonCompete",
        "risk_level": "HIGH",
        "confidence": 0.9,
        "method": "keyword",
        "retrieved_sections": [
            {"section_id": "ICA_S27", "title": "Agreement in restraint of trade, void", "text": "...", "score": 0.95, "source": "statute_map"},
        ],
    },
    {
        "clause_id": "test_c02",
        "clause_text": "This Agreement shall be governed by and construed in accordance with the laws of India.",
        "heading": "7.1 Governing Law",
        "clause_type": "GoverningLaw",
        "risk_level": "LOW",
        "confidence": 0.85,
        "method": "keyword",
        "retrieved_sections": [],
    },
]


# ---------------------------------------------------------------------------
# _build_explanation
# ---------------------------------------------------------------------------

class TestBuildExplanation:
    def test_returns_string(self):
        result = _build_explanation("Termination", [])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_statute_line_when_section_known(self):
        sections = [{"section_id": "ICA_S27"}]
        result = _build_explanation("NonCompete", sections)
        assert "ICA Section 27" in result or "S27" in result

    def test_no_placeholder_remaining(self):
        result = _build_explanation("Arbitration", [])
        assert "{statute_line}" not in result

    def test_unknown_type_returns_fallback(self):
        result = _build_explanation("Unknown", [])
        assert "manual review" in result.lower() or "could not" in result.lower()

    def test_all_types_produce_explanation(self):
        types = ["Termination","Arbitration","Confidentiality","Indemnification",
                 "NonCompete","ForceMajeure","IPAssignment","LiabilityCap",
                 "Jurisdiction","PaymentTerms","GoverningLaw","Renewal"]
        for t in types:
            result = _build_explanation(t, [])
            assert isinstance(result, str) and len(result) > 20, f"Empty explanation for {t}"


# ---------------------------------------------------------------------------
# _clause_card_html
# ---------------------------------------------------------------------------

class TestClauseCardHtml:
    def test_returns_string(self):
        result = _clause_card_html(SAMPLE_CLAUSES[0])
        assert isinstance(result, str)

    def test_contains_clause_type(self):
        result = _clause_card_html(SAMPLE_CLAUSES[0])
        assert "Termination" in result

    def test_contains_risk_level(self):
        result = _clause_card_html(SAMPLE_CLAUSES[0])
        assert "MEDIUM" in result

    def test_contains_clause_text(self):
        result = _clause_card_html(SAMPLE_CLAUSES[0])
        assert "terminate this Agreement" in result

    def test_contains_statute_pill(self):
        result = _clause_card_html(SAMPLE_CLAUSES[0])
        assert "ICA_S73" in result

    def test_high_risk_has_red_colour(self):
        result = _clause_card_html(SAMPLE_CLAUSES[1])
        assert "#ef4444" in result or "#fee2e2" in result

    def test_low_risk_has_green_colour(self):
        result = _clause_card_html(SAMPLE_CLAUSES[2])
        assert "#22c55e" in result or "#dcfce7" in result

    def test_escapes_html_in_clause_text(self):
        clause = dict(SAMPLE_CLAUSES[0])
        clause["clause_text"] = "clause with <script>alert('xss')</script>"
        result = _clause_card_html(clause)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_guaranteed_pill_has_class(self):
        result = _clause_card_html(SAMPLE_CLAUSES[0])
        assert "guaranteed" in result


# ---------------------------------------------------------------------------
# _summary_html
# ---------------------------------------------------------------------------

class TestSummaryHtml:
    def test_returns_string(self):
        result = _summary_html(SAMPLE_CLAUSES, "Test Contract")
        assert isinstance(result, str)

    def test_shows_correct_totals(self):
        result = _summary_html(SAMPLE_CLAUSES, "Test Contract")
        assert ">3<" in result or "3</div>" in result   # 3 total clauses

    def test_shows_high_count(self):
        result = _summary_html(SAMPLE_CLAUSES, "Test Contract")
        assert ">1<" in result  # 1 HIGH clause

    def test_contract_name_in_output(self):
        result = _summary_html(SAMPLE_CLAUSES, "My Agreement")
        assert "My Agreement" in result

    def test_verdict_high_when_multiple_high_risk(self):
        high_clauses = [dict(c, risk_level="HIGH") for c in SAMPLE_CLAUSES]
        result = _summary_html(high_clauses, "Test")
        assert "HIGH" in result or "strongly recommended" in result.lower()

    def test_verdict_low_when_all_low_risk(self):
        low_clauses = [dict(c, risk_level="LOW") for c in SAMPLE_CLAUSES]
        result = _summary_html(low_clauses, "Test")
        assert "low-risk" in result.lower() or "verdict low" in result.lower()


# ---------------------------------------------------------------------------
# generate_report — full output
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_returns_string(self):
        result = generate_report(SAMPLE_CLAUSES, contract_name="Test")
        assert isinstance(result, str)

    def test_is_valid_html(self):
        result = generate_report(SAMPLE_CLAUSES, contract_name="Test")
        assert result.startswith("<!DOCTYPE html>")
        assert "</html>" in result

    def test_contains_all_clause_types(self):
        result = generate_report(SAMPLE_CLAUSES, contract_name="Test")
        for clause in SAMPLE_CLAUSES:
            assert clause["clause_type"] in result

    def test_contains_all_statute_ids(self):
        result = generate_report(SAMPLE_CLAUSES, contract_name="Test")
        assert "ICA_S73" in result
        assert "ICA_S27" in result

    def test_contains_summary_section(self):
        result = generate_report(SAMPLE_CLAUSES, contract_name="Test")
        assert "Contract Summary" in result

    def test_contains_legal_disclaimer(self):
        result = generate_report(SAMPLE_CLAUSES, contract_name="Test")
        assert "not legal advice" in result.lower()

    def test_contains_export_links(self):
        result = generate_report(SAMPLE_CLAUSES, contract_name="Test")
        assert "/export/json" in result

    def test_contract_name_in_title(self):
        result = generate_report(SAMPLE_CLAUSES, contract_name="MyDeal.pdf")
        assert "MyDeal.pdf" in result

    def test_writes_file_when_output_path_given(self, tmp_path):
        out = tmp_path / "report.html"
        generate_report(SAMPLE_CLAUSES, contract_name="Test", output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_clauses_returns_html(self):
        result = generate_report([], contract_name="Empty")
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result
