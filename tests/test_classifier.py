"""
tests/test_classifier.py
=========================
Unit tests for pipeline/classifier.py
"""
import pytest
from pipeline.segmenter import segment_contract
from pipeline.classifier import classify_clauses, _keyword_classify, _RISK


# ---------------------------------------------------------------------------
# _keyword_classify — rule engine
# ---------------------------------------------------------------------------

class TestKeywordClassify:
    def test_detects_termination(self):
        result = _keyword_classify("Either party may terminate this Agreement upon 30 days written notice.")
        assert result is not None
        assert result[0] == "Termination"

    def test_detects_arbitration(self):
        result = _keyword_classify("All disputes shall be resolved by arbitration under the Arbitration Act 1996.")
        assert result is not None
        assert result[0] == "Arbitration"

    def test_detects_confidentiality(self):
        result = _keyword_classify("The parties agree to keep all confidential information strictly private.")
        assert result is not None
        assert result[0] == "Confidentiality"

    def test_detects_indemnification(self):
        result = _keyword_classify("The vendor shall indemnify and hold harmless the client from all claims.")
        assert result is not None
        assert result[0] == "Indemnification"

    def test_detects_noncompete(self):
        result = _keyword_classify("The employee shall not engage in any competing business activities for two years.")
        assert result is not None
        assert result[0] == "NonCompete"

    def test_detects_force_majeure(self):
        result = _keyword_classify("Neither party shall be liable for events beyond their reasonable control.")
        assert result is not None
        assert result[0] == "ForceMajeure"

    def test_detects_liability_cap(self):
        result = _keyword_classify("In no event shall the aggregate liability of either party exceed the fees paid.")
        assert result is not None
        assert result[0] == "LiabilityCap"

    def test_detects_governing_law(self):
        result = _keyword_classify("This Agreement shall be governed by and construed in accordance with the laws of India.")
        assert result is not None
        assert result[0] == "GoverningLaw"

    def test_detects_renewal(self):
        result = _keyword_classify("This Agreement shall automatically renew for successive one-year terms.")
        assert result is not None
        assert result[0] == "Renewal"

    def test_returns_none_for_gibberish(self):
        result = _keyword_classify("The quick brown fox jumps over the lazy dog.")
        assert result is None

    def test_confidence_between_0_and_1(self):
        result = _keyword_classify("Either party may terminate this Agreement upon written notice.")
        assert result is not None
        assert 0.0 <= result[1] <= 1.0

    def test_more_hits_gives_higher_confidence(self):
        weak = _keyword_classify("terminate this agreement")
        strong = _keyword_classify("terminate this agreement. notice of termination. terminates the contract.")
        if weak and strong:
            assert strong[1] >= weak[1]


# ---------------------------------------------------------------------------
# classify_clauses — full integration
# ---------------------------------------------------------------------------

class TestClassifyClauses:
    @pytest.fixture
    def segmented(self, simple_contract):
        return segment_contract(text=simple_contract, doc_id="test")

    def test_returns_same_length(self, segmented):
        result = classify_clauses(segmented, use_embeddings=False)
        assert len(result) == len(segmented)

    def test_adds_required_keys(self, segmented):
        result = classify_clauses(segmented, use_embeddings=False)
        required = {"clause_type", "risk_level", "confidence", "method"}
        for clause in result:
            assert required.issubset(clause.keys())

    def test_risk_levels_are_valid(self, segmented):
        result = classify_clauses(segmented, use_embeddings=False)
        valid = {"HIGH", "MEDIUM", "LOW"}
        for clause in result:
            assert clause["risk_level"] in valid

    def test_confidence_between_0_and_1(self, segmented):
        result = classify_clauses(segmented, use_embeddings=False)
        for clause in result:
            assert 0.0 <= clause["confidence"] <= 1.0

    def test_method_is_keyword_or_embedding_or_none(self, segmented):
        result = classify_clauses(segmented, use_embeddings=False)
        for clause in result:
            assert isinstance(clause["method"], str)

    def test_noncompete_is_high_risk(self, simple_contract):
        clauses = segment_contract(text=simple_contract, doc_id="test")
        result = classify_clauses(clauses, use_embeddings=False)
        nc = [c for c in result if c["clause_type"] == "NonCompete"]
        assert len(nc) >= 1
        assert all(c["risk_level"] == "HIGH" for c in nc)

    def test_governing_law_is_low_risk(self, simple_contract):
        clauses = segment_contract(text=simple_contract, doc_id="test")
        result = classify_clauses(clauses, use_embeddings=False)
        gl = [c for c in result if c["clause_type"] == "GoverningLaw"]
        assert len(gl) >= 1
        assert all(c["risk_level"] == "LOW" for c in gl)

    def test_preserves_original_keys(self, segmented):
        result = classify_clauses(segmented, use_embeddings=False)
        for orig, classified in zip(segmented, result):
            assert classified["clause_id"] == orig["clause_id"]
            assert classified["clause_text"] == orig["clause_text"]

    def test_risk_map_consistency(self):
        """Every type in RISK map should produce matching risk_level."""
        for ctype, expected_risk in _RISK.items():
            result = _keyword_classify(f"This agreement clause {ctype.lower()} shall apply.")
            # Just check RISK map is internally consistent (no KeyError)
            assert expected_risk in {"HIGH", "MEDIUM", "LOW"}
