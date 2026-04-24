"""
tests/test_segmenter.py
=======================
Unit tests for pipeline/segmenter.py
"""
import pytest
from pipeline.segmenter import segment_contract, _clean, _split_into_segments


# ---------------------------------------------------------------------------
# _clean
# ---------------------------------------------------------------------------

class TestClean:
    def test_fixes_hyphenated_linebreaks(self):
        assert _clean("obliga-\ntion") == "obligation"

    def test_collapses_blank_lines(self):
        result = _clean("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_strips_nbsp(self):
        result = _clean("hello\xa0world")
        assert "\xa0" not in result

    def test_returns_stripped_text(self):
        result = _clean("   hello   ")
        assert result == "hello"


# ---------------------------------------------------------------------------
# segment_contract — basic behaviour
# ---------------------------------------------------------------------------

class TestSegmentContract:
    def test_raises_without_path_or_text(self):
        with pytest.raises(ValueError):
            segment_contract()

    def test_returns_list(self, simple_contract):
        result = segment_contract(text=simple_contract, doc_id="test")
        assert isinstance(result, list)

    def test_extracts_multiple_clauses(self, simple_contract):
        result = segment_contract(text=simple_contract, doc_id="test")
        assert len(result) >= 5

    def test_clause_dict_has_required_keys(self, simple_contract):
        result = segment_contract(text=simple_contract, doc_id="test")
        required = {"clause_id", "clause_text", "heading", "char_start", "char_end"}
        for clause in result:
            assert required.issubset(clause.keys()), f"Missing keys in {clause}"

    def test_clause_ids_are_unique(self, simple_contract):
        result = segment_contract(text=simple_contract, doc_id="test")
        ids = [c["clause_id"] for c in result]
        assert len(ids) == len(set(ids))

    def test_clause_ids_use_doc_id_prefix(self, simple_contract):
        result = segment_contract(text=simple_contract, doc_id="mycontract")
        assert all(c["clause_id"].startswith("mycontract_") for c in result)

    def test_no_clause_shorter_than_min(self, simple_contract):
        result = segment_contract(text=simple_contract, doc_id="test")
        assert all(len(c["clause_text"]) >= 80 for c in result)

    def test_no_clause_longer_than_max(self, simple_contract):
        result = segment_contract(text=simple_contract, doc_id="test")
        assert all(len(c["clause_text"]) <= 2000 for c in result)

    def test_minimal_contract_returns_clauses(self, minimal_contract):
        result = segment_contract(text=minimal_contract, doc_id="min")
        assert len(result) >= 1

    def test_empty_text_returns_empty_list(self, unparseable_text):
        result = segment_contract(text=unparseable_text, doc_id="empty")
        assert result == []

    def test_auto_generates_doc_id(self, simple_contract):
        result = segment_contract(text=simple_contract)
        assert all("_c" in c["clause_id"] for c in result)

    def test_detects_termination_content(self, simple_contract):
        result = segment_contract(text=simple_contract, doc_id="test")
        all_text = " ".join(c["clause_text"] for c in result).lower()
        assert "terminate" in all_text

    def test_detects_arbitration_content(self, simple_contract):
        result = segment_contract(text=simple_contract, doc_id="test")
        all_text = " ".join(c["clause_text"] for c in result).lower()
        assert "arbitration" in all_text
