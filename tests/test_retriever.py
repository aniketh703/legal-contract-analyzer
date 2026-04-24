"""
tests/test_retriever.py
=======================
Unit tests for pipeline/retriever.py
"""
import pytest
from pathlib import Path
from pipeline.retriever import (
    _load_registry,
    _build_bm25,
    _tokenise,
    _rrf_merge,
    _STATUTE_MAP,
)

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge_base"


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_lowercases(self):
        assert _tokenise("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _tokenise("clause, (section).") == ["clause", "section"]

    def test_empty_string(self):
        assert _tokenise("") == []

    def test_numbers_excluded(self):
        tokens = _tokenise("section 27 agreement")
        assert "27" not in tokens


# ---------------------------------------------------------------------------
# _rrf_merge
# ---------------------------------------------------------------------------

class TestRrfMerge:
    def test_returns_sorted_by_score(self):
        lists = [["A", "B", "C"], ["A", "C", "D"]]
        result = _rrf_merge(lists)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_item_in_multiple_lists_ranks_higher(self):
        lists = [["A", "B"], ["A", "C"]]
        result = _rrf_merge(lists)
        top_id = result[0][0]
        assert top_id == "A"

    def test_empty_lists(self):
        assert _rrf_merge([]) == []

    def test_single_list(self):
        result = _rrf_merge([["X", "Y", "Z"]])
        ids = [i for i, _ in result]
        assert ids == ["X", "Y", "Z"]

    def test_scores_are_positive(self):
        result = _rrf_merge([["A", "B"], ["B", "C"]])
        assert all(s > 0 for _, s in result)

    def test_deduplicates(self):
        lists = [["A", "A", "B"], ["A"]]
        result = _rrf_merge(lists)
        ids = [i for i, _ in result]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# _load_registry
# ---------------------------------------------------------------------------

class TestLoadRegistry:
    def test_loads_dict(self):
        reg = _load_registry(KB_DIR)
        assert isinstance(reg, dict)

    def test_has_ica_sections(self):
        reg = _load_registry(KB_DIR)
        assert len(reg) > 0
        assert any(k.startswith("ICA_S") for k in reg)

    def test_section_has_required_keys(self):
        reg = _load_registry(KB_DIR)
        sample = next(iter(reg.values()))
        assert "id" in sample
        assert "text" in sample
        assert "title" in sample

    def test_key_statute_sections_present(self):
        reg = _load_registry(KB_DIR)
        critical = ["ICA_S27", "ICA_S56", "ICA_S73", "ICA_S124"]
        for sid in critical:
            assert sid in reg, f"{sid} missing from knowledge base"


# ---------------------------------------------------------------------------
# _build_bm25
# ---------------------------------------------------------------------------

class TestBuildBm25:
    @pytest.fixture
    def bm25(self):
        reg = _load_registry(KB_DIR)
        return _build_bm25(reg)

    def test_returns_three_items(self, bm25):
        assert len(bm25) == 3

    def test_section_ids_match_registry(self, bm25):
        reg = _load_registry(KB_DIR)
        section_ids, _, _ = bm25
        assert set(section_ids) == set(reg.keys())

    def test_score_is_float(self, bm25):
        section_ids, corpus, bm25_fn = bm25
        score = bm25_fn(["agreement", "terminate"], corpus[0])
        assert isinstance(score, float)

    def test_relevant_query_scores_higher(self, bm25):
        reg = _load_registry(KB_DIR)
        section_ids, corpus, bm25_fn = bm25
        # S27 is about restraint of trade — should score higher for trade query
        s27_idx = section_ids.index("ICA_S27") if "ICA_S27" in section_ids else None
        if s27_idx is None:
            pytest.skip("ICA_S27 not in index")
        relevant_score = bm25_fn(["restraint", "trade", "void"], corpus[s27_idx])
        irrelevant_score = bm25_fn(["bailee", "bailment", "goods"], corpus[s27_idx])
        assert relevant_score >= irrelevant_score


# ---------------------------------------------------------------------------
# _STATUTE_MAP
# ---------------------------------------------------------------------------

class TestStatuteMap:
    def test_all_types_have_entries(self):
        expected_types = [
            "NonCompete", "ForceMajeure", "Termination", "Indemnification",
            "Arbitration", "Confidentiality", "LiabilityCap", "IPAssignment",
            "PaymentTerms", "Renewal",
        ]
        for t in expected_types:
            assert t in _STATUTE_MAP

    def test_noncompete_maps_to_s27(self):
        assert "ICA_S27" in _STATUTE_MAP["NonCompete"]

    def test_force_majeure_maps_to_s56(self):
        assert "ICA_S56" in _STATUTE_MAP["ForceMajeure"]

    def test_termination_maps_to_s73(self):
        assert "ICA_S73" in _STATUTE_MAP["Termination"]

    def test_indemnification_maps_to_s124(self):
        assert "ICA_S124" in _STATUTE_MAP["Indemnification"]

    def test_statute_ids_exist_in_registry(self):
        reg = _load_registry(KB_DIR)
        for ctype, sids in _STATUTE_MAP.items():
            for sid in sids:
                assert sid in reg, f"Statute {sid} (for {ctype}) missing from KB"
