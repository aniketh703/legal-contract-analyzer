"""
tests/test_score_analyzer.py
============================
Unit tests for evaluation/score_analyzer.py. Fake analyzers and synthetic rows only:
no model downloads, no real eval data.
"""
import csv
import json
import time

import pytest

import evaluation.score_analyzer as sa
from evaluation.gold_set import GoldRow
from evaluation.score_analyzer import (
    AnalysisResult, V1Analyzer, percentile, run_analyzer, score_results,
    stratified_sample, write_review_sheet,
)


def gold(id, scope, index_sections, applied=None, outcome="unenforceable", ctype="NonCompete"):
    return GoldRow(
        id=id, url=f"https://indiankanoon.org/doc/{abs(hash(id)) % 10**6}/", doc_id="1", case_name="", court="",
        judgment_date="2025", clause_text=f"clause text for {id} " * 3, clause_type=ctype,
        sections_applied=applied or list(index_sections), outcome=outcome, holding="held", risk_facts=["fact"],
        source_para="", notes="", index_sections=list(index_sections), scope=scope,
    )


def result(retrieved, cited=None, ctype="NonCompete", refused=False, scores=None, cost=0.0, latency=0.01):
    return AnalysisResult(
        clause_type=ctype, retrieved=retrieved, cited=retrieved if cited is None else cited, refused=refused,
        retrieved_scores=scores or [1.0 / (i + 1) for i in range(len(retrieved))], cost_usd=cost, latency_s=latency,
    )


# --- metric math ------------------------------------------------------------

def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([10, 20, 30, 40], 95) == pytest.approx(38.5)
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7


@pytest.fixture
def scored():
    rows = [
        gold("A", "in", ["ICA_S27"]),
        gold("B", "in", ["ICA_S74", "ICA_S73"], ctype="LiquidatedDamages"),
        gold("C", "out", [], applied=["SRA_S14"]),
        gold("D", "out", [], applied=["COPY_S18"]),
        gold("E", "in", ["ICA_S56"], ctype="ForceMajeure"),
    ]
    results = [
        result(["ICA_S1", "ICA_S27", "ICA_S3"], cited=["ICA_S27", "ICA_S99"]),   # hit at rank 2; S99 unsupported
        result(["ICA_S74", "ICA_S9"], ctype="LiquidatedDamages"),                  # hit at rank 1, recall 1/2
        result(["ICA_S5"], refused=True),
        result(["ICA_S6"], refused=False),
        result([], ctype="Unknown", refused=True),                                 # in-scope but refused
    ]
    return score_results(rows, results)


def test_counts_and_hits(scored):
    assert scored["counts"] == {"rows": 5, "in_scope": 3, "mixed": 0, "out_of_scope": 2}
    r = scored["retrieval"]
    assert r["n"] == 3
    assert r["hit@1"] == pytest.approx(1 / 3)       # only B
    assert r["hit@3"] == pytest.approx(2 / 3)       # A and B
    assert r["recall@1"] == pytest.approx((0 + 0.5 + 0) / 3)
    assert r["mrr"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_type_accuracy_ignores_blank_and_other(scored):
    # A and B correct; C/D have gold type NonCompete but predicted NonCompete too; E wrong (Unknown)
    assert scored["type_accuracy"] == pytest.approx(4 / 5)


def test_citation_metrics(scored):
    c = scored["citations"]
    # A cited {S27,S99}: precision 1/2; B cited {S74,S9}: gold applied {S74,S73} -> 1/2; E cited nothing
    assert c["precision_vs_court"] == pytest.approx(0.5)
    assert c["rows_without_citations"] == 1
    # cited totals: A=2, B=2, C=1, D=1, E=0 -> 6; unsupported only S99
    assert c["unsupported_rate"] == pytest.approx(1 / 6)
    assert c["rows_with_unsupported"] == pytest.approx(1 / 5)


def test_refusal_metrics(scored):
    r = scored["refusal"]
    assert r["refused_out_of_scope"] == 0.5
    assert r["answered_out_of_scope"] == 0.5
    assert r["false_refusal_in_scope"] == pytest.approx(1 / 3)


def test_latency_cost_and_confidence(scored):
    assert scored["latency_ms"]["p50"] == pytest.approx(10.0)
    assert scored["cost_usd"]["total"] == 0.0
    assert scored["retrieval_confidence"]["mean_top_score_out_of_scope"] == pytest.approx(1.0)


def test_breakdowns_and_row_records(scored):
    assert set(scored["by_outcome"]) == {"unenforceable"}
    assert set(scored["by_gold_type"]) == {"NonCompete", "LiquidatedDamages", "ForceMajeure"}
    rec = next(r for r in scored["rows"] if r["id"] == "A")
    assert rec["first_hit_rank"] == 2 and rec["hit@1"] is False and rec["hit@3"] is True
    assert "hit@1" not in next(r for r in scored["rows"] if r["id"] == "C")


def test_mixed_rows_are_retrieval_scored_but_not_refusal_scored():
    rows = [gold("M", "mixed", ["ICA_S27"], applied=["ICA_S27", "SRA_S14"])]
    m = score_results(rows, [result(["ICA_S27"])])
    assert m["counts"]["mixed"] == 1
    assert m["retrieval"]["hit@1"] == 1.0
    assert m["refusal"]["refused_out_of_scope"] is None and m["refusal"]["false_refusal_in_scope"] is None


# --- runner -----------------------------------------------------------------

class FakeAnalyzer:
    name = "fake"

    def __init__(self):
        self.warmups = 0
        self.calls = []

    def warmup(self):
        self.warmups += 1

    def analyze(self, clause_text):
        self.calls.append(clause_text)
        time.sleep(0.01)
        return result(["ICA_S27"])


def test_run_analyzer_warms_up_once_and_times_each_call():
    rows = [gold("A", "in", ["ICA_S27"]), gold("B", "in", ["ICA_S27"])]
    fake = FakeAnalyzer()
    results = run_analyzer(fake, rows)
    assert fake.warmups == 1
    assert fake.calls == [r.clause_text for r in rows]
    assert all(r.latency_s >= 0.01 for r in results)


# --- review sheet -----------------------------------------------------------

def test_stratified_sample_is_balanced_and_deterministic():
    rows = [gold(f"in{i}", "in", ["ICA_S27"]) for i in range(10)] + \
           [gold(f"out{i}", "out", [], applied=["SRA_S14"]) for i in range(2)] + \
           [gold("ok", "in", ["ICA_S74"], outcome="enforceable")]
    picked = stratified_sample(rows, 4, seed=1)
    assert picked == stratified_sample(rows, 4, seed=1)
    groups = {(rows[i].scope, rows[i].outcome) for i in picked}
    assert groups == {("in", "unenforceable"), ("out", "unenforceable"), ("in", "enforceable")}
    assert stratified_sample(rows, 99) == list(range(len(rows)))


def test_review_sheet_has_blank_check_columns(tmp_path):
    rows = [gold("A", "in", ["ICA_S27"]), gold("B", "out", [], applied=["SRA_S14"])]
    results = [result(["ICA_S27"]), result(["ICA_S5"])]
    path = tmp_path / "review.csv"
    assert write_review_sheet(path, rows, results, "fake", n=5, seed=0) == 2
    out = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    assert {r["id"] for r in out} == {"A", "B"}
    assert all(r[col] == "" for r in out for col in sa.REVIEW_CHECKS)
    assert out[0]["analyzer"] == "fake"


# --- v1 adapter -------------------------------------------------------------

def test_v1_adapter_builds_result_from_pipeline_output(monkeypatch):
    import pipeline.classifier as classifier
    import pipeline.retriever as retriever

    sections = [{"section_id": "ICA_S27", "title": "Restraint", "score": 0.9, "source": "statute_map"},
                {"section_id": "ICA_S73", "title": "Damages", "score": 0.4, "source": "bm25"}]
    seen = {}

    def fake_classify(clauses, use_embeddings=True, **kw):
        seen["use_embeddings"] = use_embeddings
        return [dict(c, clause_type="NonCompete", risk_level="HIGH", method="ml", confidence=0.8) for c in clauses]

    def fake_retrieve(clauses, top_k=5, cache=None, **kw):
        seen["top_k"] = top_k
        return [dict(c, retrieved_sections=sections) for c in clauses]

    monkeypatch.setattr(classifier, "classify_clauses", fake_classify)
    monkeypatch.setattr(retriever, "retrieve_statutes", fake_retrieve)

    res = V1Analyzer(use_embeddings=False, top_k=2).analyze("some clause text")
    assert seen == {"use_embeddings": False, "top_k": 2}
    assert res.clause_type == "NonCompete" and res.risk_level == "HIGH"
    assert res.retrieved == res.cited == ["ICA_S27", "ICA_S73"]
    assert res.retrieved_scores == [0.9, 0.4]
    assert "<" not in res.answer_text and "Recommended action:" in res.answer_text
    assert res.extra["explained_sections"] == ["ICA_S27"]
    assert res.refused is False and res.cost_usd == 0.0


# --- CLI --------------------------------------------------------------------

def _write_gold(path, rows):
    header = ["id", "url", "clause_text", "clause_type", "sections_applied", "outcome", "holding"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def test_main_scores_and_writes_report(tmp_path, monkeypatch):
    monkeypatch.setitem(sa.ANALYZERS, "fake", lambda args: FakeAnalyzer())
    gold_csv = tmp_path / "gold.csv"
    _write_gold(gold_csv, [{
        "id": "T1", "url": "https://indiankanoon.org/doc/9000001/", "clause_type": "NonCompete",
        "clause_text": "The Employee shall not, for two years after leaving the Company, work for a competitor.",
        "sections_applied": "ICA_S27", "outcome": "unenforceable", "holding": "Held void.",
    }])
    out, sheet = tmp_path / "out.json", tmp_path / "sheet.csv"
    rc = sa.main(["--analyzer", "fake", "--gold", str(gold_csv), "--out", str(out), "--review-sheet", str(sheet)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["analyzer"] == "fake" and report["metrics"]["retrieval"]["hit@1"] == 1.0
    assert sheet.exists()


def test_main_refuses_invalid_rows_unless_skipped(tmp_path, monkeypatch):
    monkeypatch.setitem(sa.ANALYZERS, "fake", lambda args: FakeAnalyzer())
    gold_csv = tmp_path / "gold.csv"
    _write_gold(gold_csv, [
        {"id": "T1", "url": "https://indiankanoon.org/doc/9000001/", "clause_type": "NonCompete",
         "clause_text": "The Employee shall not, for two years after leaving the Company, work for a competitor.",
         "sections_applied": "ICA_S27", "outcome": "unenforceable", "holding": "Held void."},
        {"id": "T2", "url": "https://indiankanoon.org/doc/9000002/", "clause_type": "NonCompete",
         "clause_text": "Another clause that is long enough to pass the length check here.",
         "sections_applied": "ICA_S27", "outcome": "maybe", "holding": "x"},
    ])
    out = tmp_path / "out.json"
    assert sa.main(["--analyzer", "fake", "--gold", str(gold_csv), "--out", str(out)]) == 2
    assert sa.main(["--analyzer", "fake", "--gold", str(gold_csv), "--out", str(out), "--skip-invalid"]) == 0
    assert json.loads(out.read_text())["rows_skipped_invalid"] == 1


# --- v2 adapter ---------------------------------------------------------------

def _fake_pipeline(monkeypatch, sections):
    import pipeline.classifier as classifier
    import pipeline.retriever as retriever

    monkeypatch.setattr(classifier, "classify_clauses", lambda clauses, use_embeddings=True, **kw: [
        dict(c, clause_type="NonCompete", risk_level="HIGH", method="ml", confidence=0.8) for c in clauses])
    monkeypatch.setattr(retriever, "retrieve_statutes", lambda clauses, top_k=5, cache=None, **kw: [
        dict(c, retrieved_sections=sections) for c in clauses])


V2_SECTIONS = [
    {"section_id": "ICA_S27", "title": "Restraint", "score": 0.0328, "source": "statute_map",
     "text": "Section 27. Restraint. Every agreement by which any one is restrained from exercising a lawful profession."},
    {"section_id": "ICA_S73", "title": "Damages", "score": 0.0161, "source": "bm25",
     "text": "Section 73. Damages. When a contract has been broken, the party who suffers is compensated."},
]


def test_v2_adapter_reports_raw_citations_and_checks(monkeypatch):
    from pipeline.llm_client import FakeClient
    from evaluation.score_analyzer import V2Analyzer

    _fake_pipeline(monkeypatch, V2_SECTIONS)
    reply = json.dumps({"status": "answered", "risk_level": "high", "enforceability": "likely_unenforceable",
                        "risks": [{"statement": "Void restraint.", "citations": ["ICA_S27"],
                                   "evidence_quote": "is restrained from exercising a lawful profession"},
                                  {"statement": "Invented.", "citations": ["SRA_S14"], "evidence_quote": "x"}],
                        "recommended_action": "Narrow it."})
    analyzer = V2Analyzer(use_embeddings=False, client=FakeClient(responses=[reply]))
    res = analyzer.analyze("The employee shall not compete for two years.")
    assert analyzer.name == "v2_fake-echo" and analyzer.config()["backend"] == "fake"
    assert res.retrieved == ["ICA_S27", "ICA_S73"]
    assert res.cited == ["ICA_S27", "SRA_S14"]              # raw: the scorer counts SRA_S14 as unsupported
    assert res.extra["shown_citations"] == ["ICA_S27"]
    assert res.extra["checks"]["risks_dropped"] == 1 and not res.refused
    assert "SRA_S14" not in res.answer_text and "Narrow it." in res.answer_text


def test_v2_warmup_spends_no_llm_call(monkeypatch):
    from pipeline.llm_client import FakeClient
    from evaluation.score_analyzer import V2Analyzer

    _fake_pipeline(monkeypatch, V2_SECTIONS)
    client = FakeClient(responses=[])
    V2Analyzer(use_embeddings=False, client=client).warmup()
    assert client.calls == []


def test_checks_are_summed_only_when_present():
    rows = [gold("A", "in", ["ICA_S27"]), gold("B", "in", ["ICA_S27"])]
    plain = [result(["ICA_S27"]), result(["ICA_S27"])]
    assert "checks" not in score_results(rows, plain)
    with_checks = [result(["ICA_S27"]), result(["ICA_S27"])]
    with_checks[0].extra = {"checks": {"risks_dropped": 1, "llm_calls": 1}}
    with_checks[1].extra = {"checks": {"risks_dropped": 2, "llm_calls": 2, "note": "ignored"}}
    assert score_results(rows, with_checks)["checks"] == {"risks_dropped": 3, "llm_calls": 3}


def test_main_v2_with_fake_backend_end_to_end(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch, V2_SECTIONS)
    gold_csv = tmp_path / "gold.csv"
    _write_gold(gold_csv, [{
        "id": "T1", "url": "https://indiankanoon.org/doc/9000001/", "clause_type": "NonCompete",
        "clause_text": "The Employee shall not, for two years after leaving the Company, work for a competitor.",
        "sections_applied": "ICA_S27", "outcome": "unenforceable", "holding": "Held void.",
    }])
    out = tmp_path / "v2.json"
    rc = sa.main(["--analyzer", "v2", "--llm-backend", "fake", "--no-embeddings",
                  "--gold", str(gold_csv), "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["analyzer"] == "v2_fake-echo" and report["config"]["model"] == "fake-echo"
    m = report["metrics"]
    assert m["checks"]["llm_calls"] == 1 and m["refusal"]["false_refusal_in_scope"] == 0.0
    assert m["citations"]["unsupported_rate"] == 0.0


def test_unknown_cost_is_not_reported_as_zero():
    rows = [gold("A", "in", ["ICA_S27"]), gold("B", "in", ["ICA_S27"])]
    known = [result(["ICA_S27"], cost=0.001), result(["ICA_S27"], cost=0.002)]
    c = score_results(rows, known)["cost_usd"]
    assert c["rows_unknown"] == 0 and c["total"] == pytest.approx(0.003)

    mixed = [result(["ICA_S27"], cost=0.001), result(["ICA_S27"])]
    mixed[1].cost_known = False
    m = score_results(rows, mixed)
    assert m["cost_usd"] == {"total": None, "per_clause": None, "rows_unknown": 1}
    assert [r["cost_known"] for r in m["rows"]] == [True, False]


def test_v2_adapter_marks_unpriced_model(monkeypatch):
    from pipeline.llm_client import FakeClient
    from evaluation.score_analyzer import V2Analyzer

    _fake_pipeline(monkeypatch, V2_SECTIONS)
    res = V2Analyzer(use_embeddings=False, client=FakeClient(model="unpriced-model")).analyze("No competing.")
    assert res.cost_known is False and res.cost_usd == 0.0


def test_scoring_run_survives_a_failing_llm_call(tmp_path, monkeypatch):
    from pipeline.llm_client import FakeClient, LLMError

    class FlakyClient(FakeClient):
        def complete(self, messages, json_mode=True, max_tokens=1024):
            if len(self.calls) == 0:
                self.calls.append(messages)
                raise LLMError("vertex_openai request failed: Timeout: read timed out")
            return super().complete(messages, json_mode, max_tokens)

    _fake_pipeline(monkeypatch, V2_SECTIONS)
    monkeypatch.setitem(sa.ANALYZERS, "flaky", lambda args: sa.V2Analyzer(use_embeddings=False, client=FlakyClient()))
    gold_csv = tmp_path / "gold.csv"
    _write_gold(gold_csv, [
        {"id": f"T{i}", "url": f"https://indiankanoon.org/doc/900000{i}/", "clause_type": "NonCompete",
         "clause_text": f"The Employee shall not, for {i} years after leaving the Company, work for a competitor.",
         "sections_applied": "ICA_S27", "outcome": "unenforceable", "holding": "Held void."}
        for i in (1, 2)
    ])
    out = tmp_path / "flaky.json"
    assert sa.main(["--analyzer", "flaky", "--gold", str(gold_csv), "--out", str(out)]) == 0
    m = json.loads(out.read_text())["metrics"]
    assert m["checks"]["llm_errors"] == 1 and m["counts"]["rows"] == 2
