"""
tests/test_compare_analyzers.py
===============================
Unit tests for evaluation/compare_analyzers.py (v2 Step 4). Synthetic rows and fake
analyzers only: no model downloads, no LLM calls, no real eval data.
"""
import csv
import json

import pytest

import evaluation.compare_analyzers as ca
from evaluation.gold_set import GoldRow
from evaluation.score_analyzer import AnalysisResult


def gold(id, scope="in", outcome="unenforceable", facts=("fact one", "fact two")):
    return GoldRow(
        id=id, url=f"https://indiankanoon.org/doc/{abs(hash(id)) % 10**6}/", doc_id="1", case_name="", court="",
        judgment_date="2025", clause_text=f"clause text for {id} " * 3, clause_type="NonCompete",
        sections_applied=["ICA_S27"], outcome=outcome, holding="held", risk_facts=list(facts),
        source_para="", notes="", index_sections=["ICA_S27"] if scope != "out" else [], scope=scope,
    )


def res(text="answer", refused=False, enforceability=None):
    extra = {"enforceability": enforceability} if enforceability is not None else {}
    return AnalysisResult(clause_type="NonCompete", retrieved=["ICA_S27"], cited=["ICA_S27"],
                          answer_text=text, refused=refused, extra=extra)


# --- statistics ---------------------------------------------------------------

def test_wilson_interval():
    assert ca.wilson_interval(0, 0) is None
    lo, hi = ca.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = ca.wilson_interval(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)


def test_mcnemar_exact():
    assert ca.mcnemar_exact(0, 0) is None
    assert ca.mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert ca.mcnemar_exact(3, 0) == pytest.approx(0.25)
    assert ca.mcnemar_exact(4, 4) == 1.0


# --- blinded sheet --------------------------------------------------------------

def test_blind_rows_pair_each_clause_and_hide_labels():
    rows = [gold(f"G{i}") for i in range(10)]
    results = {"v1_template": [res(f"v1 {i}") for i in range(10)],
               "v2_x": [res(f"v2 {i}") for i in range(10)]}
    sheet, key = ca.build_blind_rows(rows, results, n=6, seed=3)
    assert len(sheet) == 12 and set(key) == {r["review_id"] for r in sheet}
    assert "analyzer" not in sheet[0]                                   # no label in the sheet
    for a, b in zip(sheet[::2], sheet[1::2]):                           # outputs of a clause are adjacent
        assert a["clause_id"] == b["clause_id"]
        assert {key[a["review_id"]]["analyzer"], key[b["review_id"]]["analyzer"]} == {"v1_template", "v2_x"}
        assert a["output_text"].split()[0] == key[a["review_id"]]["analyzer"][:2]
    firsts = {key[a["review_id"]]["analyzer"] for a in sheet[::2]}
    assert firsts == {"v1_template", "v2_x"}                            # order is shuffled, not fixed
    again, _ = ca.build_blind_rows(rows, results, n=6, seed=3)
    assert [r["review_id"] + r["output_text"] for r in again] == [r["review_id"] + r["output_text"] for r in sheet]


def test_risk_facts_are_numbered_for_the_reviewer():
    sheet, _ = ca.build_blind_rows([gold("G1")], {"a": [res()], "b": [res()]}, n=1, seed=0)
    assert sheet[0]["risk_facts_gold"] == "1) fact one 2) fact two" and sheet[0]["n_risk_facts"] == 2


# --- parsing the filled sheet -----------------------------------------------------

def sheet_row(rid, scope="in", outcome="unenforceable", refused="N", n_facts=2, **checks):
    row = {"review_id": rid, "scope": scope, "outcome_gold": outcome, "refused": refused, "n_risk_facts": str(n_facts)}
    for name, col in ca.CHECK_COLUMNS.items():
        row[col] = str(checks.get(name, ""))
    return row


KEY = {"R001": {"row_id": "G1", "analyzer": "a"}, "R002": {"row_id": "G1", "analyzer": "b"}}


def test_parse_accepts_lenient_yes_no():
    rows = [sheet_row("R001", faithful="yes", missed_risk_facts=1, unsupported_citation="n", contradicts_court="0")]
    recs, incomplete = ca.parse_review(rows, KEY)
    r = recs[0]
    assert r["faithful"] is True and r["unsupported_citation"] is False and r["contradicts_court"] is False
    assert r["missed_risk_facts"] == 1 and incomplete == []


@pytest.mark.parametrize("bad,msg", [
    ({"faithful": "maybe"}, "Y or N"),
    ({"missed_risk_facts": "two"}, "whole number"),
    ({"missed_risk_facts": 3}, "outside 0..2"),
])
def test_parse_rejects_bad_values(bad, msg):
    with pytest.raises(ca.SheetError, match=msg):
        ca.parse_review([sheet_row("R001", **bad)], KEY)


def test_parse_rejects_unknown_review_id():
    with pytest.raises(ca.SheetError, match="not in the key"):
        ca.parse_review([sheet_row("R999")], KEY)


def test_refusal_is_auto_filled():
    recs, incomplete = ca.parse_review([sheet_row("R001", refused="Y", outcome="enforceable")], KEY)
    r = recs[0]
    assert r["faithful"] is True and r["unsupported_citation"] is False and r["overflags_valid_clause"] is False
    assert r["missed_risk_facts"] == 2 and incomplete == []
    assert set(r["auto_filled"]) == {"faithful", "unsupported_citation", "contradicts_court",
                                     "overflags_valid_clause", "missed_risk_facts"}


def test_incomplete_detection_depends_on_row():
    full = dict(faithful="Y", unsupported_citation="N", contradicts_court="N")
    rows = [
        sheet_row("R001", outcome="enforceable", missed_risk_facts=0, **full),   # needs overflags too
        sheet_row("R002", scope="out", **full),                                  # out of scope: no missed facts needed
    ]
    _, incomplete = ca.parse_review(rows, KEY)
    assert incomplete == ["R001"]


# --- metrics ----------------------------------------------------------------------

def rec(rid, clause, analyzer, faithful=True, missed=0, unsupported=False, contradicts=False,
        overflags=None, scope="in", outcome="unenforceable", n_facts=2):
    return {"review_id": rid, "clause_id": clause, "analyzer": analyzer, "scope": scope, "outcome": outcome,
            "refused": False, "n_risk_facts": n_facts, "auto_filled": [], "faithful": faithful,
            "missed_risk_facts": missed, "unsupported_citation": unsupported, "contradicts_court": contradicts,
            "overflags_valid_clause": overflags}


def test_human_metrics_scope_and_outcome_rules():
    records = [
        rec("R1", "G1", "a", faithful=True, missed=1),
        rec("R2", "G2", "a", faithful=False, missed=0, outcome="enforceable", overflags=True),
        rec("R3", "G3", "a", missed=2, scope="out"),                     # out of scope: not in missed-risk
        rec("R4", "G1", "b"),
    ]
    m = ca.human_metrics(records, "a")
    assert m["reviewed_outputs"] == 3
    assert m["faithful"]["pass_rate"] == pytest.approx(2 / 3) and m["faithful"]["n"] == 3
    assert m["overflags_valid_clause"] == {"n": 1, "pass_rate": 0.0, "ci95": list(ca.wilson_interval(0, 1))}
    mf = m["missed_risk_facts"]
    assert (mf["facts_missed"], mf["facts_total"], mf["missed_rate"]) == (1, 4, 0.25)
    assert mf["all_facts_stated_rate"] == 0.5


def test_paired_tests_count_discordant_pairs():
    records = [
        rec("R1", "G1", "a", faithful=True, missed=0), rec("R2", "G1", "b", faithful=False, missed=2),
        rec("R3", "G2", "a", faithful=True, missed=1), rec("R4", "G2", "b", faithful=True, missed=1),
        rec("R5", "G3", "a", faithful=False, missed=2), rec("R6", "G3", "b", faithful=True, missed=0),
        rec("R7", "G4", "a"),                                            # unpaired: ignored
    ]
    t = ca.paired_tests(records, "a", "b")
    assert t["faithful"]["pairs"] == 3 and t["faithful"]["only_a_better"] == 1 and t["faithful"]["only_b_better"] == 1
    assert t["missed_risk_facts"]["only_a_better"] == 1 and t["missed_risk_facts"]["only_b_better"] == 1
    assert t["faithful"]["p_value"] == 1.0


def test_outcome_agreement_maps_enforceability():
    rows = [gold("G1", outcome="unenforceable"), gold("G2", outcome="enforceable"), gold("G3"), gold("G4")]
    results = [res(enforceability="likely_unenforceable"), res(enforceability="likely_unenforceable"),
               res(enforceability="unclear"), res(refused=True, enforceability="")]
    oa = ca.outcome_agreement(rows, results)
    assert oa == {"answered": 3, "unclear": 1, "agreement_when_decided": 0.5, "n_decided": 2}
    assert ca.outcome_agreement(rows, [res() for _ in rows]) is None     # v1 makes no call


# --- end to end ---------------------------------------------------------------------

def _fake_pipeline(monkeypatch):
    import pipeline.classifier as classifier
    import pipeline.retriever as retriever

    sections = [{"section_id": "ICA_S27", "title": "Agreement in restraint of trade, void", "score": 0.0328,
                 "source": "statute_map",
                 "text": "Section 27. Agreement in restraint of trade, void. Every agreement by which any one "
                         "is restrained from exercising a lawful profession, trade or business of any kind, "
                         "is to that extent void."}]
    monkeypatch.setattr(classifier, "classify_clauses", lambda clauses, use_embeddings=True, **kw: [
        dict(c, clause_type="NonCompete", risk_level="HIGH", method="ml", confidence=0.8) for c in clauses])
    monkeypatch.setattr(retriever, "retrieve_statutes", lambda clauses, top_k=5, cache=None, **kw: [
        dict(c, retrieved_sections=sections) for c in clauses])


def _write_gold(path, n=4):
    header = ["id", "url", "clause_text", "clause_type", "sections_applied", "outcome", "holding", "risk_facts"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for i in range(n):
            w.writerow({"id": f"T{i}", "url": f"https://indiankanoon.org/doc/91000{i}/", "clause_type": "NonCompete",
                        "clause_text": f"The Employee shall not, for {i + 1} years after leaving the Company, "
                                       "work for any competitor anywhere in India.",
                        "sections_applied": "ICA_S27", "outcome": "enforceable" if i % 2 else "unenforceable",
                        "holding": "Held.", "risk_facts": "Post-employment restraint is void || Duration is irrelevant"})


def test_run_then_score_end_to_end(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    gold_csv, out = tmp_path / "gold.csv", tmp_path / "cmp"
    _write_gold(gold_csv)
    rc = ca.main(["run", "--gold", str(gold_csv), "--out-dir", str(out), "--no-embeddings",
                  "--llm-backend", "fake", "--review-n", "3", "--clauses-per-contract", "40"])
    assert rc == 0
    key_doc = json.loads((out / ca.KEY_NAME).read_text())
    assert key_doc["manifest"]["analyzers"] == ["v1_template", "v2_fake-echo"]
    assert "ICA_S27" in (out / ca.SECTIONS_NAME).read_text()

    sheet_path = out / ca.SHEET_NAME
    with open(sheet_path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    for r in rows:
        v1 = key_doc["key"][r["review_id"]]["analyzer"] == "v1_template"
        r[ca.CHECK_COLUMNS["faithful"]] = "N" if v1 else "Y"
        r[ca.CHECK_COLUMNS["missed_risk_facts"]] = "1"
        r[ca.CHECK_COLUMNS["unsupported_citation"]] = "N"
        r[ca.CHECK_COLUMNS["contradicts_court"]] = "N"
        r[ca.CHECK_COLUMNS["overflags_valid_clause"]] = "Y" if v1 else "N"
    with open(sheet_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    assert ca.main(["score", "--out-dir", str(out)]) == 0
    comp = json.loads((out / "comparison.json").read_text())
    assert comp["human"]["v1_template"]["faithful"]["pass_rate"] == 0.0
    assert comp["human"]["v2_fake-echo"]["faithful"]["pass_rate"] == 1.0
    assert comp["paired"]["faithful"]["only_v2_fake-echo_better"] == 3
    assert comp["automatic"]["v2_fake-echo"]["cost_usd_per_contract"] == 0.0
    assert comp["incomplete"] == []
    md = (out / "comparison.md").read_text()
    assert "| Faithfulness pass rate |" in md and "Cost per contract (~40 clauses)" in md


def test_score_reports_missing_files(tmp_path):
    assert ca.main(["score", "--out-dir", str(tmp_path)]) == 2


def test_score_rejects_a_sheet_with_rows_removed(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    gold_csv, out = tmp_path / "gold.csv", tmp_path / "cmp"
    _write_gold(gold_csv)
    ca.main(["run", "--gold", str(gold_csv), "--out-dir", str(out), "--no-embeddings",
             "--llm-backend", "fake", "--review-n", "2"])
    sheet_path = out / ca.SHEET_NAME
    lines = sheet_path.read_text(encoding="utf-8-sig").splitlines()
    sheet_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8-sig")
    assert ca.main(["score", "--out-dir", str(out)]) == 2
