"""
tests/test_gold_set.py
======================
Unit tests for evaluation/gold_set.py. All rows here are synthetic; none of it is real eval data.
"""
import csv

import pytest

from evaluation.gold_set import (
    GoldSetError, load_gold, main, normalize_section, parse_sections, summarize,
)

HEADER = ["id", "url", "case_name", "court", "judgment_date", "clause_text", "clause_type",
          "sections_applied", "outcome", "holding", "risk_facts", "source_para", "notes"]

REGISTRY = {"ICA_S27", "ICA_S74", "ICA_S73", "ARB_S8"}
TAXONOMY = {"NonCompete", "LiquidatedDamages", "Arbitration"}
CLAUSE = "The Employee shall not, for two years after leaving the Company, work for any competitor."


def row(**over):
    base = {
        "id": "T1", "url": "https://indiankanoon.org/doc/9000001/", "case_name": "A v B", "court": "SC",
        "judgment_date": "2025-03-01", "clause_text": CLAUSE, "clause_type": "NonCompete",
        "sections_applied": "ICA_S27", "outcome": "unenforceable", "holding": "Held void.",
        "risk_facts": "Post-employment restraint is void || It is not saved by an exception",
        "source_para": "12", "notes": "",
    }
    base.update(over)
    return base


def write_csv(path, rows, header=HEADER, encoding="utf-8-sig"):
    with open(path, "w", newline="", encoding=encoding) as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load(tmp_path, rows, **ctx):
    p = tmp_path / "gold.csv"
    write_csv(p, rows)
    defaults = dict(registry_ids=REGISTRY, taxonomy=TAXONOMY, exclude_ids=set(), pool_texts=[])
    defaults.update(ctx)
    return load_gold(p, **defaults)


def fields(load_result, level):
    return {(i.row_id, i.field) for i in load_result.issues if i.level == level}


# --- section parsing --------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("ICA_S27", "ICA_S27"), ("ica s.27", "ICA_S27"), ("ICA-27", "ICA_S27"),
    ("ICA 27(1)(a)", "ICA_S27"), ("IT_S43A", "IT_S43A"), ("arb_s34", "ARB_S34"), ("ICA Section 74", "ICA_S74"),
])
def test_normalize_section_variants(raw, expected):
    assert normalize_section(raw) == expected


@pytest.mark.parametrize("raw", ["Section 27", "27", "S.27", "", "ICA"])
def test_normalize_section_rejects_missing_act_prefix(raw):
    with pytest.raises(ValueError):
        normalize_section(raw)


def test_parse_sections_splits_dedupes_and_collects_errors():
    secs, errs = parse_sections("ICA_S27; ICA-27, ARB_S8\nbogus")
    assert secs == ["ICA_S27", "ARB_S8"]
    assert len(errs) == 1


# --- loading and scope ------------------------------------------------------

def test_valid_row_loads_cleanly(tmp_path):
    res = load(tmp_path, [row()])
    assert not res.error_ids
    r = res.rows[0]
    assert r.doc_id == "9000001"
    assert r.year == 2025
    assert r.risk_facts == ["Post-employment restraint is void", "It is not saved by an exception"]
    assert r.scope == "in"


def test_scope_in_out_mixed(tmp_path):
    res = load(tmp_path, [
        row(id="in", sections_applied="ICA_S27"),
        row(id="mixed", url="https://indiankanoon.org/doc/2/", clause_text=CLAUSE + " x", sections_applied="ICA_S27; SRA_S14"),
        row(id="out", url="https://indiankanoon.org/doc/3/", clause_text=CLAUSE + " y", sections_applied="SRA_S14"),
    ])
    scope = {r.id: r.scope for r in res.rows}
    assert scope == {"in": "in", "mixed": "mixed", "out": "out"}
    assert next(r for r in res.rows if r.id == "mixed").index_sections == ["ICA_S27"]


def test_blank_optional_columns_warn_but_do_not_error(tmp_path):
    res = load(tmp_path, [row(clause_type="", risk_facts="", case_name="", court="", judgment_date="")])
    assert not res.error_ids
    warned = {f for _, f in fields(res, "warn")}
    assert {"clause_type", "risk_facts", "case_name", "court", "judgment_date"} <= warned


def test_outcome_synonym_accepted_with_warning(tmp_path):
    res = load(tmp_path, [row(outcome="Void")])
    assert res.rows[0].outcome == "unenforceable"
    assert ("T1", "outcome") in fields(res, "warn")
    assert not res.error_ids


def test_unknown_clause_type_warns_with_suggestion(tmp_path):
    res = load(tmp_path, [row(clause_type="NonCompet")])
    msg = next(i.message for i in res.issues if i.field == "clause_type")
    assert "NonCompete" in msg


def test_bom_and_blank_spreadsheet_rows_are_handled(tmp_path):
    p = tmp_path / "g.csv"
    write_csv(p, [row(), {k: "" for k in HEADER}])
    res = load_gold(p, registry_ids=REGISTRY, taxonomy=TAXONOMY, exclude_ids=set(), pool_texts=[])
    assert len(res.rows) == 1


def test_missing_required_column_raises(tmp_path):
    p = tmp_path / "g.csv"
    write_csv(p, [row()], header=[h for h in HEADER if h != "holding"])
    with pytest.raises(GoldSetError, match="holding"):
        load_gold(p, registry_ids=REGISTRY, taxonomy=TAXONOMY, exclude_ids=set(), pool_texts=[])


# --- validation errors ------------------------------------------------------

@pytest.mark.parametrize("override,field", [
    ({"url": "https://example.com/x"}, "url"),
    ({"clause_text": "too short"}, "clause_text"),
    ({"outcome": "maybe"}, "outcome"),
    ({"holding": ""}, "holding"),
    ({"sections_applied": ""}, "sections_applied"),
    ({"sections_applied": "Section 27"}, "sections_applied"),
])
def test_row_errors(tmp_path, override, field):
    res = load(tmp_path, [row(**override)])
    assert ("T1", field) in fields(res, "error")
    assert res.valid_rows == []


def test_excluded_judgment_is_rejected(tmp_path):
    res = load(tmp_path, [row()], exclude_ids={"9000001"})
    assert ("T1", "url") in fields(res, "error")


def test_pool_overlap_is_rejected(tmp_path):
    res = load(tmp_path, [row()], pool_texts=[CLAUSE])
    msg = next(i.message for i in res.issues if i.level == "error")
    assert "overlap" in msg


def test_duplicate_id_and_text_reject_only_the_later_row(tmp_path):
    res = load(tmp_path, [row(), row(url="https://indiankanoon.org/doc/2/")])
    assert [r.id for r in res.valid_rows] == ["T1"]
    assert res.error_ids == {"T1#2"}


def test_more_than_two_rows_from_one_judgment_warns(tmp_path):
    rows = [row(id=f"R{i}", clause_text=CLAUSE + f" variant number {i} with distinct wording here") for i in range(3)]
    res = load(tmp_path, rows)
    assert not res.error_ids
    assert any("more than 2 rows" in i.message for i in res.issues)


# --- summary and CLI --------------------------------------------------------

def test_summarize_counts(tmp_path):
    res = load(tmp_path, [
        row(id="a", outcome="unenforceable"),
        row(id="b", url="https://indiankanoon.org/doc/2/", clause_text=CLAUSE + " b", outcome="enforceable",
            sections_applied="ICA_S74", clause_type="LiquidatedDamages", judgment_date="2018"),
        row(id="c", url="https://indiankanoon.org/doc/3/", clause_text=CLAUSE + " c", sections_applied="SRA_S14"),
    ])
    s = summarize(res)
    assert s["scope"] == {"in": 2, "out": 1}
    assert s["groups"]["S.27 restraint of trade"] == 1
    assert s["groups"]["S.74 damages / penalty"] == 1
    assert s["enforceable_share_in_scope"] == 0.5
    assert s["recent_share"] == pytest.approx(2 / 3)


def test_main_exit_codes(tmp_path, capsys):
    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    write_csv(good, [row()])
    write_csv(bad, [row(outcome="maybe")])
    assert main([str(good)]) == 0
    assert main([str(bad)]) == 1
    assert main([str(tmp_path / "missing.csv")]) == 2
    assert "ERRORS" in capsys.readouterr().out
