"""
tests/test_llm_generator.py
===========================
Unit tests for pipeline/llm_generator.py with scripted fake LLM replies. No network.
"""
import json

import pytest

from pipeline.llm_client import FakeClient, LLMError
from pipeline.llm_generator import (
    OutputError, build_messages, generate_clause_analysis, parse_output, retrieval_support,
)

S27 = {
    "section_id": "ICA_S27", "title": "Agreement in restraint of trade, void", "score": 0.0328,
    "source": "statute_map",
    "text": "Section 27. Agreement in restraint of trade, void. Every agreement by which any one is "
            "restrained from exercising a lawful profession, trade or business of any kind, is to that extent void.",
}
S73 = {
    "section_id": "ICA_S73", "title": "Compensation for loss or damage caused by breach of contract",
    "score": 0.0161, "source": "bm25",
    "text": "Section 73. Compensation for loss or damage caused by breach of contract. When a contract has "
            "been broken, the party who suffers by such breach is entitled to receive compensation.",
}
CLAUSE = "The Employee shall not join any competitor for two years after leaving the Company."


def answer(risks, risk_level="high", enforceability="likely_unenforceable", action="Delete the clause."):
    return json.dumps({"status": "answered", "risk_level": risk_level, "enforceability": enforceability,
                       "risks": risks, "recommended_action": action})


def risk(cits, quote, statement="A post-employment restraint is void."):
    return {"statement": statement, "citations": cits, "evidence_quote": quote}


GOOD = answer([risk(["ICA_S27"], "is restrained from exercising a lawful profession, trade or business")])


# --- gate -------------------------------------------------------------------

def test_gate_passes_on_statute_map_section():
    assert retrieval_support([S27, S73])["passes"]


def test_gate_passes_when_retrievers_agree():
    agreed = dict(S73, score=0.0300, source="faiss")   # ranked by two lists
    assert retrieval_support([agreed])["passes"]


def test_gate_refuses_single_weak_list():
    lone = dict(S73, score=0.0164)   # 1/61 rounded: rank 1 in one list only
    sup = retrieval_support([lone])
    assert not sup["agreement"] and not sup["passes"]
    assert not retrieval_support([])["passes"]


def test_gate_refusal_never_calls_the_model():
    client = FakeClient(responses=[])
    res = generate_clause_analysis(CLAUSE, "Unknown", [dict(S73, score=0.0164)], client)
    assert res.refused and res.checks["refused_by_gate"] == 1
    assert client.calls == [] and res.cost_usd == 0.0


# --- prompt -----------------------------------------------------------------

def test_prompt_carries_clause_and_sections():
    msgs = build_messages(CLAUSE, "NonCompete", [S27, S73])
    assert msgs[0]["role"] == "system" and "ONLY" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "[ICA_S27] Agreement in restraint of trade, void" in user
    assert "[ICA_S73]" in user and CLAUSE in user and "NonCompete" in user


# --- parsing ----------------------------------------------------------------

def test_parse_accepts_fenced_json():
    assert parse_output(f"```json\n{GOOD}\n```")["status"] == "answered"


@pytest.mark.parametrize("bad,msg", [
    ("not json", "not valid JSON"),
    ("[]", "JSON object"),
    (json.dumps({"status": "maybe"}), '"status"'),
    (json.dumps({"status": "refused"}), "reason"),
    (answer([risk(["ICA_S27"], "x")], risk_level="severe"), "risk_level"),
    (answer([risk(["ICA_S27"], "x")], enforceability="void"), "enforceability"),
    (answer([]), "risks"),
    (answer([risk([], "x")]), "citations"),
    (answer([risk(["ICA_S27"], " ")]), "evidence_quote"),
])
def test_parse_rejects_malformed(bad, msg):
    with pytest.raises(OutputError, match=msg):
        parse_output(bad)


# --- end to end with a scripted model ---------------------------------------

def test_happy_path():
    res = generate_clause_analysis(CLAUSE, "NonCompete", [S27, S73], FakeClient(responses=[GOOD]))
    assert res.status == "answered" and res.risk_level == "high"
    assert res.cited == ["ICA_S27"] and res.raw_cited == ["ICA_S27"]
    assert res.checks["llm_calls"] == 1 and res.checks["risks_dropped"] == 0
    assert "ICA_S27" in res.answer_text() and "Delete the clause." in res.answer_text()


def test_model_refusal_is_recorded():
    reply = json.dumps({"status": "refused", "reason": "Sections do not cover IP assignment."})
    res = generate_clause_analysis(CLAUSE, "NonCompete", [S27], FakeClient(responses=[reply]))
    assert res.refused and res.checks["refused_by_model"] == 1
    assert "IP assignment" in res.refusal_reason


def test_invalid_then_repaired():
    client = FakeClient(responses=["oops", GOOD])
    res = generate_clause_analysis(CLAUSE, "NonCompete", [S27], client)
    assert res.status == "answered" and res.checks["repair_attempts"] == 1 and res.checks["llm_calls"] == 2
    repair_turn = client.calls[1]
    assert repair_turn[-2] == {"role": "assistant", "content": "oops"}
    assert "invalid" in repair_turn[-1]["content"]


def test_invalid_twice_refuses():
    res = generate_clause_analysis(CLAUSE, "NonCompete", [S27], FakeClient(responses=["oops", "still oops"]))
    assert res.refused and res.refusal_reason == "invalid_output" and res.checks["invalid_output"] == 1


def test_unsupported_citation_is_counted_and_dropped():
    reply = answer([
        risk(["ICA_S27"], "is restrained from exercising a lawful profession"),
        risk(["SRA_S14"], "is restrained from exercising", statement="Specific performance is barred."),
    ])
    res = generate_clause_analysis(CLAUSE, "NonCompete", [S27, S73], FakeClient(responses=[reply]))
    assert res.status == "answered"
    assert res.raw_cited == ["ICA_S27", "SRA_S14"]   # the scorer sees what the model cited
    assert res.cited == ["ICA_S27"]                  # the user sees only grounded risks
    assert res.checks["citations_unsupported"] == 1 and res.checks["risks_dropped"] == 1


def test_ungrounded_quote_is_dropped():
    reply = answer([
        risk(["ICA_S27"], "is restrained from exercising a lawful profession"),
        risk(["ICA_S73"], "non-compete clauses are always void in India"),   # not in S.73's text
    ])
    res = generate_clause_analysis(CLAUSE, "NonCompete", [S27, S73], FakeClient(responses=[reply]))
    assert len(res.risks) == 1 and res.checks["quotes_ungrounded"] == 1


def test_quote_match_ignores_case_whitespace_and_curly_quotes():
    sec = dict(S27, text=S27["text"].replace("is to that extent void", "is “to that extent” void"))
    reply = answer([risk(["ICA_S27"], 'IS "TO THAT   EXTENT" VOID.')])
    res = generate_clause_analysis(CLAUSE, "NonCompete", [sec], FakeClient(responses=[reply]))
    assert res.status == "answered" and res.checks["quotes_ungrounded"] == 0


def test_all_risks_ungrounded_refuses():
    reply = answer([risk(["ICA_S27"], "invented words")])
    res = generate_clause_analysis(CLAUSE, "NonCompete", [S27], FakeClient(responses=[reply]))
    assert res.refused and res.checks["refused_ungrounded"] == 1
    assert res.raw_cited == ["ICA_S27"]


def test_llm_error_becomes_refusal():
    class Broken(FakeClient):
        def complete(self, messages, json_mode=True, max_tokens=1024):
            raise LLMError("HTTP 403")

    res = generate_clause_analysis(CLAUSE, "NonCompete", [S27], Broken())
    assert res.refused and res.checks["llm_errors"] == 1 and "403" in res.refusal_reason


def test_cost_from_usage(monkeypatch):
    monkeypatch.setenv("LCA_LLM_PRICE_IN", "0.15")
    monkeypatch.setenv("LCA_LLM_PRICE_OUT", "0.60")
    res = generate_clause_analysis(CLAUSE, "NonCompete", [S27], FakeClient(responses=[GOOD]))
    expected = (res.input_tokens * 0.15 + res.output_tokens * 0.60) / 1_000_000
    assert res.input_tokens > 0 and res.cost_usd == pytest.approx(expected)


def test_echo_fake_produces_a_grounded_answer():
    res = generate_clause_analysis(CLAUSE, "NonCompete", [S27, S73], FakeClient())
    assert res.status == "answered" and res.cited == ["ICA_S27"]
