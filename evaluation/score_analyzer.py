"""
evaluation/score_analyzer.py
============================
Scores any clause analyzer against the gold eval set (evaluation/gold_set.py).

An analyzer takes one clause's text and returns an AnalysisResult. v1 (today's
classify -> retrieve -> template pipeline) is wrapped by V1Analyzer; v2 (same classify ->
retrieve, then the LLM generator in pipeline/llm_generator.py) by V2Analyzer. Both share
the classify/retrieve step, so retrieval metrics are identical and only the generator differs.

Automatic metrics (this file):
    retrieval     hit@k (any court-applied section in top-k), recall@k, MRR   [scope in/mixed]
    type          predicted clause type == gold type
    citations     precision vs the court-applied sections; unsupported-citation rate
                  (a cited section that was not in the retrieved context)
    refusal       out-of-scope rows should be refused; in-scope rows should not be
    cost/latency  p50 / p95 per clause (wall clock, model load excluded), USD
    checks        v2 only: summed generator self-checks (gate / model refusals, invalid JSON,
                  dropped risks, ungrounded quotes), from AnalysisResult.extra["checks"]

Not automatic: faithfulness, missed risk facts, contradicting the court. Those need a human
reading the output, so --review-sheet writes a stratified sample with blank check columns
(definitions in data/eval/README.md).

Usage:
    python evaluation/score_analyzer.py --gold data/eval/kanoon_gold.csv
    python evaluation/score_analyzer.py --gold ... --no-embeddings --review-sheet out.csv
    python evaluation/score_analyzer.py --analyzer v2 --gold ...            # Gemma 4 on Vertex AI
    python evaluation/score_analyzer.py --analyzer v2 --llm-backend fake    # dry run, no credentials
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.gold_set import DEFAULT_GOLD, GoldRow, GoldSetError, OTHER_TYPE, load_gold, print_report

RESULTS_DIR = ROOT / "evaluation" / "results"
DEFAULT_KS = (1, 3, 5)


# ---------------------------------------------------------------------------
# Analyzer interface
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    clause_type: str = ""
    risk_level: str = ""
    retrieved: list[str] = field(default_factory=list)   # ranked section ids given to the generator
    retrieved_scores: list[float] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)       # section ids shown to the user as citations
    answer_text: str = ""
    refused: bool = False
    latency_s: float = 0.0                               # set by the scorer, not the analyzer
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


class Analyzer(Protocol):
    name: str

    def analyze(self, clause_text: str) -> AnalysisResult: ...


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


class V1Analyzer:
    """Today's pipeline for one clause: classify -> retrieve -> template explanation."""

    name = "v1_template"

    def __init__(self, use_embeddings: bool = True, top_k: int = 5):
        from pipeline.classifier import classify_clauses
        from pipeline.generator import _ACTIONS, _build_explanation
        from pipeline.retriever import ARTIFACT_CACHE, retrieve_statutes

        self.use_embeddings = use_embeddings
        self.top_k = top_k
        self._classify = classify_clauses
        self._retrieve = retrieve_statutes
        self._explain = _build_explanation
        self._actions = _ACTIONS
        self._cache = ARTIFACT_CACHE

    def config(self) -> dict:
        return {"use_embeddings": self.use_embeddings, "top_k": self.top_k}

    def warmup(self) -> None:
        self.analyze("This Agreement shall be governed by the laws of India.")

    def classify_and_retrieve(self, clause_text: str) -> dict:
        clause = {"clause_id": "eval", "clause_text": clause_text}
        classified = self._classify([clause], use_embeddings=self.use_embeddings)
        return self._retrieve(classified, top_k=self.top_k, cache=self._cache)[0]

    def analyze(self, clause_text: str) -> AnalysisResult:
        enriched = self.classify_and_retrieve(clause_text)
        sections = enriched.get("retrieved_sections", [])
        ctype = enriched.get("clause_type", "Unknown")
        explanation = _strip_html(self._explain(ctype, sections))
        action = _strip_html(self._actions.get(ctype, self._actions["Unknown"]))
        ids = [s["section_id"] for s in sections]
        # The report lists every retrieved section as a "Relevant ICA Statutes" pill,
        # so all of them count as citations; the explanation text names at most one.
        return AnalysisResult(
            clause_type=ctype,
            risk_level=enriched.get("risk_level", ""),
            retrieved=ids,
            retrieved_scores=[float(s.get("score", 0.0)) for s in sections],
            cited=ids,
            answer_text=f"{explanation} Recommended action: {action}",
            extra={
                "method": enriched.get("method"),
                "confidence": enriched.get("confidence"),
                "explained_sections": [f"ICA_S{n}" for n in re.findall(r"ICA Section (\d+)", explanation)],
            },
        )


class V2Analyzer(V1Analyzer):
    """v1's classify -> retrieve, then the LLM generator (structured JSON, citations, refusal)."""

    def __init__(self, use_embeddings: bool = True, top_k: int = 5, client=None,
                 backend: str | None = None, model: str | None = None):
        super().__init__(use_embeddings=use_embeddings, top_k=top_k)
        from pipeline.llm_client import client_from_env
        from pipeline.llm_generator import generate_clause_analysis

        self.client = client or client_from_env(backend, model)
        self._generate = generate_clause_analysis
        self.name = f"v2_{self.client.model.split('/')[-1]}"

    def config(self) -> dict:
        return dict(super().config(), backend=self.client.backend, model=self.client.model)

    def warmup(self) -> None:
        self.classify_and_retrieve("This Agreement shall be governed by the laws of India.")  # no LLM spend

    def analyze(self, clause_text: str) -> AnalysisResult:
        enriched = self.classify_and_retrieve(clause_text)
        sections = enriched.get("retrieved_sections", [])
        ctype = enriched.get("clause_type", "Unknown")
        gen = self._generate(clause_text, ctype, sections, self.client)
        return AnalysisResult(
            clause_type=ctype,
            risk_level=gen.risk_level or enriched.get("risk_level", ""),
            retrieved=[s["section_id"] for s in sections],
            retrieved_scores=[float(s.get("score", 0.0)) for s in sections],
            # Raw citations, before ungrounded risks are dropped, so the unsupported-citation
            # rate measures the model, not the filter.
            cited=gen.raw_cited,
            answer_text=gen.answer_text(),
            refused=gen.refused,
            cost_usd=gen.cost_usd or 0.0,
            extra={
                "checks": gen.checks,
                "enforceability": gen.enforceability,
                "refusal_reason": gen.refusal_reason,
                "shown_citations": gen.cited,
                "tokens": {"input": gen.input_tokens, "output": gen.output_tokens},
                "cost_known": gen.cost_usd is not None,
            },
        )


ANALYZERS = {
    "v1": lambda args: V1Analyzer(use_embeddings=not args.no_embeddings),
    "v2": lambda args: V2Analyzer(use_embeddings=not args.no_embeddings,
                                  backend=args.llm_backend, model=args.llm_model),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def row_record(row: GoldRow, res: AnalysisResult, ks: tuple[int, ...]) -> dict:
    gold = set(row.index_sections)
    rank = next((i + 1 for i, s in enumerate(res.retrieved) if s in gold), None)
    rec = {
        "id": row.id, "scope": row.scope, "outcome": row.outcome, "gold_type": row.clause_type,
        "gold_sections": row.sections_applied, "predicted_type": res.clause_type,
        "retrieved": res.retrieved, "cited": res.cited,
        "top_score": res.retrieved_scores[0] if res.retrieved_scores else None,
        "refused": res.refused, "latency_ms": round(res.latency_s * 1000, 2), "cost_usd": res.cost_usd,
        "first_hit_rank": rank,
    }
    if row.scope in ("in", "mixed"):
        for k in ks:
            top = set(res.retrieved[:k])
            rec[f"hit@{k}"] = bool(gold & top)
            rec[f"recall@{k}"] = len(gold & top) / len(gold)
    return rec


def _retrieval_summary(records: list[dict], ks: tuple[int, ...]) -> dict:
    out: dict = {"n": len(records)}
    for k in ks:
        out[f"hit@{k}"] = _mean([float(r[f"hit@{k}"]) for r in records])
        out[f"recall@{k}"] = _mean([r[f"recall@{k}"] for r in records])
    out["mrr"] = _mean([1 / r["first_hit_rank"] if r["first_hit_rank"] else 0.0 for r in records])
    return out


def score_results(rows: list[GoldRow], results: list[AnalysisResult], ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    records = [row_record(r, res, ks) for r, res in zip(rows, results)]
    scored = [(r, res, rec) for r, res, rec in zip(rows, results, records) if r.scope in ("in", "mixed")]
    in_pairs = [(r, res) for r, res in zip(rows, results) if r.scope == "in"]
    out_pairs = [(r, res) for r, res in zip(rows, results) if r.scope == "out"]

    typed = [(r, res) for r, res in zip(rows, results) if r.clause_type and r.clause_type != OTHER_TYPE]

    precisions = []
    for r, res, _ in scored:
        if res.cited:
            precisions.append(len(set(res.cited) & set(r.sections_applied)) / len(set(res.cited)))

    n_cited = sum(len(res.cited) for res in results)
    n_unsupported = sum(len([c for c in res.cited if c not in res.retrieved]) for res in results)
    rows_unsupported = sum(1 for res in results if any(c not in res.retrieved for c in res.cited))

    latencies = [res.latency_s * 1000 for res in results]
    checks: dict[str, float] = {}
    for res in results:
        for key, value in (res.extra.get("checks") or {}).items():
            if isinstance(value, (int, float)):
                checks[key] = checks.get(key, 0) + value
    top_in = [rec["top_score"] for r, _, rec in scored if r.scope == "in" and rec["top_score"] is not None]
    top_out = [rec["top_score"] for r, rec in zip(rows, records) if r.scope == "out" and rec["top_score"] is not None]

    def grouped(key) -> dict:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for _, _, rec in scored:
            buckets[key(rec)].append(rec)
        return {k: _retrieval_summary(v, ks) for k, v in sorted(buckets.items())}

    return {
        "counts": {"rows": len(rows), "in_scope": len(in_pairs), "mixed": len(scored) - len(in_pairs),
                   "out_of_scope": len(out_pairs)},
        "retrieval": _retrieval_summary([rec for _, _, rec in scored], ks) if scored else None,
        "type_accuracy": _mean([float(res.clause_type == r.clause_type) for r, res in typed]),
        "citations": {
            "precision_vs_court": _mean(precisions),
            "rows_without_citations": sum(1 for res in results if not res.cited),
            "unsupported_rate": (n_unsupported / n_cited) if n_cited else None,
            "rows_with_unsupported": rows_unsupported / len(rows) if rows else None,
        },
        "refusal": {
            "refused_out_of_scope": _mean([float(res.refused) for _, res in out_pairs]),
            "answered_out_of_scope": _mean([float(not res.refused) for _, res in out_pairs]),
            "false_refusal_in_scope": _mean([float(res.refused) for _, res in in_pairs]),
        },
        "retrieval_confidence": {"mean_top_score_in_scope": _mean(top_in), "mean_top_score_out_of_scope": _mean(top_out)},
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "cost_usd": {"total": sum(res.cost_usd for res in results),
                     "per_clause": _mean([res.cost_usd for res in results])},
        **({"checks": checks} if checks else {}),
        "by_outcome": grouped(lambda rec: rec["outcome"]),
        "by_gold_type": grouped(lambda rec: rec["gold_type"] or "(blank)"),
        "rows": records,
    }


# ---------------------------------------------------------------------------
# Hand-review sheet (Step 4 failure-type checks)
# ---------------------------------------------------------------------------

REVIEW_CHECKS = [
    "faithful (Y/N)",
    "missed_risk_facts (count)",
    "unsupported_citation (Y/N)",
    "contradicts_court (Y/N)",
    "overflags_valid_clause (Y/N)",
    "reviewer_notes",
]


def stratified_sample(rows: list[GoldRow], n: int, seed: int = 0) -> list[int]:
    """Indices of up to n rows, balanced across (scope, outcome) groups, deterministic for a seed."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.scope, r.outcome)].append(i)
    queues = []
    for key in sorted(groups):
        idx = groups[key]
        rng.shuffle(idx)
        queues.append(idx)
    picked: list[int] = []
    while len(picked) < min(n, len(rows)):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop())
    return sorted(picked)


def write_review_sheet(path: Path, rows: list[GoldRow], results: list[AnalysisResult],
                       analyzer_name: str, n: int, seed: int) -> int:
    idx = stratified_sample(rows, n, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["id", "scope", "outcome_gold", "sections_applied_gold", "clause_type_gold", "clause_text",
              "holding_gold", "risk_facts_gold", "analyzer", "answer_text", "cited_sections",
              "retrieved_sections", "refused"] + REVIEW_CHECKS
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i in idx:
            r, res = rows[i], results[i]
            writer.writerow([
                r.id, r.scope, r.outcome, ";".join(r.sections_applied), r.clause_type, r.clause_text,
                r.holding, " || ".join(r.risk_facts), analyzer_name, res.answer_text,
                ";".join(res.cited), ";".join(res.retrieved), res.refused,
            ] + [""] * len(REVIEW_CHECKS))
    return len(idx)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_analyzer(analyzer: Analyzer, rows: list[GoldRow]) -> list[AnalysisResult]:
    if hasattr(analyzer, "warmup"):
        analyzer.warmup()  # keep model/index load out of the latency numbers
    results = []
    for row in rows:
        t0 = time.perf_counter()
        res = analyzer.analyze(row.clause_text)
        res.latency_s = time.perf_counter() - t0
        results.append(res)
    return results


def _fmt(value, pct: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:.1f}"


def print_summary(name: str, m: dict, ks: tuple[int, ...]) -> None:
    c = m["counts"]
    print(f"\nAnalyzer: {name}   rows: {c['rows']} (in-scope {c['in_scope']}, mixed {c['mixed']}, out-of-scope {c['out_of_scope']})")
    if m["retrieval"]:
        for k in ks:
            print(f"  retrieval  hit@{k}: {_fmt(m['retrieval'][f'hit@{k}'])}   recall@{k}: {_fmt(m['retrieval'][f'recall@{k}'])}")
        mrr = m["retrieval"]["mrr"]
        print(f"  retrieval  MRR: {mrr:.3f}" if mrr is not None else "  retrieval  MRR: n/a")
    print(f"  clause type accuracy: {_fmt(m['type_accuracy'])}")
    ci = m["citations"]
    print(f"  citations  precision vs court: {_fmt(ci['precision_vs_court'])}   unsupported: {_fmt(ci['unsupported_rate'])}")
    rf = m["refusal"]
    print(f"  refusal    out-of-scope refused: {_fmt(rf['refused_out_of_scope'])}   in-scope wrongly refused: {_fmt(rf['false_refusal_in_scope'])}")
    lat = m["latency_ms"]
    print(f"  latency    p50 {_fmt(lat['p50'], pct=False)} ms   p95 {_fmt(lat['p95'], pct=False)} ms   cost/clause ${m['cost_usd']['per_clause'] or 0:.4f}")
    if "checks" in m:
        ck = m["checks"]
        print(f"  generator  refused by gate {ck.get('refused_by_gate', 0)}, by model {ck.get('refused_by_model', 0)}, "
              f"ungrounded {ck.get('refused_ungrounded', 0)}, invalid JSON {ck.get('invalid_output', 0)}, "
              f"LLM errors {ck.get('llm_errors', 0)}")
        print(f"  generator  risks dropped {ck.get('risks_dropped', 0)}/{ck.get('risks_total', 0)}, "
              f"ungrounded quotes {ck.get('quotes_ungrounded', 0)}, repairs {ck.get('repair_attempts', 0)}")
    print("  (faithfulness and missed risk facts need the hand-review sheet)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an analyzer against the gold eval set.")
    parser.add_argument("--analyzer", choices=sorted(ANALYZERS), default="v1")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--no-embeddings", action="store_true",
                        help="skip the InLegalBERT fallback (no torch / model download needed)")
    parser.add_argument("--llm-backend", choices=["vertex_openai", "genai", "fake"], default=None,
                        help="v2: LLM backend (default: $LCA_LLM_BACKEND or vertex_openai)")
    parser.add_argument("--llm-model", default=None, help="v2: model id (default: $LCA_LLM_MODEL or per backend)")
    parser.add_argument("--skip-invalid", action="store_true", help="score only rows without validation errors")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N rows (debugging)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--review-sheet", type=Path, default=None)
    parser.add_argument("--review-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        load = load_gold(args.gold)
    except (GoldSetError, FileNotFoundError) as exc:
        print(f"Cannot read gold set: {exc}", file=sys.stderr)
        return 2
    if load.error_ids and not args.skip_invalid:
        print_report(load, args.gold)
        print("\nFix the errors above, or pass --skip-invalid to score only the valid rows.", file=sys.stderr)
        return 2
    rows = load.valid_rows[: args.limit] if args.limit else load.valid_rows
    if not rows:
        print("No scorable rows.", file=sys.stderr)
        return 2

    from pipeline.llm_client import LLMError

    try:
        analyzer = ANALYZERS[args.analyzer](args)
    except LLMError as exc:
        print(f"Cannot set up the {args.analyzer} analyzer: {exc}", file=sys.stderr)
        return 2
    results = run_analyzer(analyzer, rows)
    metrics = score_results(rows, results)
    print_summary(analyzer.name, metrics, DEFAULT_KS)

    # v2 files carry the model name so a Gemma run and a Gemini run don't overwrite each other.
    out = args.out or RESULTS_DIR / (f"{args.analyzer}_baseline.json" if args.analyzer == "v1"
                                     else f"{analyzer.name}_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "analyzer": analyzer.name,
        "config": analyzer.config() if hasattr(analyzer, "config") else {},
        "gold_path": str(args.gold),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows_skipped_invalid": len(load.error_ids),
        "metrics": metrics,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")

    if args.review_sheet:
        n = write_review_sheet(args.review_sheet, rows, results, analyzer.name, args.review_n, args.seed)
        print(f"Review sheet ({n} rows) -> {args.review_sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
