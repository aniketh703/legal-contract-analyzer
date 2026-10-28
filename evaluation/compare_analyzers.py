"""
evaluation/compare_analyzers.py
===============================
v2 Step 4: measure v1 (template) against v2 (LLM) on the same gold rows.

Two steps, with a human in between:

    run    score both analyzers on the gold set (automatic metrics, same code as
           score_analyzer.py) and write ONE blinded hand-review sheet: the same stratified
           sample of clauses for both analyzers, each output under an anonymous review id
           (R001, R002, ...). Which analyzer wrote which output is kept in review_key.json;
           don't open it until the sheet is filled in.

    score  read the filled sheet + key and write comparison.json / comparison.md: hand-review
           pass rates with 95% Wilson intervals, paired McNemar tests (same clauses, both
           analyzers), and the automatic metrics side by side (hit@k, unsupported-citation
           rate, refusals, p50/p95 latency, cost per clause and per contract).

Blinding is partial: template and LLM outputs read differently, and v1 never refuses. It still
stops the reviewer from seeing labels, and the shuffled order within each clause pair stops
"left column = old system" habits.

Hand-review checks (definitions in data/eval/README.md):
    faithful                Y = pass     every legal claim is supported by the retrieved text or the holding
    missed_risk_facts       count        risk_facts the output does not state; in/mixed-scope rows only
    unsupported_citation    N = pass     a cited section that was not in the retrieved context
    contradicts_court       N = pass     says valid when the court held void, or the reverse
    overflags_valid_clause  N = pass     only scored on rows the court held enforceable

A refused output with blank checks is filled automatically: it claims nothing (faithful Y,
no unsupported citation, no contradiction, no over-flag) and misses every risk fact.

Usage:
    python evaluation/compare_analyzers.py run   --gold data/eval/kanoon_gold.csv --out-dir evaluation/results/compare
    python evaluation/compare_analyzers.py score --out-dir evaluation/results/compare
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.gold_set import DEFAULT_GOLD, GoldRow, GoldSetError, load_gold, print_report
from evaluation.score_analyzer import (
    ANALYZERS, AnalysisResult, DEFAULT_KS, REVIEW_CHECKS, run_analyzer, score_results, stratified_sample,
)

SHEET_NAME = "review_blind.csv"
KEY_NAME = "review_key.json"
SECTIONS_NAME = "review_sections.md"

CHECK_COLUMNS = {
    "faithful": "faithful (Y/N)",
    "missed_risk_facts": "missed_risk_facts (count)",
    "unsupported_citation": "unsupported_citation (Y/N)",
    "contradicts_court": "contradicts_court (Y/N)",
    "overflags_valid_clause": "overflags_valid_clause (Y/N)",
}
BINARY_CHECKS = ("faithful", "unsupported_citation", "contradicts_court", "overflags_valid_clause")
PASS_VALUE = {"faithful": True, "unsupported_citation": False, "contradicts_court": False,
              "overflags_valid_clause": False}
_YES = {"y", "yes", "1", "true", "t"}
_NO = {"n", "no", "0", "false", "f"}

# v2's enforceability call mapped onto the court's outcome labels (automatic, v2 only).
_ENFORCEABILITY_TO_OUTCOME = {
    "likely_unenforceable": "unenforceable",
    "partly_enforceable": "partly_enforceable",
    "likely_enforceable": "enforceable",
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value from the discordant counts (b: only A passes, c: only B)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# run: automatic metrics + blinded sheet
# ---------------------------------------------------------------------------

def outcome_agreement(rows: list[GoldRow], results: list[AnalysisResult]) -> dict | None:
    """How often v2's enforceability call matches the court. None for analyzers that don't make one."""
    calls = [(r, res.extra.get("enforceability")) for r, res in zip(rows, results)]
    if not any(c for _, c in calls):
        return None
    answered = [(r, c) for r, c in calls if c]
    decided = [(r, c) for r, c in answered if c in _ENFORCEABILITY_TO_OUTCOME]
    agree = sum(1 for r, c in decided if _ENFORCEABILITY_TO_OUTCOME[c] == r.outcome)
    return {
        "answered": len(answered),
        "unclear": len(answered) - len(decided),
        "agreement_when_decided": agree / len(decided) if decided else None,
        "n_decided": len(decided),
    }


def build_blind_rows(rows: list[GoldRow], results_by_analyzer: dict[str, list[AnalysisResult]],
                     n: int, seed: int) -> tuple[list[dict], dict]:
    """One sheet row per (sampled clause, analyzer); the two outputs of a clause sit together in random order."""
    rng = random.Random(seed)
    names = list(results_by_analyzer)
    sheet, key = [], {}
    for i in stratified_sample(rows, n, seed):
        order = names[:]
        rng.shuffle(order)
        for name in order:
            res = results_by_analyzer[name][i]
            rid = f"R{len(sheet) + 1:03d}"
            r = rows[i]
            key[rid] = {"row_id": r.id, "analyzer": name}
            sheet.append({
                "review_id": rid, "clause_id": r.id, "scope": r.scope, "outcome_gold": r.outcome,
                "sections_applied_gold": ";".join(r.sections_applied), "clause_text": r.clause_text,
                "holding_gold": r.holding,
                "risk_facts_gold": " ".join(f"{k}) {f}" for k, f in enumerate(r.risk_facts, 1)),
                "n_risk_facts": len(r.risk_facts),
                "output_text": res.answer_text, "cited_sections": ";".join(res.cited),
                "retrieved_sections": ";".join(res.retrieved), "refused": "Y" if res.refused else "N",
                **{col: "" for col in REVIEW_CHECKS},
            })
    return sheet, key


def write_sections_md(path: Path, sheet: list[dict], registry: dict) -> int:
    ids = sorted({s for row in sheet for col in ("cited_sections", "retrieved_sections")
                  for s in row[col].split(";") if s})
    lines = ["# Statute text for the review sheet", "",
             "Every section retrieved or cited in `review_blind.csv`, as the analyzers saw it.", ""]
    for sid in ids:
        entry = registry.get(sid)
        lines += [f"## {sid}", "", entry["text"] if entry else "_Not in the knowledge base (cited but never retrieved)._", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(ids)


def clauses_per_contract(paths: list[Path]) -> float | None:
    if not paths:
        return None
    from pipeline.segmenter import segment_contract

    counts = [len(segment_contract(path=p)) for p in paths]
    return sum(counts) / len(counts)


def _analyzer_args(args) -> Namespace:
    return Namespace(no_embeddings=args.no_embeddings, llm_backend=args.llm_backend, llm_model=args.llm_model)


def cmd_run(args) -> int:
    try:
        load = load_gold(args.gold)
    except (GoldSetError, FileNotFoundError) as exc:
        print(f"Cannot read gold set: {exc}", file=sys.stderr)
        return 2
    if load.error_ids and not args.skip_invalid:
        print_report(load, args.gold)
        print("\nFix the errors above, or pass --skip-invalid.", file=sys.stderr)
        return 2
    rows = load.valid_rows[: args.limit] if args.limit else load.valid_rows
    if not rows:
        print("No scorable rows.", file=sys.stderr)
        return 2

    from pipeline.llm_client import LLMError

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_by_analyzer: dict[str, list[AnalysisResult]] = {}
    reports: dict[str, str] = {}
    for key in args.analyzers:
        try:
            analyzer = ANALYZERS[key](_analyzer_args(args))
        except LLMError as exc:
            print(f"Cannot set up the {key} analyzer: {exc}", file=sys.stderr)
            return 2
        print(f"Running {analyzer.name} on {len(rows)} rows ...")
        results = run_analyzer(analyzer, rows)
        metrics = score_results(rows, results)
        metrics["outcome_agreement"] = outcome_agreement(rows, results)
        report = {
            "analyzer": analyzer.name,
            "config": analyzer.config() if hasattr(analyzer, "config") else {},
            "gold_path": str(args.gold),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows_skipped_invalid": len(load.error_ids),
            "metrics": metrics,
        }
        fname = f"{analyzer.name}.json"
        (out_dir / fname).write_text(json.dumps(report, indent=2), encoding="utf-8")
        results_by_analyzer[analyzer.name] = results
        reports[analyzer.name] = fname

    sheet, key = build_blind_rows(rows, results_by_analyzer, args.review_n, args.seed)
    header = list(sheet[0]) if sheet else []
    with open(out_dir / SHEET_NAME, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(sheet)

    registry = json.loads((ROOT / "knowledge_base" / "chunk_registry.json").read_text(encoding="utf-8"))
    n_sections = write_sections_md(out_dir / SECTIONS_NAME, sheet, registry)

    cpc = args.clauses_per_contract
    cpc_source = "--clauses-per-contract" if cpc is not None else None
    if cpc is None and args.contracts:
        cpc = clauses_per_contract(args.contracts)
        cpc_source = f"segmented {len(args.contracts)} contract(s): " + ", ".join(p.name for p in args.contracts)

    gold_bytes = Path(args.gold).read_bytes()
    manifest = {
        "gold_path": str(args.gold), "gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
        "analyzers": list(results_by_analyzer), "reports": reports,
        "review_n_clauses": len(sheet) // max(1, len(results_by_analyzer)), "seed": args.seed,
        "clauses_per_contract": cpc, "clauses_per_contract_source": cpc_source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / KEY_NAME).write_text(json.dumps({
        "note": "Do not open until review_blind.csv is filled in.", "manifest": manifest, "key": key,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote to {out_dir}/")
    for name, fname in reports.items():
        print(f"  {fname:<40} automatic metrics for {name}")
    print(f"  {SHEET_NAME:<40} {len(sheet)} outputs to review ({manifest['review_n_clauses']} clauses x {len(reports)})")
    print(f"  {SECTIONS_NAME:<40} text of the {n_sections} sections they cite or retrieved")
    print(f"  {KEY_NAME:<40} who wrote what -- don't open before reviewing")
    print(f"\nFill in the check columns, then: python evaluation/compare_analyzers.py score --out-dir {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# score: hand review + automatic metrics -> comparison
# ---------------------------------------------------------------------------

class SheetError(ValueError):
    pass


def _parse_yn(value: str, where: str) -> bool | None:
    v = value.strip().lower()
    if not v:
        return None
    if v in _YES:
        return True
    if v in _NO:
        return False
    raise SheetError(f"{where}: expected Y or N, got {value!r}")


def parse_review(sheet_rows: list[dict], key: dict) -> tuple[list[dict], list[str]]:
    """Validated per-output review records, plus a list of incomplete review ids."""
    records, incomplete = [], []
    seen: dict[str, int] = {}
    for n, row in enumerate(sheet_rows, start=2):   # row 1 is the header
        rid = row.get("review_id", "").strip()
        if rid not in key:
            raise SheetError(f"review id {rid!r} is not in the key (was the sheet edited or re-sorted with ids changed?)")
        if rid in seen:
            # A copied row would be counted twice in the pass rates and silently override the
            # other copy in the paired tests.
            raise SheetError(f"review id {rid!r} appears more than once (rows {seen[rid]} and {n}); "
                             "remove the duplicate row")
        seen[rid] = n
        refused = row.get("refused", "").strip().upper() == "Y"
        n_facts = int(row.get("n_risk_facts") or 0)
        scope, outcome = row.get("scope", ""), row.get("outcome_gold", "")
        rec = {"review_id": rid, "clause_id": key[rid]["row_id"], "analyzer": key[rid]["analyzer"],
               "scope": scope, "outcome": outcome, "refused": refused, "n_risk_facts": n_facts, "auto_filled": []}
        for check in BINARY_CHECKS:
            rec[check] = _parse_yn(row.get(CHECK_COLUMNS[check], ""), f"{rid} {check}")
        raw_missed = row.get(CHECK_COLUMNS["missed_risk_facts"], "").strip()
        if raw_missed:
            try:
                rec["missed_risk_facts"] = int(raw_missed)
            except ValueError as exc:
                raise SheetError(f"{rid} missed_risk_facts: expected a whole number, got {raw_missed!r}") from exc
            if not 0 <= rec["missed_risk_facts"] <= n_facts:
                raise SheetError(f"{rid} missed_risk_facts: {rec['missed_risk_facts']} is outside 0..{n_facts}")
        else:
            rec["missed_risk_facts"] = None

        if refused:  # a refusal claims nothing and states no risk fact
            for check in BINARY_CHECKS:
                if rec[check] is None:
                    rec[check] = PASS_VALUE[check]
                    rec["auto_filled"].append(check)
            if rec["missed_risk_facts"] is None:
                rec["missed_risk_facts"] = n_facts
                rec["auto_filled"].append("missed_risk_facts")

        needed = ["faithful", "unsupported_citation", "contradicts_court"]
        if outcome == "enforceable":
            needed.append("overflags_valid_clause")
        if scope in ("in", "mixed") and n_facts:
            needed.append("missed_risk_facts")
        if any(rec[c] is None for c in needed):
            incomplete.append(rid)
        records.append(rec)
    return records, incomplete


def _applies(rec: dict, check: str) -> bool:
    if check == "overflags_valid_clause":
        return rec["outcome"] == "enforceable"
    if check == "missed_risk_facts":
        return rec["scope"] in ("in", "mixed") and rec["n_risk_facts"] > 0
    return True


def human_metrics(records: list[dict], analyzer: str) -> dict:
    mine = [r for r in records if r["analyzer"] == analyzer]
    out: dict = {"reviewed_outputs": len(mine), "auto_filled_refusals": sum(1 for r in mine if r["auto_filled"])}
    for check in BINARY_CHECKS:
        vals = [r[check] for r in mine if _applies(r, check) and r[check] is not None]
        passes = sum(1 for v in vals if v == PASS_VALUE[check])
        ci = wilson_interval(passes, len(vals))
        out[check] = {"n": len(vals), "pass_rate": passes / len(vals) if vals else None,
                      "ci95": list(ci) if ci else None}
    facts = [r for r in mine if _applies(r, "missed_risk_facts") and r["missed_risk_facts"] is not None]
    total = sum(r["n_risk_facts"] for r in facts)
    missed = sum(r["missed_risk_facts"] for r in facts)
    complete = sum(1 for r in facts if r["missed_risk_facts"] == 0)
    out["missed_risk_facts"] = {
        "n": len(facts), "facts_total": total, "facts_missed": missed,
        "missed_rate": missed / total if total else None,
        "all_facts_stated_rate": complete / len(facts) if facts else None,
        "ci95_all_facts_stated": list(wilson_interval(complete, len(facts)) or []) or None,
    }
    return out


def paired_tests(records: list[dict], a: str, b: str) -> dict:
    by_clause: dict[str, dict[str, dict]] = {}
    for r in records:
        by_clause.setdefault(r["clause_id"], {})[r["analyzer"]] = r
    pairs = [(d[a], d[b]) for d in by_clause.values() if a in d and b in d]
    out = {}
    for check in BINARY_CHECKS + ("missed_risk_facts",):
        if check == "missed_risk_facts":
            usable = [(x, y) for x, y in pairs if _applies(x, check)
                      and x[check] is not None and y[check] is not None]
            only_a = sum(1 for x, y in usable if x[check] < y[check])   # a missed fewer facts
            only_b = sum(1 for x, y in usable if y[check] < x[check])
        else:
            usable = [(x, y) for x, y in pairs if _applies(x, check)
                      and x[check] is not None and y[check] is not None]
            only_a = sum(1 for x, y in usable if x[check] == PASS_VALUE[check] and y[check] != PASS_VALUE[check])
            only_b = sum(1 for x, y in usable if y[check] == PASS_VALUE[check] and x[check] != PASS_VALUE[check])
        out[check] = {"pairs": len(usable), f"only_{a}_better": only_a, f"only_{b}_better": only_b,
                      "p_value": mcnemar_exact(only_a, only_b)}
    return out


def automatic_summary(report: dict, cpc: float | None) -> dict:
    m = report["metrics"]
    per_clause = m["cost_usd"]["per_clause"]
    return {
        "hit@k": {f"hit@{k}": (m["retrieval"] or {}).get(f"hit@{k}") for k in DEFAULT_KS},
        "mrr": (m["retrieval"] or {}).get("mrr"),
        "unsupported_citation_rate": m["citations"]["unsupported_rate"],
        "citation_precision_vs_court": m["citations"]["precision_vs_court"],
        "refused_out_of_scope": m["refusal"]["refused_out_of_scope"],
        "false_refusal_in_scope": m["refusal"]["false_refusal_in_scope"],
        "latency_ms_p50": m["latency_ms"]["p50"], "latency_ms_p95": m["latency_ms"]["p95"],
        "cost_usd_per_clause": per_clause,
        "cost_usd_per_contract": per_clause * cpc if per_clause is not None and cpc is not None else None,
        "cost_rows_unknown": m["cost_usd"].get("rows_unknown", 0),
        "outcome_agreement": m.get("outcome_agreement"),
        "checks": m.get("checks"),
    }


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _ci(ci) -> str:
    return "" if not ci else f" ({ci[0]:.0%}–{ci[1]:.0%})"


def render_markdown(comp: dict) -> str:
    names = comp["analyzers"]
    h, a = comp["human"], comp["automatic"]
    head = "| Metric | " + " | ".join(names) + " |"
    sep = "|---|" + "---|" * len(names)
    lines = [f"# v1 vs v2 on the gold set", "",
             f"Gold: `{comp['gold_path']}` (sha256 `{comp['gold_sha256'][:12]}`), "
             f"{comp['review_n_clauses']} clauses hand-reviewed per analyzer. "
             "Ranges are 95% Wilson intervals; p-values are exact McNemar tests on the same clauses.", "",
             "## Hand review", "", head, sep]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(n) for n in names) + " |")

    labels = {"faithful": "Faithfulness pass rate", "unsupported_citation": "No unsupported citation (human)",
              "contradicts_court": "Doesn't contradict the court", "overflags_valid_clause": "Doesn't over-flag an upheld clause"}
    for check, label in labels.items():
        row(label, lambda n, c=check: f"{_pct(h[n][c]['pass_rate'])}{_ci(h[n][c]['ci95'])} (n={h[n][c]['n']})")
    row("Missed-risk rate (facts missed / facts)",
        lambda n: f"{_pct(h[n]['missed_risk_facts']['missed_rate'])} "
                  f"({h[n]['missed_risk_facts']['facts_missed']}/{h[n]['missed_risk_facts']['facts_total']})")
    row("All risk facts stated", lambda n: f"{_pct(h[n]['missed_risk_facts']['all_facts_stated_rate'])}"
                                           f"{_ci(h[n]['missed_risk_facts']['ci95_all_facts_stated'])}")

    lines += ["", "## Automatic", "", head, sep]
    for k in DEFAULT_KS:
        row(f"Hit@{k}", lambda n, k=k: _pct(a[n]["hit@k"][f"hit@{k}"]))
    row("Unsupported-citation rate", lambda n: _pct(a[n]["unsupported_citation_rate"]))
    row("Citation precision vs court", lambda n: _pct(a[n]["citation_precision_vs_court"]))
    row("Out-of-scope clauses refused", lambda n: _pct(a[n]["refused_out_of_scope"]))
    row("In-scope clauses wrongly refused", lambda n: _pct(a[n]["false_refusal_in_scope"]))
    row("Court-outcome agreement (when decided)",
        lambda n: "n/a" if not a[n]["outcome_agreement"] else
        f"{_pct(a[n]['outcome_agreement']['agreement_when_decided'])} (n={a[n]['outcome_agreement']['n_decided']})")
    row("Latency p50 / p95 per clause",
        lambda n: "n/a" if a[n]["latency_ms_p50"] is None else f"{a[n]['latency_ms_p50']:.0f} / {a[n]['latency_ms_p95']:.0f} ms")
    row("Cost per clause", lambda n: "unknown" if a[n]["cost_rows_unknown"] else f"${a[n]['cost_usd_per_clause'] or 0:.5f}")
    cpc = comp.get("clauses_per_contract")
    row(f"Cost per contract{f' (~{cpc:.0f} clauses)' if cpc else ''}",
        lambda n: "unknown" if a[n]["cost_rows_unknown"] else
        ("n/a" if a[n]["cost_usd_per_contract"] is None else f"${a[n]['cost_usd_per_contract']:.4f}"))

    if comp.get("paired"):
        x, y = names[0], names[1]
        lines += ["", f"## Paired tests ({x} vs {y}, same clauses)", "",
                  f"| Check | pairs | only {x} better | only {y} better | p |", "|---|---|---|---|---|"]
        for check, t in comp["paired"].items():
            p = "n/a" if t["p_value"] is None else f"{t['p_value']:.3f}"
            lines.append(f"| {check} | {t['pairs']} | {t[f'only_{x}_better']} | {t[f'only_{y}_better']} | {p} |")
    if comp["incomplete"]:
        lines += ["", f"**Incomplete reviews ({len(comp['incomplete'])}):** " + ", ".join(comp["incomplete"])
                  + " — these are left out of the checks they're missing."]
    return "\n".join(lines) + "\n"


def cmd_score(args) -> int:
    out_dir: Path = args.out_dir
    try:
        key_doc = json.loads((out_dir / KEY_NAME).read_text(encoding="utf-8"))
        with open(args.sheet or out_dir / SHEET_NAME, encoding="utf-8-sig", newline="") as fh:
            sheet_rows = list(csv.DictReader(fh))
    except FileNotFoundError as exc:
        print(f"Missing file: {exc.filename}. Run the `run` step first.", file=sys.stderr)
        return 2
    manifest, key = key_doc["manifest"], key_doc["key"]
    try:
        records, incomplete = parse_review(sheet_rows, key)
    except SheetError as exc:
        print(f"Review sheet problem: {exc}", file=sys.stderr)
        return 2
    missing_ids = sorted(set(key) - {r["review_id"] for r in records})
    if missing_ids:
        print(f"Review sheet is missing {len(missing_ids)} review ids (e.g. {missing_ids[:3]}).", file=sys.stderr)
        return 2

    names = manifest["analyzers"]
    cpc = args.clauses_per_contract if args.clauses_per_contract is not None else manifest.get("clauses_per_contract")
    reports = {n: json.loads((out_dir / manifest["reports"][n]).read_text(encoding="utf-8")) for n in names}
    comp = {
        "gold_path": manifest["gold_path"], "gold_sha256": manifest["gold_sha256"],
        "analyzers": names, "review_n_clauses": manifest["review_n_clauses"],
        "clauses_per_contract": cpc,
        "human": {n: human_metrics(records, n) for n in names},
        "automatic": {n: automatic_summary(reports[n], cpc) for n in names},
        "paired": paired_tests(records, names[0], names[1]) if len(names) == 2 else None,
        "incomplete": incomplete,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    md = render_markdown(comp)
    (out_dir / "comparison.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved -> {out_dir / 'comparison.json'} and comparison.md")
    if incomplete:
        print(f"Warning: {len(incomplete)} outputs have blank checks.", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v1 vs v2 comparison (v2 Step 4)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="score both analyzers and write the blinded review sheet")
    run.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    run.add_argument("--analyzers", nargs="+", default=["v1", "v2"], choices=sorted(ANALYZERS))
    run.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results" / "compare")
    run.add_argument("--review-n", type=int, default=30, help="clauses to hand-review (each reviewed once per analyzer)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-embeddings", action="store_true")
    run.add_argument("--llm-backend", choices=["vertex_openai", "genai", "fake"], default=None)
    run.add_argument("--llm-model", default=None)
    run.add_argument("--skip-invalid", action="store_true")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--contracts", type=Path, nargs="*", default=[],
                     help="contract files to segment for the clauses-per-contract figure used in cost per contract")
    run.add_argument("--clauses-per-contract", type=float, default=None)

    score = sub.add_parser("score", help="score the filled review sheet against the key")
    score.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results" / "compare")
    score.add_argument("--sheet", type=Path, default=None, help=f"default: <out-dir>/{SHEET_NAME}")
    score.add_argument("--clauses-per-contract", type=float, default=None)

    args = parser.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
