"""
evaluation/gold_set.py
======================
Loader and validator for the hand-collected gold eval set (see data/eval/README.md).

Each row is a contract clause quoted verbatim in a court judgment, the sections the
court applied to it, and whether the court held it enforceable. This module turns the
CSV into typed rows, works out which rows the retriever index can answer ("scope"),
and refuses to let leaked or malformed rows through silently.

Scope, computed from the knowledge-base registry (never entered by hand):
    in     every court-applied section is in the index
    out    none is in the index (the system should decline to answer)
    mixed  some are -- scored for retrieval, excluded from the refusal test

Usage:
    python evaluation/gold_set.py data/eval/kanoon_gold.csv
Exit code is 1 when any row has an error, so it can gate other scripts.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "data" / "eval" / "kanoon_gold.csv"
DEFAULT_REGISTRY = ROOT / "knowledge_base" / "chunk_registry.json"
DEFAULT_TAXONOMY = ROOT / "models" / "risk_map.json"
DEFAULT_EXCLUDE = ROOT / "data" / "eval" / "exclude_kanoon_doc_ids.txt"
DEFAULT_POOL = ROOT / "data" / "processed" / "clauses.jsonl"

OUTCOMES = ("unenforceable", "enforceable", "partly_enforceable")
OTHER_TYPE = "OTHER"
REQUIRED_COLUMNS = ("url", "clause_text", "sections_applied", "outcome", "holding")

_OUTCOME_SYNONYMS = {
    "void": "unenforceable", "invalid": "unenforceable", "struck_down": "unenforceable",
    "not_enforceable": "unenforceable", "valid": "enforceable", "upheld": "enforceable",
    "partly": "partly_enforceable", "partial": "partly_enforceable",
    "read_down": "partly_enforceable", "narrowed": "partly_enforceable",
}
_NOT_AN_ACT_PREFIX = {"SECTION", "SEC", "SECT", "SECTIONS"}
_SECTION_RE = re.compile(
    r"^([A-Za-z]{2,8})[\s_.\-]*(?:S(?:ec(?:tion)?)?\.?)?[\s_.\-]*(\d+[A-Za-z]{0,2})(?:\s*\(.*\))*$",
    re.IGNORECASE,
)
_DOC_ID_RE = re.compile(r"indiankanoon\.org/(?:doc|docfragment)/(\d+)")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")

# Progress targets: (minimum, target). See data/eval/README.md.
TARGET_TOTAL = (30, 60)
TARGET_OUT_OF_SCOPE = (6, 12)
TARGET_PER_GROUP = (3, 5)
TARGET_ENFORCEABLE_SHARE = (0.30, 0.40)
TARGET_RECENT_SHARE = (1 / 3, 1 / 2)
SECTION_GROUPS = {
    "S.27 restraint of trade": ["ICA_S27"],
    "S.74 damages / penalty": ["ICA_S74"],
    "S.73 breach / termination": ["ICA_S73"],
    "S.56 force majeure": ["ICA_S56"],
    "S.124-125 indemnity": ["ICA_S124", "ICA_S125"],
    "S.28 arbitration / jurisdiction": ["ICA_S28", "ARB_S8", "ARB_S11"],
    "S.55 payment / time": ["ICA_S55"],
    "S.62 novation / renewal": ["ICA_S62"],
}
TYPE_GROUPS = {"liability limits": ["LiabilityCap", "UncappedLiability"]}
POOL_OVERLAP_ERROR = 0.8


class GoldSetError(Exception):
    pass


@dataclass
class Issue:
    row_id: str
    level: str  # "error" | "warn"
    field: str
    message: str


@dataclass
class GoldRow:
    id: str
    url: str
    doc_id: str
    case_name: str
    court: str
    judgment_date: str
    clause_text: str
    clause_type: str
    sections_applied: list[str]
    outcome: str
    holding: str
    risk_facts: list[str]
    source_para: str
    notes: str
    index_sections: list[str] = field(default_factory=list)
    scope: str = "out"

    @property
    def year(self) -> int | None:
        m = _YEAR_RE.search(self.judgment_date or "")
        return int(m.group()) if m else None


@dataclass
class GoldLoad:
    rows: list[GoldRow]
    issues: list[Issue]

    @property
    def error_ids(self) -> set[str]:
        return {i.row_id for i in self.issues if i.level == "error"}

    @property
    def valid_rows(self) -> list[GoldRow]:
        bad = self.error_ids
        return [r for r in self.rows if r.id not in bad]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def normalize_section(raw: str) -> str:
    """'ICA s.27', 'ICA-27', 'ICA_S27(1)' -> 'ICA_S27'. Raises ValueError with a hint."""
    token = raw.strip()
    m = _SECTION_RE.match(token)
    if not m or m.group(1).upper() in _NOT_AN_ACT_PREFIX:
        raise ValueError(f"'{token}' needs an act prefix, like ICA_S27 or ARB_S34")
    return f"{m.group(1).upper()}_S{m.group(2).upper()}"


def parse_sections(raw: str) -> tuple[list[str], list[str]]:
    sections: list[str] = []
    errors: list[str] = []
    for token in re.split(r"[;,\n]", raw or ""):
        if not token.strip():
            continue
        try:
            sec = normalize_section(token)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if sec not in sections:
            sections.append(sec)
    return sections, errors


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _max_overlap(text: str, pool_shingles: list[set]) -> float:
    mine = _shingles(text)
    if not mine:
        return 0.0
    return max((len(mine & other) / len(mine) for other in pool_shingles), default=0.0)


# ---------------------------------------------------------------------------
# Default context (registry, taxonomy, exclude list, training pool)
# ---------------------------------------------------------------------------

def _read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")}


def _read_pool(path: Path) -> list[str]:
    if not path.exists():
        return []
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            texts.append(json.loads(line).get("clause_text", ""))
    return texts


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_gold(
    path: str | Path,
    registry_ids: set[str] | None = None,
    taxonomy: set[str] | None = None,
    exclude_ids: set[str] | None = None,
    pool_texts: list[str] | None = None,
) -> GoldLoad:
    """Parse and validate a gold CSV. Any context argument left as None loads the repo default."""
    if registry_ids is None:
        registry_ids = set(json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8")))
    if taxonomy is None:
        taxonomy = set(json.loads(DEFAULT_TAXONOMY.read_text(encoding="utf-8")))
    if exclude_ids is None:
        exclude_ids = _read_ids(DEFAULT_EXCLUDE)
    if pool_texts is None:
        pool_texts = _read_pool(DEFAULT_POOL)
    pool_shingles = [_shingles(t) for t in pool_texts if t]

    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise GoldSetError(f"{path}: empty file")
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise GoldSetError(f"{path}: missing required column(s): {', '.join(missing)}")
        raw_rows = list(reader)

    rows: list[GoldRow] = []
    issues: list[Issue] = []
    seen_ids: set[str] = set()
    dup_count: dict[str, int] = {}
    seen_text: dict[str, str] = {}
    per_doc: Counter = Counter()

    def get(rec: dict, key: str) -> str:
        return (rec.get(key) or "").strip()

    for n, rec in enumerate(raw_rows, start=1):
        if not any((v or "").strip() for v in rec.values() if isinstance(v, str)):
            continue  # blank spreadsheet row

        rid = get(rec, "id")
        if not rid:
            rid = f"ROW-{n}"
            issues.append(Issue(rid, "warn", "id", "blank, auto-assigned"))
        duplicate_id = rid in seen_ids
        if duplicate_id:
            # Give the later row its own key so only it is rejected, not the first one.
            dup_count[rid] = dup_count.get(rid, 1) + 1
            rid = f"{rid}#{dup_count[rid]}"
        seen_ids.add(rid)

        def err(fld: str, msg: str) -> None:
            issues.append(Issue(rid, "error", fld, msg))

        def warn(fld: str, msg: str) -> None:
            issues.append(Issue(rid, "warn", fld, msg))

        if duplicate_id:
            err("id", "duplicate of an earlier row's id")

        url = get(rec, "url")
        m = _DOC_ID_RE.search(url)
        doc_id = m.group(1) if m else ""
        if not doc_id:
            err("url", "must be an indiankanoon.org/doc/<number>/ link")
        elif doc_id in exclude_ids:
            err("url", f"judgment {doc_id} was already used for the classifier data (see exclude list)")
        if doc_id:
            per_doc[doc_id] += 1
            if per_doc[doc_id] == 3:
                warn("url", f"more than 2 rows from judgment {doc_id}")

        clause_text = get(rec, "clause_text")
        if len(clause_text) < 20:
            err("clause_text", "missing or too short (copy the clause word for word)")
        else:
            key = _norm_text(clause_text)
            if key in seen_text:
                err("clause_text", f"same clause text as {seen_text[key]}")
            else:
                seen_text[key] = rid
            overlap = _max_overlap(clause_text, pool_shingles)
            if overlap >= POOL_OVERLAP_ERROR:
                err("clause_text", f"{overlap:.0%} overlap with data/processed/clauses.jsonl (training/reference pool)")
            if len(clause_text.split()) > 250:
                warn("clause_text", "over 250 words, trim to the operative clause")

        sections, sec_errors = parse_sections(get(rec, "sections_applied"))
        for msg in sec_errors:
            err("sections_applied", msg)
        if not sections and not sec_errors:
            err("sections_applied", "empty: record the section(s) the court applied")

        outcome_raw = get(rec, "outcome")
        outcome = re.sub(r"[\s\-]+", "_", outcome_raw.lower())
        if outcome in _OUTCOME_SYNONYMS:
            warn("outcome", f"'{outcome_raw}' read as '{_OUTCOME_SYNONYMS[outcome]}'")
            outcome = _OUTCOME_SYNONYMS[outcome]
        if outcome not in OUTCOMES:
            err("outcome", f"'{outcome_raw}' is not one of {', '.join(OUTCOMES)}")

        holding = get(rec, "holding")
        if not holding:
            err("holding", "empty: one or two sentences on what the court held")

        clause_type = get(rec, "clause_type")
        if not clause_type:
            warn("clause_type", "blank, to be filled")
        elif clause_type != OTHER_TYPE and clause_type not in taxonomy:
            close = difflib.get_close_matches(clause_type, sorted(taxonomy), n=2, cutoff=0.5)
            hint = f" (did you mean {' or '.join(close)}?)" if close else ""
            warn("clause_type", f"'{clause_type}' is not in the 47-type taxonomy{hint}")

        risk_facts = [f.strip() for f in get(rec, "risk_facts").split("||") if f.strip()]
        if not risk_facts:
            warn("risk_facts", "blank, to be drafted from the holding")
        elif len(risk_facts) > 3:
            warn("risk_facts", "more than 3 facts, keep the 1-3 that matter most")

        judgment_date = get(rec, "judgment_date")
        if not _YEAR_RE.search(judgment_date):
            warn("judgment_date", "no year found (needed to check the recent-judgments share)")
        for col in ("case_name", "court", "source_para"):
            if not get(rec, col):
                warn(col, "blank")

        index_sections = [s for s in sections if s in registry_ids]
        if sections and len(index_sections) == len(sections):
            scope = "in"
        elif index_sections:
            scope = "mixed"
        else:
            scope = "out"

        rows.append(GoldRow(
            id=rid, url=url, doc_id=doc_id, case_name=get(rec, "case_name"), court=get(rec, "court"),
            judgment_date=judgment_date, clause_text=clause_text, clause_type=clause_type,
            sections_applied=sections, outcome=outcome, holding=holding, risk_facts=risk_facts,
            source_para=get(rec, "source_para"), notes=get(rec, "notes"),
            index_sections=index_sections, scope=scope,
        ))

    return GoldLoad(rows=rows, issues=issues)


# ---------------------------------------------------------------------------
# Summary and progress report
# ---------------------------------------------------------------------------

def summarize(load: GoldLoad) -> dict:
    valid = load.valid_rows
    scored = [r for r in valid if r.scope in ("in", "mixed")]
    in_rows = [r for r in valid if r.scope == "in"]
    section_counts = Counter(s for r in scored for s in set(r.index_sections))
    groups = {name: sum(1 for r in scored if set(r.index_sections) & set(secs))
              for name, secs in SECTION_GROUPS.items()}
    for name, types in TYPE_GROUPS.items():
        groups[name] = sum(1 for r in scored if r.clause_type in types)
    dated = [r for r in valid if r.year]
    return {
        "rows_total": len(load.rows),
        "rows_valid": len(valid),
        "rows_with_errors": len(load.error_ids),
        "scope": dict(Counter(r.scope for r in valid)),
        "outcomes": dict(Counter(r.outcome for r in valid)),
        "enforceable_share_in_scope": (
            sum(1 for r in in_rows if r.outcome != "unenforceable") / len(in_rows) if in_rows else None),
        "recent_share": (sum(1 for r in dated if r.year >= 2024) / len(dated)) if dated else None,
        "groups": groups,
        "section_counts": dict(section_counts.most_common()),
    }


def _mark(have: float, target: tuple[float, float]) -> str:
    if have >= target[1]:
        return "ok (target met)"
    if have >= target[0]:
        return "ok (minimum met)"
    return "below minimum"


def print_report(load: GoldLoad, path: Path) -> None:
    s = summarize(load)
    print(f"\nGold set: {path}")
    print(f"  rows read: {s['rows_total']}   usable: {s['rows_valid']}   with errors: {s['rows_with_errors']}")
    errors = [i for i in load.issues if i.level == "error"]
    if errors:
        print("\nERRORS (row is not scored until fixed):")
        for i in errors:
            print(f"  {i.row_id:<10} {i.field:<16} {i.message}")
    warns = [i for i in load.issues if i.level == "warn"]
    if warns:
        print("\nWarnings:")
        by_msg: dict[tuple[str, str], list[str]] = {}
        for i in warns:
            by_msg.setdefault((i.field, i.message), []).append(i.row_id)
        for (fld, msg), ids in by_msg.items():
            if len(ids) > 5:
                print(f"  {fld:<16} {msg}  ({len(ids)} rows, e.g. {', '.join(ids[:3])})")
            else:
                for rid in ids:
                    print(f"  {rid:<10} {fld:<16} {msg}")

    out = s["scope"].get("out", 0)
    print("\nProgress vs targets (minimum / target):")
    print(f"  {'usable rows':<34} {s['rows_valid']:>3}  ({TARGET_TOTAL[0]} / {TARGET_TOTAL[1]})  {_mark(s['rows_valid'], TARGET_TOTAL)}")
    print(f"  {'out-of-scope rows':<34} {out:>3}  ({TARGET_OUT_OF_SCOPE[0]} / {TARGET_OUT_OF_SCOPE[1]})  {_mark(out, TARGET_OUT_OF_SCOPE)}")
    for name, count in s["groups"].items():
        print(f"  {name:<34} {count:>3}  ({TARGET_PER_GROUP[0]} / {TARGET_PER_GROUP[1]})  {_mark(count, TARGET_PER_GROUP)}")
    share = s["enforceable_share_in_scope"]
    if share is not None:
        print(f"  {'enforceable / partly, in-scope':<34} {share:>4.0%}  ({TARGET_ENFORCEABLE_SHARE[0]:.0%} / {TARGET_ENFORCEABLE_SHARE[1]:.0%})  {_mark(share, TARGET_ENFORCEABLE_SHARE)}")
    if s["recent_share"] is not None:
        print(f"  {'judgments dated 2024+':<34} {s['recent_share']:>4.0%}  ({TARGET_RECENT_SHARE[0]:.0%} / {TARGET_RECENT_SHARE[1]:.0%})  {_mark(s['recent_share'], TARGET_RECENT_SHARE)}")
    print(f"\nScope: {s['scope']}   Outcomes: {s['outcomes']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the gold eval CSV and show progress.")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args(argv)
    try:
        load = load_gold(args.path)
    except (GoldSetError, FileNotFoundError) as exc:
        print(f"Cannot read gold set: {exc}", file=sys.stderr)
        return 2
    print_report(load, args.path)
    return 1 if load.error_ids else 0


if __name__ == "__main__":
    sys.exit(main())
