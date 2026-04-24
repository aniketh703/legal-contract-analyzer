import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "clauses.jsonl"
RAW_DIR = ROOT / "data" / "raw"

TARGET_PER_TYPE = 30

CLAUSE_TYPE_MAP = {
    "termination":      "Termination",
    "indemnification":  "Indemnification",
    "non_compete":      "NonCompete",
    "arbitration":      "Arbitration",
    "force_majeure":    "ForceMajeure",
    "jurisdiction":     "Jurisdiction",
    "payment_terms":    "PaymentTerms",
    "liability_cap":    "LiabilityCap",
    "ip_assignment":    "IPAssignment",
    "confidentiality":  "Confidentiality",
    "governing_law":    "GoverningLaw",
    "renewal":          "Renewal",
}

RISK_MAP = {
    "Termination":     "MEDIUM",
    "Indemnification": "HIGH",
    "NonCompete":      "HIGH",
    "Arbitration":     "MEDIUM",
    "ForceMajeure":    "MEDIUM",
    "Jurisdiction":    "MEDIUM",
    "PaymentTerms":    "LOW",
    "LiabilityCap":    "HIGH",
    "IPAssignment":    "HIGH",
    "Confidentiality": "MEDIUM",
    "GoverningLaw":    "LOW",
    "Renewal":         "LOW",
}

# Keywords that signal court discourse — reject if found
_COURT_NOISE = [
    "this court", "it is submitted", "it was observed",
    "petitioner", "respondent", "appellant", "held that",
    "signature not verified", "digitally signed",
    "as observed by", "learned arbitrator", "learned counsel",
    "issue no.", "supra", "analysis and findings",
    "it may be noted", "supreme court", "high court",
]

# Per-type anchor keywords — at least one must appear for quality_ok
_TYPE_ANCHORS: dict[str, list[str]] = {
    "Termination":     [r"\bterminat\w*\b"],
    "Indemnification": [r"\bindemnif\w*\b", r"\bhold harmless\b"],
    "NonCompete":      [r"\bnon.?compet\w*\b", r"\brestraint of trade\b", r"\bcompeting business\b"],
    "Arbitration":     [r"\barbitrat\w*\b", r"\bconciliation\b"],
    "ForceMajeure":    [r"\bforce majeure\b", r"\bbeyond.*control\b", r"\bact of god\b"],
    "Jurisdiction":    [r"\bjurisdiction\b", r"\bcourts? at\b"],
    "PaymentTerms":    [r"\bpayment\b", r"\binvoice\b", r"\bfee\b"],
    "LiabilityCap":    [r"\bliabilit\w*\b", r"\blimit\w*\b", r"\bcap\b"],
    "IPAssignment":    [r"\bintellectual property\b", r"\bassign\w*\b", r"\bcopyright\b", r"\bpatent\b"],
    "Confidentiality": [r"\bconfidential\w*\b", r"\bnon.?disclosure\b"],
    "GoverningLaw":    [r"\bgoverning law\b", r"\bgoverned by\b", r"\bconstrued\b"],
    "Renewal":         [r"\brenew\w*\b", r"\bauto.?renew\w*\b", r"\bextension\b"],
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def quality_ok(text: str, clause_type: str = "") -> bool:
    t = norm(text)
    if len(t) < 80 or len(t) > 2000:
        return False
    if any(b in t for b in _COURT_NOISE):
        return False
    # Must contain at least one contractual party/document word
    if not re.search(r"\b(agreement|contract|deed|clause|party|parties|lessee|lessor|seller|buyer|licensor|licensee|employer|employee|vendor|client|service provider)\b", t):
        return False
    # Must contain at least one obligation/modal word
    if not re.search(r"\b(shall|may|must|will|liable|terminate|indemn|arbitrat|governed|jurisdiction|payment|confidential|renew|assign|disclose)\b", t):
        return False
    # Type-specific anchor check — at least one anchor must fire
    anchors = _TYPE_ANCHORS.get(clause_type, [])
    if anchors and not any(re.search(p, t) for p in anchors):
        return False
    return True


def load_existing() -> list[dict]:
    if not PROCESSED.exists():
        return []
    rows = []
    for line in PROCESSED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ---------------------------------------------------------------------------
# Extraction patterns — three complementary patterns
# ---------------------------------------------------------------------------

# Pattern 1: numbered sub-clauses  "17.2 Heading\n body..."
_PAT_NUMBERED = re.compile(
    r'(\d{1,2}\.\d{1,3}\.?\s+[A-Za-z][^\n]{0,140})\n((?:[^\n]{20,300}\n){1,15})',
    re.MULTILINE,
)

# Pattern 2: quoted block in double/curly quotes spanning 80–1500 chars
_PAT_QUOTED = re.compile(
    r'["""\'\']((?:[^"""\'\']{80,1500}))["""\'\']',
    re.DOTALL,
)

# Pattern 3: paragraph starting with a clause keyword anchor
_PAT_KEYWORD = re.compile(
    r'(?:^|\n\n)((?:(?:termination|indemnif|non.compete|arbitration|force majeure|'
    r'jurisdiction|payment|liabilit|intellectual property|confidential|governing law|renewal)'
    r'[^\n]{0,120}\n(?:[^\n]{20,300}\n){1,10}))',
    re.IGNORECASE | re.MULTILINE,
)


def _extract_candidates(text: str) -> list[str]:
    """Pull candidate clause texts using all three patterns."""
    seen_norms: set[str] = set()
    candidates: list[str] = []

    def _add(raw: str) -> None:
        cleaned = re.sub(r"\s+", " ", raw).strip()
        n = norm(cleaned)
        if n not in seen_norms and len(cleaned) >= 80:
            seen_norms.add(n)
            candidates.append(cleaned[:1500])

    for m in _PAT_NUMBERED.finditer(text):
        _add(m.group(1) + " " + m.group(2))
    for m in _PAT_QUOTED.finditer(text):
        _add(m.group(1))
    for m in _PAT_KEYWORD.finditer(text):
        _add(m.group(1))

    return candidates


def mine_from_raw(existing_norms: set[str], counts: Counter) -> list[dict]:
    """Extract clauses from raw files, filling gaps per clause type."""
    mined: list[dict] = []

    for raw_file in sorted(RAW_DIR.glob("*.json")):
        try:
            doc = json.loads(raw_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        clause_type_raw = str(doc.get("clause_type", "unknown"))
        mapped_type = CLAUSE_TYPE_MAP.get(clause_type_raw, clause_type_raw)

        # Skip if this type already has enough
        if counts.get(mapped_type, 0) >= TARGET_PER_TYPE:
            continue

        text = doc.get("full_text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        local_count = 0
        for candidate in _extract_candidates(text):
            if counts.get(mapped_type, 0) >= TARGET_PER_TYPE:
                break
            n = norm(candidate)
            if n in existing_norms:
                continue
            if not quality_ok(candidate, mapped_type):
                continue

            row = {
                "contract_id": f"{clause_type_raw}_{doc.get('doc_id','x')}_{local_count}",
                "clause_text": candidate[:1200],
                "clause_type": mapped_type,
                "risk_level": RISK_MAP.get(mapped_type, "MEDIUM"),
                "statute_reference": "",
                "source_case": doc.get("title", ""),
                "source_url": doc.get("url", ""),
                "source_date": doc.get("date", ""),
                "extraction_method": "auto_from_raw",
                "verified": False,
            }
            mined.append(row)
            existing_norms.add(n)
            counts[mapped_type] = counts.get(mapped_type, 0) + 1
            local_count += 1

    return mined


def main():
    # Load and re-validate existing clauses
    existing = load_existing()
    clean: list[dict] = []
    seen: set[str] = set()
    counts: Counter = Counter()

    for r in existing:
        text = r.get("clause_text", "")
        ctype = r.get("clause_type", "")
        if not isinstance(text, str):
            continue
        n = norm(text)
        if n in seen:
            continue
        if not quality_ok(text, ctype):
            continue
        seen.add(n)
        clean.append(r)
        counts[ctype] += 1

    print(f"Existing valid clauses: {len(clean)}")

    # Mine more from raw files to fill gaps
    new_rows = mine_from_raw(seen, counts)
    clean.extend(new_rows)

    # Write output
    out = "\n".join(json.dumps(r, ensure_ascii=False) for r in clean) + "\n"
    PROCESSED.write_text(out, encoding="utf-8")

    # Report
    final_counts = Counter(r["clause_type"] for r in clean)
    print(f"\nFinal clause counts ({len(clean)} total):")
    for t in sorted(CLAUSE_TYPE_MAP.values()):
        c = final_counts.get(t, 0)
        gap = max(0, TARGET_PER_TYPE - c)
        status = "OK" if gap == 0 else f"still need +{gap}"
        print(f"  {t:<18} {c:>3}/{TARGET_PER_TYPE}  {status}")
    print(f"\nWrote {len(clean)} clauses to {PROCESSED}")


if __name__ == "__main__":
    main()
