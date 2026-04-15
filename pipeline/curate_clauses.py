import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "clauses.jsonl"
RAW_DIR = ROOT / "data" / "raw"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def quality_ok(text: str) -> bool:
    t = norm(text)
    if len(t) < 80:
        return False
    if len(t) > 1400:
        return False
    bad = [
        "this court", "it is submitted", "it was observed", "the court",
        "petitioner", "respondent", "appellant", "judgment", "scc", "held that",
        "signature not verified", "digitally signed", "http://", "https://",
        "as observed by", "learned arbitrator", "issue no.", "supra", "analysis and findings",
        "there is no", "it may be noted", "for consideration",
    ]
    if any(b in t for b in bad):
        return False
    # Keep drafting-like text (usually numbered clauses or direct contract language).
    starts_numbered = bool(re.match(r"^\d{1,2}\.\d{1,3}\b", t))
    contract_phrases = bool(re.search(r"\b(this agreement|under this agreement|the parties|party shall|shall not|in the event|provided that)\b", t))
    if not starts_numbered and not contract_phrases:
        return False
    # Prefer contractual language.
    if not re.search(r"\b(agreement|contract|deed|clause|party|parties|lessee|lessor|seller|buyer|licensor|licensee)\b", t):
        return False
    if not re.search(r"\b(shall|may|must|will|liable|terminate|termination|indemn|arbitrat|governed|jurisdiction|payment|confidential|renew)\b", t):
        return False
    # Drop statute-heavy prose unless clearly contractual.
    if "section " in t and "this agreement" not in t and "under this agreement" not in t and "clause" not in t:
        return False
    return True


def load_existing():
    rows = []
    if not PROCESSED.exists():
        return rows
    for line in PROCESSED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def mine_from_raw(existing_norms: set[str], target_count: int):
    mined = []
    # Capture numbered clause blocks likely verbatim from judgments.
    block_pat = re.compile(r'["“”]?\s*(\d{1,2}\.\d{1,3}\s+[A-Za-z][^\n]{0,140})\n((?:[^\n]{20,220}\n){2,20})', re.MULTILINE)
    for raw_file in sorted(RAW_DIR.glob("*.json")):
        if len(mined) >= target_count:
            break
        try:
            doc = json.loads(raw_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        clause_type_raw = str(doc.get("clause_type", "unknown"))
        mapped_type = {
            "termination": "Termination",
            "indemnification": "Indemnification",
            "non_compete": "NonCompete",
            "arbitration": "Arbitration",
            "force_majeure": "ForceMajeure",
            "jurisdiction": "Jurisdiction",
            "payment_terms": "PaymentTerms",
            "liability_cap": "LiabilityCap",
            "ip_assignment": "IPAssignment",
            "confidentiality": "Confidentiality",
            "governing_law": "GoverningLaw",
            "renewal": "Renewal",
        }.get(clause_type_raw, clause_type_raw)
        text = doc.get("full_text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        local_count = 0
        for m in block_pat.finditer(text):
            clause_text = re.sub(r"\s+", " ", (m.group(1) + "\n" + m.group(2)).strip())
            n = norm(clause_text)
            if n in existing_norms:
                continue
            if not quality_ok(clause_text):
                continue
            row = {
                "contract_id": f"manual_{clause_type_raw}_{doc.get('doc_id','x')}_{local_count}",
                "clause_text": clause_text[:1000],
                "clause_type": mapped_type,
                "risk_level": {
                    "Termination": "MEDIUM",
                    "Indemnification": "HIGH",
                    "NonCompete": "HIGH",
                    "Arbitration": "MEDIUM",
                    "ForceMajeure": "MEDIUM",
                    "Jurisdiction": "MEDIUM",
                    "PaymentTerms": "LOW",
                    "LiabilityCap": "HIGH",
                    "IPAssignment": "HIGH",
                    "Confidentiality": "MEDIUM",
                    "GoverningLaw": "LOW",
                    "Renewal": "LOW",
                }.get(mapped_type, "MEDIUM"),
                "statute_reference": "",
                "source_case": doc.get("title", ""),
                "source_url": doc.get("url", ""),
                "source_date": doc.get("date", ""),
                "extraction_method": "manual_from_raw",
                "verified": False,
            }
            mined.append(row)
            existing_norms.add(n)
            local_count += 1
            if local_count >= 2:
                break
            if len(mined) >= target_count:
                break
    return mined


def main():
    rows = load_existing()
    clean = []
    seen = set()
    for r in rows:
        text = r.get("clause_text", "")
        if not isinstance(text, str):
            continue
        n = norm(text)
        if n in seen:
            continue
        if not quality_ok(text):
            continue
        seen.add(n)
        clean.append(r)

    if len(clean) < 50:
        clean.extend(mine_from_raw(seen, 50 - len(clean)))

    # Keep exactly 50 high-quality snippets.
    clean = clean[:50]
    out = "\n".join(json.dumps(r, ensure_ascii=False) for r in clean) + "\n"
    PROCESSED.write_text(out, encoding="utf-8")
    print(f"Wrote {len(clean)} curated rows to {PROCESSED}")


if __name__ == "__main__":
    main()
