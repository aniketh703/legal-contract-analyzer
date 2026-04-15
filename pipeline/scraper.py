"""
pipeline/scraper.py
===================
Scrapes Indian Kanoon for commercial dispute judgments containing
verbatim contract clause text, and extracts those clauses.

Week 2 target: 50 clause snippets across all 12 clause types,
saved as JSON into data/raw/ and data/processed/.

Usage:
    # Scrape all clause types (slow — ~2 hrs for 50 results)
    python pipeline/scraper.py --mode all --max-per-type 5

    # Scrape one clause type only (good for testing)
    python pipeline/scraper.py --mode single --clause-type termination --max 5

    # Resume interrupted scrape (skips already-saved cases)
    python pipeline/scraper.py --mode all --max-per-type 5 --resume

Output files:
    data/raw/{clause_type}_{doc_id}.json       ← full judgment text + metadata
    data/processed/clauses.jsonl               ← extracted clause snippets only
    data/processed/scrape_log.json             ← what was scraped, when, status

IMPORTANT — Rate limiting:
    Indian Kanoon is a free public service. This scraper uses 2-second delays
    between requests. Do NOT reduce this. If you get blocked, email
    contact@indiankanoon.org citing MTech research use.
"""

import json
import os
import re
import time
import argparse
import hashlib
import logging
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/processed/scrape_log.txt", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# ── Search queries per clause type ────────────────────────────────────────────
# These are tuned based on your manual verification above.
# Each query is designed to surface judgments that QUOTE the clause verbatim.

SEARCH_QUERIES = {
    "termination": [
        '"vendor agreement" termination clause',
        '"termination clause" vendor agreement dispute',
        '"service agreement" termination notice breach',
    ],
    "indemnification": [
        '"service contract" indemnification clause',
        '"indemnify and hold harmless" contract',
        '"indemnification clause" commercial agreement',
    ],
    "non_compete": [
        '"restraint of trade" section 27 contract',
        '"non-compete" clause employment agreement section 27',
        '"shall not engage" competing business contract',
    ],
    "arbitration": [
        '"arbitration clause" agreement dispute',
        '"law and arbitration" clause contract',
        '"arbitration agreement" section 7 commercial',
    ],
    "force_majeure": [
        '"force majeure" clause contract',
        '"beyond reasonable control" clause agreement',
        '"force majeure" means contract dispute',
    ],
    "jurisdiction": [
        '"jurisdiction clause" agreement courts',
        '"exclusive jurisdiction" contract dispute',
        '"subject to jurisdiction" commercial contract',
    ],
    "payment_terms": [
        '"payment terms" clause contract dispute',
        '"invoice" "payment within" days contract',
        '"delayed payment" clause commercial agreement',
    ],
    "liability_cap": [
        '"limitation of liability" clause contract',
        '"liability shall not exceed" agreement',
        '"cap on liability" commercial contract',
    ],
    "ip_assignment": [
        '"intellectual property" assignment clause contract',
        '"IP assignment" agreement technology',
        '"copyright assignment" clause service agreement',
    ],
    "confidentiality": [
        '"confidentiality clause" NDA agreement',
        '"non-disclosure" clause commercial contract',
        '"confidential information" clause agreement breach',
    ],
    "governing_law": [
        '"governing law" clause contract India',
        '"this agreement shall be governed" Indian law',
        '"applicable law" clause commercial agreement',
    ],
    "renewal": [
        '"renewal clause" contract agreement',
        '"automatic renewal" commercial agreement',
        '"term and renewal" clause contract dispute',
    ],
}

# ── Risk level defaults per clause type (from your annotation schema) ─────────
DEFAULT_RISK = {
    "termination": "MEDIUM",
    "indemnification": "HIGH",
    "non_compete": "HIGH",
    "arbitration": "MEDIUM",
    "force_majeure": "MEDIUM",
    "jurisdiction": "MEDIUM",
    "payment_terms": "LOW",
    "liability_cap": "HIGH",
    "ip_assignment": "HIGH",
    "confidentiality": "MEDIUM",
    "governing_law": "LOW",
    "renewal": "LOW",
}

# Map scraper clause type names to your 12-label taxonomy
CLAUSE_TYPE_MAP = {
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
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MTech-Research-Bot/1.0; "
        "VNIT-Nagpur-Legal-NLP-Research; contact: your-email@students.vnit.ac.in)"
    )
}

BASE_URL = "https://indiankanoon.org"
SEARCH_URL = f"{BASE_URL}/search/"
DELAY_SECONDS = 2.5   # Be a good citizen — don't reduce this


# ── Core scraping functions ───────────────────────────────────────────────────

def search_indiankanoon(query: str, page: int = 0) -> list[dict]:
    """
    Search Indian Kanoon and return list of {title, url, snippet} dicts.
    Page 0 = first 10 results, page 1 = next 10, etc.
    """
    params = {"formInput": f"{query} doctypes:judgments", "pagenum": page}
    try:
        resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Search failed for '{query}': {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    results = []
    seen_doc_ids = set()

    def _append_result(title_tag, snippet_text=""):
        href = (title_tag.get("href", "") if title_tag else "").strip()
        if not href.startswith("/doc/"):
            return
        doc_id = href.strip("/").split("/")[-1]
        if not doc_id or doc_id in seen_doc_ids:
            return
        seen_doc_ids.add(doc_id)
        results.append({
            "title": title_tag.get_text(strip=True) if title_tag else "",
            "url": BASE_URL + href,
            "doc_id": doc_id,
            "snippet": snippet_text.strip(),
        })

    # Preferred structure: article.result with docfragment title + full-document link
    for item in soup.select("article.result, div.result"):
        title_tag = item.select_one("h4.result_title a") or item.select_one("a.title_text")
        full_doc_tag = item.select_one("a[href^='/doc/']")
        snippet_tag = item.select_one("div.headline") or item.select_one("div.snippet") or item.select_one("p")

        if full_doc_tag and title_tag:
            original_href = title_tag.get("href", "")
            title_tag["href"] = full_doc_tag.get("href", "")
            _append_result(title_tag, snippet_tag.get_text(strip=True) if snippet_tag else "")
            title_tag["href"] = original_href
        else:
            _append_result(full_doc_tag or title_tag, snippet_tag.get_text(strip=True) if snippet_tag else "")

    # Fallback for newer markup: collect all doc links and use nearby text
    if not results:
        for a_tag in soup.select("a[href^='/doc/']"):
            container = a_tag.find_parent(["div", "article", "li"]) or a_tag.parent
            container_text = container.get_text(" ", strip=True) if container else ""
            title_text = a_tag.get_text(strip=True)
            snippet = container_text.replace(title_text, "", 1).strip()
            _append_result(a_tag, snippet)

    # If quoted query yields nothing, retry once with quotes removed.
    if not results and '"' in query:
        simplified_query = query.replace('"', "")
        params = {"formInput": f"{simplified_query} doctypes:judgments", "pagenum": page}
        try:
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for a_tag in soup.select("a[href^='/doc/']"):
                container = a_tag.find_parent(["div", "article", "li"]) or a_tag.parent
                container_text = container.get_text(" ", strip=True) if container else ""
                title_text = a_tag.get_text(strip=True)
                snippet = container_text.replace(title_text, "", 1).strip()
                _append_result(a_tag, snippet)
            log.info(f"  Retry without quotes for '{query[:50]}' -> {len(results)} results")
        except requests.RequestException as e:
            log.error(f"Search retry failed for '{simplified_query}': {e}")

    log.info(f"  Search '{query[:50]}' page {page}: {len(results)} results")
    return results


def fetch_judgment(url: str) -> dict | None:
    """
    Fetch a single judgment page and return its full text + metadata.
    Returns None on failure.
    """
    time.sleep(DELAY_SECONDS)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"  Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Extract judgment text (main content div)
    judgment_div = (
        soup.select_one("div#judgments") or
        soup.select_one("div.judgment") or
        soup.select_one("div#doc_content") or
        soup.select_one("div.doc_content")
    )

    if not judgment_div:
        # Fallback: get all paragraph text
        paragraphs = soup.select("p")
        full_text = "\n".join(p.get_text() for p in paragraphs)
    else:
        # Remove watermark/signature noise frequently injected in scanned orders.
        for noisy in judgment_div.select("span.hidden_text"):
            noisy.decompose()
        full_text = judgment_div.get_text(separator="\n")

    # Extract metadata
    title = ""
    title_tag = soup.select_one("h2.doc_title") or soup.select_one("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    date = ""
    date_tag = soup.select_one("span.docsource_main") or soup.select_one("div.doc_date")
    if date_tag:
        date = date_tag.get_text(strip=True)

    return {
        "url": url,
        "title": title,
        "date": date,
        "full_text": full_text.strip(),
        "scraped_at": datetime.now().isoformat(),
    }


def extract_clause_snippets(judgment: dict, clause_type: str) -> list[dict]:
    """
    Extract verbatim contract clause text from a judgment.

    Strategy: Find paragraphs that contain quoted clause text.
    Indian Kanoon judgments quote clauses in patterns like:
      - "Clause X reads as follows: '...'"
      - "The relevant clause is reproduced below: ..."
      - Direct block quotes following "CLAUSES OF VENDOR AGREEMENT"
    """
    text = judgment["full_text"]
    snippets = []

    # Pattern 1: quoted clause blocks around explicit "reads/states/reproduced" cues
    clause_intro_patterns = [
        r'[Cc]lause\s+\d+[\.\-—]?\s*(?:[A-Z][a-z]+\s+)*(?:reads?|states?|provides?|is reproduced|is as follows|is quoted)[^\n]*\n((?:[^\n]+\n){1,8})',
        r'[Rr]elevant\s+clause[s]?\s+(?:is|are)\s+reproduced[^\n]*\n((?:[^\n]+\n){2,10})',
        r'[Tt]he\s+(?:said|relevant|aforesaid)\s+clause[^\n]*[:\-]\s*["\u201c]([^""\u201d]{50,500})["\u201d]',
        r'(?:reads?\s+as\s+follows|is\s+as\s+under)[:\s]*["\u201c\n]([^""\u201d]{50,600})',
        r'Clause\s+\d+(?:\.\d+)?[^\n]*\s+states?\s+that[:\s]*\n((?:[^\n]+\n){2,14})',
        r'relevant\s+portion\s+reads[^\n]*[:\s]*\n((?:[^\n]+\n){2,12})',
    ]

    def looks_like_contract_clause(snippet_text: str) -> bool:
        t = re.sub(r"\s+", " ", snippet_text).strip()
        lower = t.lower()
        if len(t) < 60:
            return False
        # Contract clauses usually carry a clause marker + operative verbs.
        if not re.search(r"\b(clause|agreement|contract)\b", lower):
            return False
        if not re.search(r"\b(shall|may|will|must|means|including|provided that|liable|termination|indemnif|arbitrat|governed by|jurisdiction)\b", lower):
            return False
        if len(re.findall(r"\b(clause|agreement|contract|lease|deed)\b", lower)) < 1:
            return False
        # Drop likely OCR/appendix chunks with overwhelmingly uppercase text.
        letters = [c for c in t if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio > 0.45:
                return False
        if re.search(r"\b(section|article)\s+\d+\b", lower) and "agreement" not in lower and "contract" not in lower:
            return False
        if re.search(r"\b(supreme court|high court|learned counsel|petitioner|respondent|appellant|tribunal)\b", lower):
            return False
        if re.search(r"\b(cites|cited by|judgment|decree|plaintiff|defendant)\b", lower):
            return False
        if re.search(r"\b(it is submitted|this court|it is contended|heard learned counsel)\b", lower):
            return False
        if re.search(r"\b(thanksfor|thanks for sending|proposed agreement once)\b", lower):
            return False
        return True

    # Pattern 0: explicit numbered clause blocks (e.g., "15.2 Force Majeure - ...")
    numbered_clause_pattern = r'["\u201c]?\s*(\d{1,2}\.\d{1,3}\s+[A-Za-z][^\n]{0,120})\n((?:[^\n]{20,220}\n){2,18})'
    for match in re.finditer(numbered_clause_pattern, text, re.MULTILINE):
        snippet_text = (match.group(1) + "\n" + match.group(2)).strip()
        snippet_text = re.sub(r'\s+', ' ', snippet_text)
        if len(snippet_text) < 80:
            continue
        if len(snippet_text) > 900:
            snippet_text = snippet_text[:900] + "..."
        if not looks_like_contract_clause(snippet_text):
            continue
        if any(s["clause_text"][:50] == snippet_text[:50] for s in snippets):
            continue
        snippets.append({
            "contract_id": f"{clause_type}_{judgment.get('doc_id', hashlib.md5(judgment['url'].encode()).hexdigest()[:8])}_{len(snippets)}",
            "clause_text": snippet_text,
            "clause_type": CLAUSE_TYPE_MAP.get(clause_type, clause_type),
            "risk_level": DEFAULT_RISK.get(clause_type, "MEDIUM"),
            "statute_reference": "",
            "source_case": judgment["title"],
            "source_url": judgment["url"],
            "source_date": judgment["date"],
            "extraction_method": "numbered_clause_block",
            "verified": False,
        })

    for pattern in clause_intro_patterns:
        for match in re.finditer(pattern, text, re.MULTILINE):
            snippet_text = match.group(1).strip()
            snippet_text = re.sub(r'\s+', ' ', snippet_text)

            # Quality filters
            if len(snippet_text) < 40:
                continue
            if len(snippet_text) > 800:
                snippet_text = snippet_text[:800] + "..."
            if not looks_like_contract_clause(snippet_text):
                continue

            # Avoid duplicates within this judgment
            if any(s["clause_text"][:50] == snippet_text[:50] for s in snippets):
                continue

            snippets.append({
                "contract_id": f"{clause_type}_{judgment.get('doc_id', hashlib.md5(judgment['url'].encode()).hexdigest()[:8])}_{len(snippets)}",
                "clause_text": snippet_text,
                "clause_type": CLAUSE_TYPE_MAP.get(clause_type, clause_type),
                "risk_level": DEFAULT_RISK.get(clause_type, "MEDIUM"),  # preliminary — override in Label Studio
                "statute_reference": "",  # fill during annotation in Label Studio
                "source_case": judgment["title"],
                "source_url": judgment["url"],
                "source_date": judgment["date"],
                "extraction_method": "regex_pattern",
                "verified": False,  # set to True after Label Studio annotation
            })

    # Pattern 2: Look for indented/quoted blocks near clause-type keywords
    type_keywords = {
        "termination": [r'terminat', r'notice period', r'termination clause'],
        "indemnification": [r'indemnif', r'hold harmless', r'indemnity clause'],
        "non_compete": [r'restrain', r'not engage', r'not compete', r'non.compete'],
        "arbitration": [r'arbitrat', r'arbitration clause', r'seat of arbitration'],
        "force_majeure": [r'force majeure', r'beyond.*control', r'act of god'],
        "jurisdiction": [r'jurisdiction', r'courts of', r'subject to.*jurisdiction'],
        "payment_terms": [r'payment.*days', r'invoice.*due', r'payment terms'],
        "liability_cap": [r'liability.*shall not exceed', r'limitation.*liability', r'cap.*liability'],
        "ip_assignment": [r'intellectual property', r'IP.*assign', r'copyright.*assign'],
        "confidentiality": [r'confidential', r'non.disclosure', r'proprietary information'],
        "governing_law": [r'governed by', r'applicable law', r'governing law'],
        "renewal": [r'renew', r'auto.*renewal', r'term.*renew'],
    }

    keywords = type_keywords.get(clause_type, [])
    paragraphs = text.split('\n')

    for i, para in enumerate(paragraphs):
        if len(para.strip()) < 50:
            continue
        if not any(re.search(kw, para, re.IGNORECASE) for kw in keywords):
            continue

        # Check if this paragraph looks like quoted clause text (not analysis)
        # Heuristic: starts with quote mark, or is in a block after "reads as follows"
        is_quoted = (
            para.strip().startswith(('"', "'", '\u201c', '\u2018')) or
            (i > 0 and re.search(r'reads?|quoted|reproduced|extracted', paragraphs[i-1], re.IGNORECASE))
        )

        if not is_quoted:
            continue

        snippet_text = para.strip().strip('"\'"\u201c\u201d').strip()
        if len(snippet_text) < 40:
            continue
        if not looks_like_contract_clause(snippet_text):
            continue
        if any(s["clause_text"][:50] == snippet_text[:50] for s in snippets):
            continue

        snippets.append({
            "contract_id": f"{clause_type}_{hashlib.md5(judgment['url'].encode()).hexdigest()[:8]}_{len(snippets)}",
            "clause_text": snippet_text[:800],
            "clause_type": CLAUSE_TYPE_MAP.get(clause_type, clause_type),
            "risk_level": DEFAULT_RISK.get(clause_type, "MEDIUM"),
            "statute_reference": "",
            "source_case": judgment["title"],
            "source_url": judgment["url"],
            "source_date": judgment["date"],
            "extraction_method": "keyword_proximity",
            "verified": False,
        })

    return snippets


# ── Orchestration ─────────────────────────────────────────────────────────────

def scrape_clause_type(clause_type: str, max_results: int = 5, resume: bool = False) -> list[dict]:
    """
    Scrape Indian Kanoon for one clause type.
    Returns list of extracted clause snippets.
    """
    log.info(f"\n{'='*50}")
    log.info(f"Scraping clause type: {clause_type} (target: {max_results} snippets)")
    log.info(f"{'='*50}")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_snippets = []
    queries = SEARCH_QUERIES.get(clause_type, [])

    for query in queries:
        if len(all_snippets) >= max_results:
            break

        log.info(f"\nQuery: {query}")
        time.sleep(DELAY_SECONDS)
        results = search_indiankanoon(query, page=0)

        if not results:
            log.warning(f"  No results for query: {query}")
            continue

        for result in results:
            if len(all_snippets) >= max_results:
                break

            doc_id = result["doc_id"]
            raw_path = raw_dir / f"{clause_type}_{doc_id}.json"

            # Resume: skip already-scraped judgments
            if resume and raw_path.exists():
                log.info(f"  Skipping (already scraped): {doc_id}")
                # Still load snippets from saved file
                with open(raw_path, encoding="utf-8") as f:
                    saved = json.load(f)
                all_snippets.extend(saved.get("extracted_snippets", []))
                continue

            log.info(f"  Fetching: {result['title'][:60]}")
            judgment = fetch_judgment(result["url"])

            if not judgment:
                continue

            judgment["doc_id"] = doc_id
            snippets = extract_clause_snippets(judgment, clause_type)
            judgment["extracted_snippets"] = snippets
            judgment["clause_type"] = clause_type

            # Save raw judgment
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(judgment, f, indent=2, ensure_ascii=False)

            log.info(f"    Extracted {len(snippets)} snippet(s) from this judgment")
            all_snippets.extend(snippets)

            time.sleep(DELAY_SECONDS)

    log.info(f"\nTotal snippets for {clause_type}: {len(all_snippets)}")
    return all_snippets


def save_processed_clauses(all_snippets: list[dict]):
    """Append extracted snippets to data/processed/clauses.jsonl"""
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "clauses.jsonl"

    with open(output_path, "a", encoding="utf-8") as f:
        for snippet in all_snippets:
            f.write(json.dumps(snippet, ensure_ascii=False) + "\n")

    log.info(f"Saved {len(all_snippets)} snippets -> {output_path}")


def print_summary(all_snippets: list[dict]):
    """Print a table of how many snippets per clause type."""
    from collections import Counter
    counts = Counter(s["clause_type"] for s in all_snippets)

    print("\n" + "="*50)
    print("SCRAPING SUMMARY")
    print("="*50)
    print(f"{'Clause Type':<20} {'Snippets':>8}")
    print(f"{'-'*20} {'-'*8}")
    for clause_type in CLAUSE_TYPE_MAP.values():
        count = counts.get(clause_type, 0)
        bar = "#" * count
        status = "OK" if count >= 3 else "LOW - scrape more"
        print(f"{clause_type:<20} {count:>8}  {bar}  {'' if count >= 3 else status}")
    print(f"{'TOTAL':<20} {sum(counts.values()):>8}")
    print("="*50)
    print("\nNext step: Load data/processed/clauses.jsonl into Label Studio")
    print("Then annotate: confirm clause_type, set risk_level, add statute_reference")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Indian Kanoon contract clause scraper")
    parser.add_argument("--mode", choices=["all", "single"], default="single",
                        help="Scrape all clause types or just one")
    parser.add_argument("--clause-type", choices=list(SEARCH_QUERIES.keys()),
                        default="termination", help="Clause type (for --mode single)")
    parser.add_argument("--max", type=int, default=5,
                        help="Max snippets per clause type (for --mode single)")
    parser.add_argument("--max-per-type", type=int, default=5,
                        help="Max snippets per clause type (for --mode all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-scraped judgments")
    args = parser.parse_args()

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    all_snippets = []

    if args.mode == "single":
        snippets = scrape_clause_type(args.clause_type, max_results=args.max, resume=args.resume)
        all_snippets.extend(snippets)
        save_processed_clauses(snippets)
        print_summary(all_snippets)

    elif args.mode == "all":
        for clause_type in SEARCH_QUERIES.keys():
            snippets = scrape_clause_type(clause_type, max_results=args.max_per_type, resume=args.resume)
            all_snippets.extend(snippets)
            save_processed_clauses(snippets)
            # Pause between clause types to avoid rate limiting
            log.info("Pausing 10 seconds between clause types...")
            time.sleep(10)

        print_summary(all_snippets)

    log.info("\nDone. Review data/processed/clauses.jsonl before loading into Label Studio.")
    log.info("Manual review is important — regex extraction is not perfect.")


if __name__ == "__main__":
    main()
