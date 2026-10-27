"""
build_knowledge_base.py
=======================
Step 4 of your Week 1 checklist.

What this script does:
  1. Extracts raw text from the ICA 1872 PDF using pdfplumber
  2. Cleans and normalises the text
  3. Segments text into section-level chunks (by "Section X" headers)
  4. Assigns deterministic IDs (ICA_S27, ICA_S73, etc.)
  5. Builds a FAISS index from InLegalBERT embeddings
  6. Saves chunk_registry.json  →  {section_id: text}
  7. Runs a PROOF-OF-CONCEPT retrieval test:
       query = "employee shall not engage in competing business activities"
       → Does ICA_S27 appear in top-5?  (It should, but may not — that's the finding)

Usage:
  python build_knowledge_base.py --pdf path/to/indian_contract_act_1872.pdf
  python build_knowledge_base.py --clean-text knowledge_base/statutes/ica_1872_clean.txt
  python build_knowledge_base.py --refresh-registry      # rewrite texts only, no torch

Requirements (already in your requirements.txt):
  pip install pdfplumber pymupdf faiss-cpu transformers torch numpy tqdm
"""

import argparse
import json
import re
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── 1. PDF TEXT EXTRACTION ────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Try pdfplumber first (better layout), fall back to PyMuPDF (fitz).
    The indiacode.nic.in PDF is text-based (not scanned), so no OCR needed.
    """
    print(f"\n[1/5] Extracting text from: {pdf_path}")

    try:
        import pdfplumber
        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            print(f"      Pages found: {len(pdf.pages)}")
            for i, page in enumerate(tqdm(pdf.pages, desc="      Reading pages")):
                text = page.extract_text()
                if text:
                    full_text.append(text)
        result = "\n".join(full_text)
        print(f"      Characters extracted: {len(result):,}")
        return result

    except Exception as e:
        print(f"      pdfplumber failed ({e}), trying PyMuPDF...")
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = []
        for page in tqdm(doc, desc="      Reading pages"):
            full_text.append(page.get_text())
        result = "\n".join(full_text)
        print(f"      Characters extracted: {len(result):,}")
        return result


# ── 2. TEXT CLEANING ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove headers/footers, fix hyphenation, normalise whitespace."""
    print("\n[2/5] Cleaning text...")

    # Remove page numbers (standalone numbers on a line)
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)

    # Remove common header/footer lines from indiacode.nic.in PDFs
    text = re.sub(r'THE INDIAN CONTRACT ACT,?\s*1872\s*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Ministry of Law and Justice.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'indiacode\.nic\.in.*?\n', '', text, flags=re.IGNORECASE)

    # Fix hyphenated line breaks (e.g., "obliga-\ntion" → "obligation")
    text = re.sub(r'-\n(\w)', r'\1', text)

    # Collapse multiple blank lines to one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalise whitespace within lines
    lines = [' '.join(line.split()) for line in text.split('\n')]
    text = '\n'.join(lines)

    print(f"      Cleaned text length: {len(text):,} characters")
    return text


# ── 3. SECTION SEGMENTATION ───────────────────────────────────────────────────

# A section header in the body of the Act: "27. Agreement in restraint of trade, void.—Every ..."
# Some headers carry an amendment marker ("1[16.“Undue influence” defined.—") and some have no
# space after the number ("73.Compensation for loss ..."), and two have a footnote superscript
# glued to the number ("1151. Care to be taken by bailee" is footnote 1 + S.151).
_HEADER_RE = re.compile(r"^(?:\d\[|\d(?=\d{3}\.))?(\d{1,3})([A-Z]{0,2})\.\s*(?=[^\d\s.])", re.MULTILINE)
# Title runs from the header to the first em-dash; may wrap onto a second line.
_TITLE_RE = re.compile(r"([^—]{2,350}?)\.?\s*—", re.DOTALL)
# Page-footer amendment notes: "1. Subs. by Act 6 of 1899, s. 2, for the original s. 16."
_FOOTNOTE_LINE_RE = re.compile(
    r"^\d{1,2}\.\s*(?:Subs|Ins|Rep|Omitted|The words|Added|Certain|Cl\.|Section|S\.|Ss\.|Now|See|Cf\.)\b.*$",
    re.MULTILINE,
)
_AMENDMENT_MARK_RE = re.compile(r"(?<![\w\[])\d{1,2}\[")
MAX_SECTION_CHARS = 3000


def _section_key(num: str, suffix: str) -> tuple[int, str]:
    return int(num), suffix


def segment_into_sections(text: str, verbose: bool = True) -> list[dict]:
    """
    Split the cleaned ICA text into one chunk per section.

    The cleaned text opens with an "ARRANGEMENT OF SECTIONS" table of contents whose lines
    look like headers ("27. Agreement in restraint of trade, void."), so parsing starts at the
    second "1. Short title". In the body a real header always has an em-dash after its title,
    and section numbers only go up, which rejects footnote lines ("1. Subs. by Act ...") and
    numbered lists inside a section.

    The section's operative text (everything after the em-dash) is kept in full; an earlier
    segmenter captured a fixed 80-char title and silently dropped the opening words of every
    section (e.g. S.27 began "ined from exercising ...").
    """
    if verbose:
        print("\n[3/5] Segmenting into sections...")

    toc_ends = [m.start() for m in re.finditer(r"^1\.\s*Short title", text, re.MULTILINE)]
    body_start = toc_ends[1] if len(toc_ends) > 1 else (toc_ends[0] if toc_ends else 0)

    headers = []
    last = (0, "")
    for m in _HEADER_RE.finditer(text, body_start):
        key = _section_key(m.group(1), m.group(2))
        if key <= last:
            continue
        window = text[m.end(): m.end() + 400]
        if not (_TITLE_RE.match(window) or window.startswith("[")):
            continue
        headers.append(m)
        last = key

    sections = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[m.end():end]
        title_m = _TITLE_RE.match(chunk)
        if not title_m:
            continue  # "[Repealed.]" stubs and other title-less entries
        title = " ".join(title_m.group(1).split()).strip().rstrip(".")
        body = chunk[title_m.end():]
        body = _FOOTNOTE_LINE_RE.sub("", body)
        body = _AMENDMENT_MARK_RE.sub("", body)
        body = re.sub(r"\n{2,}", "\n", body).strip()
        if len(body) < 20 or title.startswith("[") or "Repealed" in title:
            continue  # repealed stubs: "[Delivery how made.] Rep. by s. 65, ibid."
        if len(body) > MAX_SECTION_CHARS:
            body = body[:MAX_SECTION_CHARS] + "..."

        section_num = f"{m.group(1)}{m.group(2)}"
        sections.append({
            "id": f"ICA_S{section_num}",
            "section_number": section_num,
            "title": title,
            "text": f"Section {section_num}. {title}. {body}",
            "source": "Indian Contract Act 1872",
        })

    if verbose:
        print(f"      Valid sections extracted: {len(sections)}")
        if sections:
            print(f"      Section IDs sample: {[s['id'] for s in sections[:5]]}")
    return sections


def refresh_registry_texts(registry: dict[str, dict], sections: list[dict]) -> tuple[dict[str, dict], dict]:
    """
    Rewrite the ICA entries of an existing chunk registry from freshly segmented sections,
    without re-embedding.

    Existing ids keep their position, so FAISS row i still maps to the i-th registry key.
    Sections missing from the registry are appended at the end: BM25 and the statute map can
    reach them immediately, FAISS only after the index is rebuilt. Non-ICA entries are untouched.
    """
    by_id = {s["id"]: s for s in sections}
    updated, unchanged, not_found = [], [], []
    out: dict[str, dict] = {}
    for sid, entry in registry.items():
        fresh = by_id.get(sid)
        if not sid.startswith("ICA_") or fresh is None:
            if sid.startswith("ICA_"):
                not_found.append(sid)
            out[sid] = entry
            continue
        new_entry = dict(entry, title=fresh["title"], text=fresh["text"])
        (updated if new_entry != entry else unchanged).append(sid)
        out[sid] = new_entry
    added = [s["id"] for s in sections if s["id"] not in registry]
    for sid in added:
        out[sid] = by_id[sid]
    return out, {"updated": updated, "unchanged": unchanged, "not_found": not_found, "added": added}


# ── 4. EMBEDDING + FAISS INDEX ────────────────────────────────────────────────

def build_faiss_index(sections: list[dict], model_name: str = "law-ai/InLegalBERT"):
    """
    Embed each section with InLegalBERT and build a FAISS index.
    Returns: (faiss_index, embeddings_array)
    """
    print(f"\n[4/5] Building FAISS index with {model_name}...")
    print(f"      Sections to embed: {len(sections)}")

    import torch
    import faiss
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      Device: {device} {'(GTX 1650 - good!)' if device == 'cuda' else '(CPU - slower but fine)'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def embed_text(text: str) -> np.ndarray:
        """Mean-pool the last hidden state to get a 768-dim embedding."""
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pool over token dimension
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding.cpu().numpy().astype("float32")

    embeddings = []
    for section in tqdm(sections, desc="      Embedding sections"):
        emb = embed_text(section["text"])
        embeddings.append(emb)

    embeddings_array = np.vstack(embeddings)  # shape: (n_sections, 768)
    print(f"      Embedding matrix shape: {embeddings_array.shape}")

    # Build FAISS index (inner product = cosine similarity after normalisation)
    faiss.normalize_L2(embeddings_array)
    index = faiss.IndexFlatIP(768)
    index.add(embeddings_array)
    print(f"      FAISS index built — {index.ntotal} vectors indexed")

    return index, embeddings_array


# ── 5. SAVE ARTIFACTS ─────────────────────────────────────────────────────────

def save_artifacts(sections, faiss_index, embeddings_array, output_dir="knowledge_base"):
    """Save chunk_registry.json and FAISS index to disk."""
    import faiss

    os.makedirs(f"{output_dir}/faiss_index", exist_ok=True)
    os.makedirs(f"{output_dir}/statutes", exist_ok=True)

    # chunk_registry.json — {section_id: full section dict}
    chunk_registry = {s["id"]: s for s in sections}
    registry_path = f"{output_dir}/chunk_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(chunk_registry, f, indent=2, ensure_ascii=False)
    print(f"      Saved chunk_registry.json ({len(chunk_registry)} sections)")

    # FAISS index
    faiss_path = f"{output_dir}/faiss_index/ica_1872.index"
    faiss.write_index(faiss_index, faiss_path)
    print(f"      Saved FAISS index -> {faiss_path}")

    # Embeddings (for inspection/debugging)
    np.save(f"{output_dir}/faiss_index/ica_1872_embeddings.npy", embeddings_array)
    print(f"      Saved embeddings array -> {output_dir}/faiss_index/ica_1872_embeddings.npy")

    return chunk_registry


# ── 6. PROOF-OF-CONCEPT RETRIEVAL TEST ───────────────────────────────────────

def run_retrieval_test(chunk_registry, faiss_index, model_name="law-ai/InLegalBERT"):
    """
    THE CRITICAL TEST FROM YOUR PRD:
    Query: indirect non-compete language
    → Does ICA_S27 appear in top-5 without the KG?

    This is the empirical justification for your Knowledge Graph.
    If S27 appears → semantic search is sufficient for this case.
    If S27 does NOT appear → KG is essential. (Expected result for your paper.)
    """
    import torch
    import faiss
    from transformers import AutoTokenizer, AutoModel

    print("\n" + "="*60)
    print("PROOF-OF-CONCEPT RETRIEVAL TEST")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    section_ids = list(chunk_registry.keys())  # ordered list for index→ID mapping

    TEST_QUERIES = [
        {
            "label": "Non-Compete (indirect wording — KG test case)",
            "query": "The employee shall not engage in any competing business activities for 2 years after termination.",
            "expected_section": "ICA_S27",
            "clause_type": "NonCompete"
        },
        {
            "label": "Force Majeure (no 'force majeure' phrase — KG test case)",
            "query": "Neither party shall be liable for delays caused by events beyond their reasonable control.",
            "expected_section": "ICA_S56",
            "clause_type": "ForceMajeure"
        },
        {
            "label": "Termination (direct language — should retrieve S73)",
            "query": "Either party may terminate this agreement upon breach with 30 days written notice.",
            "expected_section": "ICA_S73",
            "clause_type": "Termination"
        },
    ]

    for test in TEST_QUERIES:
        print(f"\n--- {test['label']} ---")
        print(f"Query: {test['query']}")

        # Embed query
        inputs = tokenizer(test["query"], return_tensors="pt",
                           max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        query_emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype("float32")
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        query_emb = query_emb.reshape(1, -1)

        # Search FAISS
        k = min(5, faiss_index.ntotal)
        scores, indices = faiss_index.search(query_emb, k)

        print(f"\nTop-{k} retrieved sections:")
        found_expected = False
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(section_ids):
                sid = section_ids[idx]
                title = chunk_registry[sid].get("title", "")
                is_expected = "[EXPECTED]" if sid == test["expected_section"] else ""
                if sid == test["expected_section"]:
                    found_expected = True
                print(f"  {rank+1}. [{sid}] {title[:60]} | score={score:.4f} {is_expected}")

        if found_expected:
            print(f"\nPASS: '{test['expected_section']}' FOUND in top-{k} - semantic search works for this case")
        else:
            print(f"\nFAIL: '{test['expected_section']}' NOT in top-{k} - KG IS ESSENTIAL for {test['clause_type']} clauses")
            print(f"   -> This is your key finding: pure semantic retrieval misses critical statutes")
            print(f"   -> The Knowledge Graph guarantees {test['expected_section']} is always included")

    print("\n" + "="*60)
    print("RETRIEVAL TEST COMPLETE")
    print("Save these results — they justify the KG in your paper!")
    print("="*60)


def refresh_registry_main(clean_text_path: str, output_dir: str) -> None:
    registry_path = f"{output_dir}/chunk_registry.json"
    with open(clean_text_path, encoding="utf-8") as f:
        clean = f.read()
    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    sections = segment_into_sections(clean)
    refreshed, report = refresh_registry_texts(registry, sections)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(refreshed, f, indent=2, ensure_ascii=False)

    print(f"\nRefreshed {registry_path}")
    print(f"  updated   : {len(report['updated'])}")
    print(f"  unchanged : {len(report['unchanged'])}")
    print(f"  not found : {report['not_found'] or 'none'}")
    print(f"  appended  : {report['added'] or 'none'}")
    print("FAISS vectors now describe the old text. Re-embed on a machine with torch:")
    print("  python build_knowledge_base.py --clean-text knowledge_base/statutes/ica_1872_clean.txt")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build ICA 1872 knowledge base for RAG pipeline")
    parser.add_argument("--pdf", help="Path to indian_contract_act_1872.pdf")
    parser.add_argument("--clean-text", help="Use an already-cleaned text file instead of extracting from --pdf "
                        "(e.g. knowledge_base/statutes/ica_1872_clean.txt)")
    parser.add_argument("--refresh-registry", action="store_true",
                        help="Re-segment --clean-text (default: <output>/statutes/ica_1872_clean.txt) and rewrite "
                             "the ICA texts in chunk_registry.json in place, keeping id order so the existing "
                             "FAISS index stays aligned. No PDF, torch or embedding model needed.")
    parser.add_argument("--output", default="knowledge_base", help="Output directory (default: knowledge_base/)")
    parser.add_argument("--model", default="law-ai/InLegalBERT", help="HuggingFace model for embeddings")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding (use if FAISS index already exists)")
    args = parser.parse_args()

    if args.refresh_registry:
        refresh_registry_main(args.clean_text or f"{args.output}/statutes/ica_1872_clean.txt", args.output)
        return

    if args.clean_text:
        with open(args.clean_text, encoding="utf-8") as f:
            clean = f.read()
        print(f"Using cleaned text from {args.clean_text}")
    else:
        if not args.pdf or not os.path.exists(args.pdf):
            print(f"ERROR: PDF not found at '{args.pdf}'")
            print("Download the Indian Contract Act 1872 PDF from https://indiacode.nic.in,")
            print("or pass --clean-text knowledge_base/statutes/ica_1872_clean.txt")
            sys.exit(1)

        # Step 1: Extract
        raw_text = extract_text_from_pdf(args.pdf)

        # Step 2: Clean
        clean = clean_text(raw_text)

        # Save cleaned text for inspection
        os.makedirs(f"{args.output}/statutes", exist_ok=True)
        with open(f"{args.output}/statutes/ica_1872_clean.txt", "w", encoding="utf-8") as f:
            f.write(clean)
        print(f"\n      Saved cleaned text -> {args.output}/statutes/ica_1872_clean.txt")
        print("      WARNING: Open this file and verify section headers were detected correctly before proceeding!")

    # Step 3: Segment
    sections = segment_into_sections(clean)

    if len(sections) < 5:
        print("\nWARNING: Very few sections detected.")
        print("   Open knowledge_base/statutes/ica_1872_clean.txt")
        print("   Find how section headers actually look in your PDF")
        print("   Then edit _HEADER_RE / _TITLE_RE used by segment_into_sections()")
        print("   Common issue: indiacode PDFs sometimes have non-standard formatting")

    # Inject stubs for Multi-statute KB (Arbitration Act 1996 & IT Act 2000)
    print("\n[3.5/5] Injecting Arbitration Act & IT Act stubs...")
    stubs = [
        {
            "id": "ARB_S8",
            "section_number": "8",
            "title": "Power to refer parties to arbitration where there is an arbitration agreement",
            "text": "Section 8. Power to refer parties to arbitration where there is an arbitration agreement. A judicial authority before which an action is brought in a matter which is the subject of an arbitration agreement shall, if a party so applies, refer the parties to arbitration.",
            "source": "Arbitration and Conciliation Act 1996"
        },
        {
            "id": "ARB_S11",
            "section_number": "11",
            "title": "Appointment of arbitrators",
            "text": "Section 11. Appointment of arbitrators. A person of any nationality may be an arbitrator. The parties are free to agree on a procedure for appointing the arbitrator.",
            "source": "Arbitration and Conciliation Act 1996"
        },
        {
            "id": "IT_S43A",
            "section_number": "43A",
            "title": "Compensation for failure to protect data",
            "text": "Section 43A. Compensation for failure to protect data. Where a body corporate, possessing, dealing or handling any sensitive personal data or information in a computer resource which it owns, controls or operates, is negligent in implementing and maintaining reasonable security practices and procedures and thereby causes wrongful loss or wrongful gain to any person, such body corporate shall be liable to pay damages by way of compensation to the person so affected.",
            "source": "Information Technology Act 2000"
        },
        {
            "id": "IT_S79",
            "section_number": "79",
            "title": "Exemption from liability of intermediary in certain cases",
            "text": "Section 79. Exemption from liability of intermediary in certain cases. An intermediary shall not be liable for any third party information, data, or communication link made available or hosted by him.",
            "source": "Information Technology Act 2000"
        }
    ]
    sections.extend(stubs)
    print(f"      Added {len(stubs)} multi-statute stubs.")

    if args.skip_embed:
        print("\nSkipping embedding (--skip-embed flag set)")
        return

    # Step 4: Embed + FAISS
    faiss_index, embeddings = build_faiss_index(sections, args.model)

    # Step 5: Save
    print(f"\n[5/5] Saving artifacts to {args.output}/")
    chunk_registry = save_artifacts(sections, faiss_index, embeddings, args.output)

    # Step 6: Retrieval test
    run_retrieval_test(chunk_registry, faiss_index, args.model)

    print(f"\nKnowledge base built successfully!")
    print(f"   Sections indexed: {len(sections)}")
    print(f"   Output: {args.output}/")
    print(f"\nNext step: Run the same script for Arb. Act 1996 and IT Act 2000")
    print(f"Then build clause_statute_kg.py (Week 7)")


if __name__ == "__main__":
    main()
