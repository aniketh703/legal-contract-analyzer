"""
pipeline/segmenter.py
=====================
Splits a raw contract (plain text or PDF) into individual clause segments.

Output per clause:
    {
        "clause_id":   "doc_001_c03",
        "clause_text": "...",
        "heading":     "17.2 Arbitration",   # or "" if no heading detected
        "char_start":  1240,
        "char_end":    1890,
    }

Usage:
    from pipeline.segmenter import segment_contract
    clauses = segment_contract("path/to/contract.pdf")
    # or pass raw text:
    clauses = segment_contract(text="The parties agree that ...")
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Matches numbered headings like:  1.   2.3   17.2.1   CLAUSE 5   Article IV
_HEADING_PAT = re.compile(
    r"""
    (?:^|\n)
    (
        (?:
            (?:clause|article|section|schedule|annexure|exhibit)\s+   # keyword prefix
            [\dIVXivx]+(?:\.\d+)*                                     #   + number
        )
        |
        (?:\d{1,3}(?:\.\d{1,3}){0,3}\.?\s+[A-Z][^\n]{0,80})          # "17.2 Arbitration..."
        |
        (?:[A-Z][A-Z\s]{4,60}:?\s*$)                                  # ALL-CAPS heading line
    )
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

# Minimum / maximum character lengths to keep a segment
_MIN_CHARS = 80
_MAX_CHARS = 2000

# Below this char count from native PDF text, try OCR (scanned PDFs)
_MIN_NATIVE_PDF_CHARS = 200


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_text_from_pdf_native(pdf_path: str) -> str:
    """Extract embedded text via pdfplumber, then PyMuPDF."""
    path = str(pdf_path)
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
        text = "\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass

    import fitz  # PyMuPDF fallback
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def _extract_text_from_pdf_ocr(pdf_path: str) -> str:
    """
    OCR fallback for scanned PDFs. Requires Tesseract on PATH and:
      pip install pytesseract Pillow
    """
    try:
        import io

        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    pages: list[str] = []
    try:
        doc = fitz.open(str(pdf_path))
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            pages.append(pytesseract.image_to_string(img))
        return "\n".join(pages)
    except Exception:
        return ""


def _extract_text_from_pdf(pdf_path: str) -> str:
    native = _extract_text_from_pdf_native(pdf_path)
    if len(native.strip()) >= _MIN_NATIVE_PDF_CHARS:
        return native

    ocr = _extract_text_from_pdf_ocr(pdf_path)
    if len(ocr.strip()) > len(native.strip()):
        return ocr
    return native


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    # Fix hyphenated line-breaks
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalise non-breaking spaces and other unicode whitespace
    text = re.sub(r"[\xa0\u200b\u2009\u202f]", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def _split_into_segments(text: str) -> list[dict]:
    """
    Split text at every detected heading boundary.
    Returns raw segments (before length filtering).
    """
    matches = list(_HEADING_PAT.finditer(text))

    if not matches:
        # No headings found — fall back to double-newline paragraph splits
        paragraphs = re.split(r"\n{2,}", text)
        segments = []
        pos = 0
        for para in paragraphs:
            start = text.find(para, pos)
            end = start + len(para)
            segments.append({"heading": "", "text": para.strip(), "start": start, "end": end})
            pos = end
        return segments

    segments = []
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        # Fold heading into body so the clause is self-contained
        full_text = (heading + " " + body).strip() if body else heading

        segments.append({
            "heading": heading,
            "text": full_text,
            "start": match.start(),
            "end": body_end,
        })

    return segments


def _filter_and_label(segments: list[dict], doc_id: str) -> list[dict]:
    """Apply length filter and assign clause_id."""
    results = []
    counter = 0
    for seg in segments:
        text = re.sub(r"\s+", " ", seg["text"]).strip()
        if len(text) < _MIN_CHARS:
            continue
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS]
        results.append({
            "clause_id": f"{doc_id}_c{counter:02d}",
            "clause_text": text,
            "heading": seg["heading"],
            "char_start": seg["start"],
            "char_end": seg["end"],
        })
        counter += 1
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_contract_text(
    path: str | Path | None = None,
    text: str | None = None,
) -> str:
    """Load raw contract text from a file path or in-memory string."""
    if path is None and text is None:
        raise ValueError("Provide either 'path' or 'text'.")
    if path is not None:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            return _extract_text_from_pdf(path)
        return path.read_text(encoding="utf-8")
    return text or ""


def segment_contract(
    path: str | Path | None = None,
    text: str | None = None,
    doc_id: str | None = None,
) -> list[dict]:
    """
    Segment a contract into clauses.

    Args:
        path:   Path to a .pdf or .txt file.
        text:   Raw contract text (alternative to path).
        doc_id: Identifier prefix for clause IDs. Auto-generated if omitted.

    Returns:
        List of clause dicts with keys:
            clause_id, clause_text, heading, char_start, char_end
    """
    raw = load_contract_text(path=path, text=text)

    if doc_id is None:
        doc_id = path.stem if path is not None else uuid.uuid4().hex[:8]

    cleaned = _clean(raw)
    segments = _split_into_segments(cleaned)
    clauses = _filter_and_label(segments, doc_id)
    return clauses


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.segmenter <contract.pdf|contract.txt>")
        sys.exit(1)

    results = segment_contract(path=sys.argv[1])
    print(f"Extracted {len(results)} clauses\n")
    for c in results[:5]:
        print(f"[{c['clause_id']}] heading={c['heading']!r}")
        print(f"  {c['clause_text'][:120]}...")
        print()
