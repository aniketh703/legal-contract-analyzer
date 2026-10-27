"""
tests/test_kb_text.py
=====================
Regression tests for the statute text in knowledge_base/chunk_registry.json and the
segmenter in build_knowledge_base.py that produces it.

An earlier segmenter dropped the operative opening of every ICA section (S.27 began
"ined from exercising ..."). An LLM grounded on that text would quote broken law.
"""
import json
import re
from pathlib import Path

import pytest

from build_knowledge_base import refresh_registry_texts, segment_into_sections

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "knowledge_base" / "chunk_registry.json"
CLEAN_TEXT_PATH = ROOT / "knowledge_base" / "statutes" / "ica_1872_clean.txt"

# Opening words of the operative text, as enacted.
OPENINGS = {
    "ICA_S10": "All agreements are contracts if they are made by the free consent of parties",
    "ICA_S16": "(1) A contract is said to be induced by “undue influence” where",
    "ICA_S23": "The consideration or object of an agreement is lawful, unless",
    "ICA_S27": "Every agreement by which any one is restrained from exercising a lawful profession",
    "ICA_S28": "Every agreement,",
    "ICA_S56": "An agreement to do an act impossible in itself is void.",
    "ICA_S73": "When a contract has been broken, the party who suffers by such breach",
    "ICA_S74": "When a contract has been broken, if a sum is named in the contract",
    "ICA_S124": "A contract by which one party promises to save the other from loss",
    "ICA_S151": "In all cases of bailment the bailee is bound to take as much care",
    "ICA_S125": "The promisee in a contract of indemnity, acting within the scope of his authority",
}


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("sid,opening", sorted(OPENINGS.items()))
def test_registry_section_starts_with_operative_text(registry, sid, opening):
    entry = registry[sid]
    head = f"Section {entry['section_number']}. {entry['title']}. "
    assert entry["text"].startswith(head)
    assert _flat(entry["text"][len(head):]).startswith(opening)


def test_no_ica_body_starts_mid_word(registry):
    bad = []
    for sid, entry in registry.items():
        if not sid.startswith("ICA_") or sid == "ICA_S90":  # S.90 is a repealed placeholder
            continue
        head = f"Section {entry['section_number']}. {entry['title']}. "
        body = entry["text"][len(head):]
        if not re.match(r"[\"“(A-Z\[]", body):
            bad.append((sid, body[:40]))
    assert not bad


def test_registry_keeps_faiss_alignment(registry):
    # FAISS row i maps to the i-th registry key; the 189 embedded entries must stay first, in order.
    ids = list(registry)
    assert ids[0] == "ICA_S1"
    assert ids[185:189] == ["ARB_S8", "ARB_S11", "IT_S43A", "IT_S79"]


def test_segmenter_skips_table_of_contents():
    sections = segment_into_sections(CLEAN_TEXT_PATH.read_text(encoding="utf-8"), verbose=False)
    by_id = {s["id"]: s for s in sections}
    assert len(by_id) == len(sections)  # one chunk per section, no duplicate ids
    s27 = by_id["ICA_S27"]
    assert s27["title"] == "Agreement in restraint of trade, void"
    # The TOC line for S.27 is followed by the TOC entry for its exception; that must not leak in.
    assert "28. Agreements in restraint of legal proceeding void" not in s27["text"]
    assert "Exception 1.—Saving of agreement not to carry on business" in s27["text"]


SAMPLE = """ARRANGEMENT OF SECTIONS
1. Short title.
2. Interpretation-clause.
3. Lost words.
1. Short title.—This Act may be called the Sample Act.
2. Interpretation-clause.—In this Act, unless the context otherwise requires,
words mean what they say.
1. Subs. by Act 1 of 1900, s. 2, for the original words.
3.Lost words.—1[Every agreement by which a party is restrained
from doing something is void.]
4. [Repealed.].
"""


def test_segmenter_on_synthetic_text():
    sections = segment_into_sections(SAMPLE, verbose=False)
    assert [s["id"] for s in sections] == ["ICA_S1", "ICA_S2", "ICA_S3"]
    s2, s3 = sections[1], sections[2]
    assert "Subs. by Act" not in s2["text"]          # page-footer amendment note removed
    assert s3["text"].startswith("Section 3. Lost words. Every agreement by which a party")  # "1[" marker dropped


def test_refresh_preserves_order_and_appends_new():
    registry = {
        "ICA_S1": {"id": "ICA_S1", "section_number": "1", "title": "Short title", "text": "old", "source": "x"},
        "ARB_S8": {"id": "ARB_S8", "section_number": "8", "title": "t", "text": "arb", "source": "y"},
        "ICA_S9": {"id": "ICA_S9", "section_number": "9", "title": "t", "text": "keep", "source": "x"},
    }
    sections = [
        {"id": "ICA_S1", "section_number": "1", "title": "Short title", "text": "new", "source": "x"},
        {"id": "ICA_S5", "section_number": "5", "title": "Five", "text": "five", "source": "x"},
    ]
    out, report = refresh_registry_texts(registry, sections)
    assert list(out) == ["ICA_S1", "ARB_S8", "ICA_S9", "ICA_S5"]
    assert out["ICA_S1"]["text"] == "new"
    assert out["ARB_S8"] == registry["ARB_S8"]
    assert out["ICA_S9"]["text"] == "keep"
    assert report["updated"] == ["ICA_S1"] and report["added"] == ["ICA_S5"]
    assert report["not_found"] == ["ICA_S9"]
