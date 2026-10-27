"""Update the VNIT Eval-1 PPTX to reflect the Eval-2 (24/05/2026) project status.

Preserves the original deck's master/template, group decorations, and text-box
positioning. Only the title (TextBox 4), body text (TextBox 5), and footer
date (TextBox 8) of each content slide are replaced. Slide 1 (title slide)
and Slide 11 (Thank You) get targeted edits.
"""

import copy
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor

SRC = r"C:\Users\Ani\Downloads\RAG_Legal_Contract_Analyzer_VNIT.pptx.pptx"
DST = r"C:\Users\Ani\Downloads\RAG_Legal_Contract_Analyzer_VNIT_Eval2.pptx"
NEW_DATE = "24-05-2026"


# -----------------------------------------------------------------------------
# Slide content for Eval-2
# -----------------------------------------------------------------------------
# Each entry for slides 2-10: (title, body_paragraphs)
# body_paragraphs: list of (bold_lead, rest) tuples. bold_lead is bolded, rest
# is normal weight. Use empty bold_lead for a plain paragraph.

SLIDE2_TITLE = "Table of Contents"
SLIDE2_BODY = [
    ("", "1. Introduction & Phase-1 Status"),
    ("", "2. Objective Recap & Delivered Scope"),
    ("", "3. System Architecture (As-Built)"),
    ("", "4. Project Timeline & Progress"),
    ("", "5. Methodology & Pipeline Implementation"),
    ("", "6. Evaluation Results (Actual vs Expected)"),
    ("", "7. Conclusion & Contributions"),
    ("", "8. References"),
]

SLIDE3_TITLE = "Introduction & Phase-1 Status"
SLIDE3_BODY = [
    ("", "India has over 63 million SMBs that sign vendor agreements, NDAs, service contracts and lease deeds governed by the Indian Contract Act 1872. Legal consultation costs Rs. 3,000-15,000 per hour, so most founders skip legal review and discover unfavourable clauses only when disputes arise. Existing contract-AI tools (Ironclad, Kira) are trained on US/UK datasets and miss India-specific concepts such as Section 27 void non-competes, GST indemnification and Arbitration & Conciliation Act 1996."),
    ("Research Gap:", " No publicly available annotated Indian commercial-contract dataset exists. No RAG system has been applied to Indian contract clause extraction and risk scoring. NyayaRAG (2025) and ILDC (ACL 2021) target court judgments, not commercial contracts."),
    ("Phase-1 Status (24 May 2026):", " End-to-end pipeline fully built and demoed in browser. 9,598 labelled clauses used to train a 42-type classifier (83.2 % accuracy). Indian Contract Act 1872 knowledge base (185 sections) indexed with FAISS + BM25 + statute map. 116/116 unit tests passing in CI. Project ready for Eval-2 review."),
]

SLIDE4_TITLE = "Objective Recap & Delivered Scope"
SLIDE4_BODY = [
    ("Original Objective:", " Build a RAG-based system that extracts, classifies and risk-scores clauses in Indian commercial contracts using InLegalBERT and open-source LLMs, providing plain-English explanations grounded in Indian statute law."),
    ("Delivered in Phase 1:", ""),
    ("1. Clause classifier:", " TF-IDF + Logistic Regression trained on 9,447 CUAD clauses + 151 hand-crafted Indian clauses, covering 42 clause types. InLegalBERT embeddings are wired in as a fallback stage."),
    ("2. Indian Contract Act 1872 knowledge base:", " 185 sections chunked, embedded and indexed in FAISS, BM25 and a statute-map for guaranteed lookup."),
    ("3. End-to-end pipeline & web app:", " Flask application that accepts a PDF/TXT contract and returns an HTML risk report with ICA section references, plain-English explanations and recommended actions."),
    ("Scope adjustment:", " The generator is template-based (statute-aware) rather than LLM-driven. This keeps the project fully offline, reproducible and cost-free, while still grounding every explanation in a retrieved ICA section."),
]

SLIDE5_TITLE = "System Architecture (As-Built)"
SLIDE5_BODY = [
    ("", "[INPUT]  Contract PDF / TXT  (OCR fallback via Tesseract)"),
    ("", "      |"),
    ("", "[STEP 1] Segmenter  ->  splits the document into clause dicts"),
    ("", "      |"),
    ("", "[STEP 2] Classifier  (3-stage cascade)"),
    ("", "         Stage A: TF-IDF + Logistic Regression  - 42 clause types"),
    ("", "         Stage B: keyword-rule fallback"),
    ("", "         Stage C: InLegalBERT embedding fallback"),
    ("", "      |"),
    ("", "[STEP 3] Retriever  over Indian Contract Act 1872 (185 sections)"),
    ("", "         Statute map  ->  FAISS dense  ->  BM25 sparse  ->  RRF fusion"),
    ("", "      |"),
    ("", "[STEP 4] Generator  ->  template-based, statute-aware HTML report"),
    ("", "      |"),
    ("", "[OUTPUT] Flask web app  -  Clause | Type | Risk (H/M/L) | ICA Section | Plain-English Explanation | Recommended Action"),
]

SLIDE6_TITLE = "Project Timeline & Progress"
SLIDE6_BODY = [
    ("Phase 1  (Mar 2026 - May 2026) - COMPLETE", ""),
    ("Month 1 (Mar-Apr):", " Literature review, Indian Kanoon scraping (330 judgment JSONs), clause-type schema, classifier baseline, FAISS index for ICA 1872."),
    ("Month 2 (Apr-May):", " 212 clauses curated, keyword-rule classifier (78.8 % accuracy, 12 types), retriever evaluation (BM25 23 % Hit@5 -> statute-map+BM25 100 % Hit@5)."),
    ("Month 3 (May):", " Migrated to TF-IDF + LR ML classifier on CUAD + Indian supplemental data (83.2 % accuracy, 42 types). Flask web app, 116-test pytest suite, GitHub Actions CI."),
    ("Phase 2  (Jun 2026 onwards) - PLANNED", ""),
    ("Month 4 (Jun-Jul):", " InLegalBERT-LoRA fine-tune comparison, ablation studies (chunk size, retrieval-k, embedding model), RAGAS faithfulness evaluation against an LLM baseline."),
    ("Month 5 (Jul-Aug):", " 10-15 SMB-founder human evaluation (Likert), paper draft, final demonstration."),
]

SLIDE7_TITLE = "Methodology & Pipeline Implementation"
SLIDE7_BODY = [
    ("Step 1 - Dataset:", " 330 Indian Kanoon judgments scraped, 212 clauses curated. CUAD (9,447 labelled US-contract clauses) used for primary classifier training. 151 hand-crafted Indian clauses (Arbitration, Confidentiality, Indemnification) added as supplemental training data in Indian drafting style."),
    ("Step 2 - Clause Classifier:", " TF-IDF (uni- + bi-gram) + Logistic Regression on 9,598 examples, 80/20 train/test split. 42 clause types. InLegalBERT (law-ai/InLegalBERT, 5.4 M Indian legal docs) used for embedding-fallback in Stage C."),
    ("Step 3 - Knowledge Base:", " Indian Contract Act 1872 chunked by section (185 sections). Indexed in FAISS (dense, MiniLM) and BM25 (sparse). A hardcoded clause-type -> ICA-section statute map sits in front for guaranteed retrieval."),
    ("Step 4 - Retrieval:", " 3-layer hybrid: statute map -> FAISS -> BM25, fused with Reciprocal Rank Fusion. Top-k ICA sections feed the report generator."),
    ("Step 5 - Risk Report Generator:", " Template-based, statute-aware HTML report. Per-clause template selects wording, recommended actions and risk colour based on retrieved ICA sections. Legal disclaimer included in UI, HTML and JSON exports."),
    ("Tech Stack:", " Python | scikit-learn | HuggingFace Transformers | FAISS | rank_bm25 | PyMuPDF / pdfplumber + Tesseract OCR | Flask | pytest + GitHub Actions CI."),
]

SLIDE8_TITLE = "Evaluation Results - Actual vs Expected"
SLIDE8_BODY = [
    ("Classifier (TF-IDF + Logistic Regression, 42 clause types):", ""),
    ("  - Overall accuracy", " 83.2 %  (expected > 82 % - MET)."),
    ("  - Weighted F1", " 0.83  on a held-out 20 % test split of 9,598 labelled clauses."),
    ("  - Per-type F1 highlights:", " GoverningLaw 0.99 | Arbitration 0.95 | Confidentiality 0.95 | Indemnification 0.95 | AuditRights 0.96 | Insurance 0.96 | LiabilityCap 0.87 | Termination 0.84."),
    ("  - Baseline jump:", " 78.8 % accuracy / 12 types (keyword rules)  ->  83.2 % / 42 types (ML + Indian supplemental)."),
    ("Retriever (Indian Contract Act 1872, 185 sections):", ""),
    ("  - BM25 only", " Hit@5 = 23.0 %."),
    ("  - Statute map + BM25", " Hit@5 = 100 %   <- key Phase-1 finding."),
    ("End-to-End Validation:", " Real MSA processed - 15 clauses, 10 distinct types, 30 KB HTML report, 5 HIGH + 9 MEDIUM + 1 LOW risk labels. ICA references S73, S36, S130, S74, S48 verified present in the report. 116/116 pytest tests pass; CI green."),
    ("Deferred to Phase 2:", " RAGAS faithfulness vs LLM-only, InLegalBERT-LoRA fine-tune comparison, NDCG@5 ablation, SMB-founder Likert human evaluation."),
]

SLIDE9_TITLE = "Conclusion & Contributions"
SLIDE9_BODY = [
    ("", "Phase 1 delivered a working, offline, reproducible Legal Contract Analyzer for the Indian SMB context, with every result grounded in a retrieved Indian Contract Act 1872 section and explained in plain English."),
    ("Research / Engineering Contributions (Phase 1):", ""),
    ("1.", " First end-to-end Indian-contract risk-analysis pipeline that combines a 42-type clause classifier with an ICA 1872 retriever and template-based explanation generator."),
    ("2.", " Empirical comparison of retrieval strategies on Indian statute lookup - BM25 alone achieves only 23 % Hit@5, while a statute-map + BM25 hybrid reaches 100 % Hit@5."),
    ("3.", " Curated assets: 330 Indian Kanoon judgment JSONs, 212 curated clauses and 151 hand-crafted Indian-style clauses across Arbitration, Confidentiality and Indemnification - extending CUAD with Indian legal drafting patterns."),
    ("4.", " Open, offline reference implementation: Flask web app, 116 unit tests, GitHub Actions CI, legal disclaimer in UI / HTML / JSON exports."),
    ("Next Steps:", " InLegalBERT-LoRA fine-tune, RAGAS faithfulness evaluation against an LLM baseline, ablation studies, SMB-founder human evaluation and paper submission in Phase 2."),
]

SLIDE10_TITLE = "References"
SLIDE10_BODY = [
    ("", "[1]  Malik et al. \"ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation.\" ACL 2021."),
    ("", "[2]  Hendrycks et al. \"CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.\" NeurIPS 2021."),
    ("", "[3]  Guha et al. \"LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in LLMs.\" NeurIPS 2023."),
    ("", "[4]  \"IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning.\" ACL 2024."),
    ("", "[5]  \"NyayaRAG: Retrieval-Augmented Generation for Indian Legal Judgment Prediction.\" arXiv 2025."),
    ("", "[6]  \"LawPal: A RAG-Based Legal Query System for Indian Judiciary.\" arXiv Feb 2025."),
    ("", "[7]  Paul et al. \"InLegalBERT: Pre-trained Language Models for the Indian Legal Domain.\" HuggingFace law-ai/InLegalBERT, 2022."),
    ("", "[8]  Lewis et al. \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.\" NeurIPS 2020."),
    ("", "[9]  Hu et al. \"LoRA: Low-Rank Adaptation of Large Language Models.\" ICLR 2022."),
    ("", "[10] Robertson & Zaragoza. \"The Probabilistic Relevance Framework: BM25 and Beyond.\" FnTIR 2009."),
    ("", "[11] Johnson et al. \"Billion-scale similarity search with GPUs (FAISS).\" IEEE Trans. Big Data 2019."),
    ("", "[12] Cormack et al. \"Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.\" SIGIR 2009."),
]

NEW_CONTENT = {
    2:  (SLIDE2_TITLE,  SLIDE2_BODY),
    3:  (SLIDE3_TITLE,  SLIDE3_BODY),
    4:  (SLIDE4_TITLE,  SLIDE4_BODY),
    5:  (SLIDE5_TITLE,  SLIDE5_BODY),
    6:  (SLIDE6_TITLE,  SLIDE6_BODY),
    7:  (SLIDE7_TITLE,  SLIDE7_BODY),
    8:  (SLIDE8_TITLE,  SLIDE8_BODY),
    9:  (SLIDE9_TITLE,  SLIDE9_BODY),
    10: (SLIDE10_TITLE, SLIDE10_BODY),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def iter_text_shapes(shape):
    """Yield all text-bearing shapes recursively (descending into groups)."""
    if shape.shape_type == 6:  # GROUP
        for s in shape.shapes:
            yield from iter_text_shapes(s)
    elif getattr(shape, "has_text_frame", False):
        yield shape


def find_named(slide, name):
    for top in slide.shapes:
        for s in iter_text_shapes(top):
            if s.name == name:
                return s
    return None


def set_single_run_text(shape, text):
    """Replace shape text with a single run, preserving its first-run formatting."""
    tf = shape.text_frame
    # Capture style of the first run
    first_p = tf.paragraphs[0]
    style = None
    if first_p.runs:
        r = first_p.runs[0]
        style = {
            "size":  r.font.size,
            "bold":  r.font.bold,
            "italic": r.font.italic,
            "name":  r.font.name,
            "color": None,
        }
        try:
            if r.font.color and r.font.color.type is not None:
                style["color"] = RGBColor(*r.font.color.rgb)
        except Exception:
            pass
    # Clear all paragraphs except the first
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    # Clear runs in the first paragraph
    for r in list(first_p.runs):
        r._r.getparent().remove(r._r)
    new_run = first_p.add_run()
    new_run.text = text
    if style:
        if style["size"]:    new_run.font.size = style["size"]
        if style["bold"] is not None:   new_run.font.bold = style["bold"]
        if style["italic"] is not None: new_run.font.italic = style["italic"]
        if style["name"]:    new_run.font.name = style["name"]
        if style["color"]:   new_run.font.color.rgb = style["color"]


def replace_body_text(shape, paragraphs, font_size_pt=None):
    """Replace TextBox 5 body content with the given (bold_lead, rest) paragraphs."""
    tf = shape.text_frame
    # Sample formatting from first existing run
    sample_run = None
    for p in tf.paragraphs:
        if p.runs:
            sample_run = p.runs[0]
            break
    sample_size = sample_run.font.size if sample_run and sample_run.font.size else Pt(20)
    sample_name = sample_run.font.name if sample_run and sample_run.font.name else "Calibri"
    sample_color = None
    try:
        if sample_run and sample_run.font.color and sample_run.font.color.type is not None:
            sample_color = RGBColor(*sample_run.font.color.rgb)
    except Exception:
        pass
    if font_size_pt is not None:
        sample_size = Pt(font_size_pt)

    # Remove all existing paragraphs (we keep one to seed)
    for p in list(tf.paragraphs):
        p._p.getparent().remove(p._p)

    # Recreate paragraphs from scratch using lxml
    from pptx.oxml.ns import qn
    from lxml import etree

    txBody = tf._txBody
    for bold_lead, rest in paragraphs:
        a_p = etree.SubElement(txBody, qn("a:p"))
        if bold_lead:
            r1 = etree.SubElement(a_p, qn("a:r"))
            rPr1 = etree.SubElement(r1, qn("a:rPr"))
            rPr1.set("lang", "en-IN")
            rPr1.set("b", "1")
            if sample_size: rPr1.set("sz", str(sample_size))
            if sample_name:
                lat = etree.SubElement(rPr1, qn("a:latin"))
                lat.set("typeface", sample_name)
            if sample_color:
                fill = etree.SubElement(rPr1, qn("a:solidFill"))
                srgb = etree.SubElement(fill, qn("a:srgbClr"))
                srgb.set("val", "{:02X}{:02X}{:02X}".format(*sample_color))
            t1 = etree.SubElement(r1, qn("a:t"))
            t1.text = bold_lead
        if rest:
            r2 = etree.SubElement(a_p, qn("a:r"))
            rPr2 = etree.SubElement(r2, qn("a:rPr"))
            rPr2.set("lang", "en-IN")
            if sample_size: rPr2.set("sz", str(sample_size))
            if sample_name:
                lat = etree.SubElement(rPr2, qn("a:latin"))
                lat.set("typeface", sample_name)
            if sample_color:
                fill = etree.SubElement(rPr2, qn("a:solidFill"))
                srgb = etree.SubElement(fill, qn("a:srgbClr"))
                srgb.set("val", "{:02X}{:02X}{:02X}".format(*sample_color))
            t2 = etree.SubElement(r2, qn("a:t"))
            t2.text = rest
        if not bold_lead and not rest:
            # empty paragraph spacer
            etree.SubElement(a_p, qn("a:endParaRPr")).set("lang", "en-IN")


# -----------------------------------------------------------------------------
# Per-slide font-size targets so dense bodies still fit
# -----------------------------------------------------------------------------
BODY_FONT_PT = {
    2:  28,   # ToC - large, only 8 lines
    3:  18,   # Intro - 3 long paras
    4:  16,   # Objective - many lines
    5:  16,   # Architecture diagram - monospace-ish lines
    6:  16,   # Timeline - many lines
    7:  15,   # Methodology - many lines
    8:  15,   # Results - many lines
    9:  16,   # Conclusion
    10: 14,   # References - 12 entries
}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    prs = Presentation(SRC)

    # ---- Slide 1: Title slide ----
    slide1 = prs.slides[0]
    slide1_shapes = []
    for s in slide1.shapes:
        slide1_shapes.extend(iter_text_shapes(s))
    for shape in slide1_shapes:
        for p in shape.text_frame.paragraphs:
            for r in p.runs:
                if "Eval. 1" in r.text:
                    r.text = r.text.replace("Eval. 1", "Eval. 2")
                if "14/03/2026" in r.text:
                    r.text = r.text.replace("14/03/2026", "24/05/2026")
                if r.text.strip() == "12-03-2026":
                    r.text = NEW_DATE

    # ---- Slides 2-10: content slides ----
    for slide_no, (title, body) in NEW_CONTENT.items():
        slide = prs.slides[slide_no - 1]
        title_shape = find_named(slide, "TextBox 4")
        body_shape  = find_named(slide, "TextBox 5")
        date_shape  = find_named(slide, "TextBox 8")
        if title_shape:
            set_single_run_text(title_shape, title)
        if body_shape:
            replace_body_text(body_shape, body, font_size_pt=BODY_FONT_PT.get(slide_no))
        if date_shape:
            set_single_run_text(date_shape, NEW_DATE)

    # ---- All slides: replace any stale "12-03-2026" footer ----
    for slide in prs.slides:
        for top in slide.shapes:
            for s in iter_text_shapes(top):
                for p in s.text_frame.paragraphs:
                    for r in p.runs:
                        if r.text.strip() == "12-03-2026":
                            r.text = NEW_DATE

    prs.save(DST)
    print(f"Saved: {DST}")


if __name__ == "__main__":
    main()
