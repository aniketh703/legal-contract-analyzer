"""

pipeline/document_gate.py

=========================

Heuristic gate: reject resumes/profiles before classification and retrieval.



Project titles such as "Legal Contract Analyzer" or portfolio blurbs that mention

"contract" or "agreement" in a product name must NOT bypass this gate. Those

strings are not legal-contract phrasing; we require operative clause language

("the parties", "hereby agree", "shall not", "governing law", etc.) and penalise

resume structure (section headers, job timelines, contact links) so CVs are

rejected even when a side project references contracts.

"""



from __future__ import annotations



import re



# ---------------------------------------------------------------------------

# Thresholds (tune here — no ML)

# ---------------------------------------------------------------------------



MIN_TEXT_CHARS = 150

MIN_CONTRACT_SCORE = 5

RESUME_REJECT_THRESHOLD = 4

RESUME_STRUCTURAL_REJECT = 3



CONTRACT_GATE_MESSAGE = (

    "This file does not appear to be a legal contract. Please upload a commercial "

    "agreement (MSA, NDA, employment contract, etc.). LinkedIn profiles and resumes "

    "are not supported."

)



# ---------------------------------------------------------------------------

# Lexicons

# ---------------------------------------------------------------------------



# Weak tokens alone do not prove a legal agreement (product names, job blurbs).

_CONTRACT_WEAK_KEYWORDS = (

    "agreement",

    "contract",

    "msa",

    "nda",

)



# Strong operative / formal legal phrasing (weighted higher).

_LEGAL_CONTRACT_PHRASES = (

    "the parties",

    "hereby agree",

    "shall not",

    "shall indemnify",

    "indemnify",

    "governing law",

    "whereas",

    "in witness whereof",

    "executed as of",

    "effective date",

    "hereinafter",

    "party agrees",

    "either party",

    "material breach",

    "notice period",

    "force majeure",

    "non-compete",

    "noncompete",

    "arbitration",

    "confidentiality",

    "limitation of liability",

)



_RESUME_KEYWORDS = (

    "linkedin",

    "resume",

    "résumé",

    "curriculum vitae",

    "portfolio",

    "github",

    "gitlab",

    "education",

    "bachelor",

    "master",

    "mba",

    "internship",

    "honors",

    "honours",

    "publications",

    "references",

    "panterra",

    "panterra networks",

)



_RESUME_SECTION_HEADERS = (

    "summary",

    "experience",

    "education",

    "skills",

    "certifications",

    "certification",

    "project",

    "projects",

    "objective",

    "contact",

    "languages",

    "awards",

    "publications",

    "internship",

    "internships",

)



_KEYWORD_WEIGHT = 1

_LEGAL_PHRASE_WEIGHT = 2

_STRUCTURE_WEIGHT = 2

_JOB_DATE_WEIGHT = 1

_RESUME_PAGINATION_WEIGHT = 3

_RESUME_SECTION_WEIGHT = 3

_CONTACT_PATTERN_WEIGHT = 2

_FILENAME_RESUME_BOOST = 3



_CLAUSE_NUMBER_RE = re.compile(r"\b\d{1,2}\.\d{1,2}(?:\.\d+)?\b")

_SECTION_HEADING_RE = re.compile(

    r"(?:^|\n)\s*(?:section|article|clause|schedule)\s+[\dIVXivx]+",

    re.IGNORECASE | re.MULTILINE,

)

_ARTICLE_CAPS_RE = re.compile(r"\bARTICLE\s+[IVX\d]+\b", re.IGNORECASE)

_NUMBERED_ARTICLE_RE = re.compile(

    r"(?:^|\n)\s*(?:article|section)\s+\d+[\.\)]\s+\w+",

    re.IGNORECASE | re.MULTILINE,

)

_RESUME_PAGE_RE = re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE)



_RESUME_SECTION_RE = re.compile(

    r"(?:^|\n)\s*(" + "|".join(_RESUME_SECTION_HEADERS) + r")\s*[:\-–—]?\s*(?:\n|$)",

    re.IGNORECASE | re.MULTILINE,

)

_RESUME_SECTION_WORD_RE = re.compile(

    r"\b(" + "|".join(_RESUME_SECTION_HEADERS) + r")\b",

    re.IGNORECASE,

)



_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b", re.IGNORECASE)

_PHONE_RE = re.compile(

    r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b"

)

_PROFILE_URL_RE = re.compile(

    r"(?:https?://)?(?:www\.)?"

    r"(?:linkedin\.com|github\.com|gitlab\.com|behance\.net|dribbble\.com)"

    r"/[\w\-./]+",

    re.IGNORECASE,

)

_PORTFOLIO_RE = re.compile(

    r"(?:https?://)?[\w.-]+\.(?:com|io|dev|me|net)/[\w\-./]*portfolio",

    re.IGNORECASE,

)



_JOB_DATE_RE = re.compile(

    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"

    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

    r"\.?\s+\d{4}\s*[-–—]\s*(?:present|current|\d{4})",

    re.IGNORECASE,

)

_YEAR_RANGE_RE = re.compile(

    r"\b(19|20)\d{2}\s*[-–—]\s*(?:present|current|(19|20)\d{2})\b",

    re.IGNORECASE,

)



_FILENAME_RESUME_RE = re.compile(r"(?:cv|resume|résumé|profile|linkedin)", re.IGNORECASE)





def _normalise(text: str) -> str:

    return re.sub(r"\s+", " ", (text or "").strip()).lower()





def _count_substrings(text: str, terms: tuple[str, ...], weight: int = 1) -> int:

    score = 0

    for term in terms:

        if term in text:

            score += weight

    return score





def _contract_structure_score(text: str) -> int:

    score = 0

    if len(_CLAUSE_NUMBER_RE.findall(text)) >= 2:

        score += _STRUCTURE_WEIGHT

    if _SECTION_HEADING_RE.search(text):

        score += _STRUCTURE_WEIGHT

    if _ARTICLE_CAPS_RE.search(text):

        score += _STRUCTURE_WEIGHT

    if _NUMBERED_ARTICLE_RE.search(text):

        score += _STRUCTURE_WEIGHT

    return score





def _count_resume_sections(text: str) -> int:
    # Only count section headers that appear as standalone line-start entries.
    # Word-level fallback caused false positives on legal contracts that naturally
    # contain words like "experience" or "project" in clause body text.
    line_hits = {m.group(1).lower() for m in _RESUME_SECTION_RE.finditer(text)}
    return len(line_hits)





def _resume_structural_score(text: str, norm: str) -> int:

    score = 0



    section_count = _count_resume_sections(text)

    if section_count >= 2:

        score += _RESUME_SECTION_WEIGHT

    if section_count >= 3:

        score += _RESUME_SECTION_WEIGHT



    if _RESUME_PAGE_RE.search(text):

        score += _RESUME_PAGINATION_WEIGHT



    job_dates = len(_JOB_DATE_RE.findall(text)) + len(_YEAR_RANGE_RE.findall(text))

    if job_dates >= 2:

        score += job_dates * _JOB_DATE_WEIGHT

    elif job_dates == 1:

        score += _JOB_DATE_WEIGHT



    if _EMAIL_RE.search(text):

        score += _CONTACT_PATTERN_WEIGHT

    if _PHONE_RE.search(text):

        score += _CONTACT_PATTERN_WEIGHT

    if _PROFILE_URL_RE.search(text) or _PORTFOLIO_RE.search(text):

        score += _CONTACT_PATTERN_WEIGHT

    if "linkedin.com" in norm or "github.com" in norm:

        score += _CONTACT_PATTERN_WEIGHT



    return score





def _contract_score(norm: str, combined: str) -> int:

    score = _count_substrings(norm, _LEGAL_CONTRACT_PHRASES, _LEGAL_PHRASE_WEIGHT)

    score += _contract_structure_score(combined)

    # Weak tokens contribute at most one point total (product-name mentions).

    weak_hits = sum(1 for kw in _CONTRACT_WEAK_KEYWORDS if kw in norm)

    if weak_hits:

        score += min(weak_hits, 1)

    return score





def _resume_score(norm: str, combined: str) -> tuple[int, int]:

    keyword_score = _count_substrings(norm, _RESUME_KEYWORDS, _KEYWORD_WEIGHT)

    structural_score = _resume_structural_score(combined, norm)

    return keyword_score + structural_score, structural_score





def _filename_resume_boost(filename: str | None) -> int:

    if not filename:

        return 0

    if _FILENAME_RESUME_RE.search(filename):

        return _FILENAME_RESUME_BOOST

    return 0





def _has_experience_education_skills_combo(norm: str) -> bool:

    has_experience = bool(re.search(r"\bexperience\b", norm, re.IGNORECASE))

    has_education = bool(re.search(r"\beducation\b", norm, re.IGNORECASE))

    has_skills = bool(re.search(r"\bskills\b", norm, re.IGNORECASE))

    return has_experience and (has_education or has_skills)





def assess_document(
    text: str,
    segments: list[dict] | None = None,
    filename: str | None = None,
) -> tuple[bool, str]:
    """
    Returns (is_contract, user_message_if_not).

    ``segments`` is optional; when provided, segment text is merged with
    ``text`` so short PDF extractions still get full coverage.
    ``filename`` optional path or stem (e.g. Aniketh_Vustepalle_CV.pdf) boosts
    resume detection.
    """
    parts = [text or ""]
    if segments:
        for seg in segments:
            parts.append(seg.get("clause_text") or seg.get("text") or "")
    combined = "\n".join(parts)
    norm = _normalise(combined)

    if len(norm) < MIN_TEXT_CHARS:
        return False, CONTRACT_GATE_MESSAGE

    contract_score = _contract_score(norm, combined)
    resume_score, resume_structural = _resume_score(norm, combined)
    resume_score += _filename_resume_boost(filename)

    section_count = _count_resume_sections(combined)

    # --- Reject rules (order: structural resume signals first) ---
    if resume_structural >= RESUME_STRUCTURAL_REJECT:
        return False, CONTRACT_GATE_MESSAGE

    if section_count >= 2:
        return False, CONTRACT_GATE_MESSAGE

    if _has_experience_education_skills_combo(norm):
        return False, CONTRACT_GATE_MESSAGE

    if resume_score >= RESUME_REJECT_THRESHOLD:
        return False, CONTRACT_GATE_MESSAGE

    legal_hits = sum(
        1 for phrase in _LEGAL_CONTRACT_PHRASES if phrase in norm
    )
    
    # STRICT ENFORCEMENT: A document MUST contain at least one strongly indicative
    # legal phrase to be considered a contract. This completely prevents lists, 
    # medical data, or non-legal text from bypassing the gate.
    if legal_hits == 0:
        return False, CONTRACT_GATE_MESSAGE

    if contract_score < MIN_CONTRACT_SCORE:
        return False, CONTRACT_GATE_MESSAGE

    if resume_score > contract_score and resume_score >= 2:
        return False, CONTRACT_GATE_MESSAGE

    return True, ""
