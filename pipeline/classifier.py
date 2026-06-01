"""
pipeline/classifier.py
======================
Labels each clause segment with clause_type and risk_level.

Strategy (three-stage):
  1. ML classifier   — TF-IDF + Logistic Regression trained on labeled CSV data.
                       Used when models/clause_classifier.pkl exists and
                       confidence >= ML_THRESHOLD.
  2. Keyword rules   — fast fallback when ML model is absent or uncertain.
  3. Embedding sim   — cosine sim against curated clauses.jsonl using InLegalBERT.

Output adds to each clause dict:
    {
        ...segmenter fields...,
        "clause_type":  "Termination",
        "risk_level":   "MEDIUM",
        "confidence":   0.87,
        "method":       "ml" | "keyword" | "embedding",
    }

Usage:
    from pipeline.classifier import classify_clauses
    labelled = classify_clauses(clauses)   # clauses from segmenter
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RISK = {
    "Termination":     "MEDIUM",
    "Arbitration":     "MEDIUM",
    "ForceMajeure":    "MEDIUM",
    "Confidentiality": "MEDIUM",
    "Jurisdiction":    "MEDIUM",
    "Indemnification": "HIGH",
    "LiabilityCap":    "HIGH",
    "IPAssignment":    "MEDIUM",
    "NonCompete":      "HIGH",
    "PaymentTerms":    "LOW",
    "GoverningLaw":    "LOW",
    "Renewal":         "LOW",
}

_UNKNOWN_TYPE = "Unknown"
_UNKNOWN_RISK = "MEDIUM"

# Minimum ML predict_proba score to trust the ML result over keyword rules
_ML_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# ML classifier (Stage 1) — lazy-loaded
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_ml_cache: dict = {}


def _load_ml_model() -> dict | None:
    """Lazy-load the trained TF-IDF + LR model. Returns None if not found."""
    if "loaded" in _ml_cache:
        return _ml_cache if _ml_cache.get("ok") else None

    _ml_cache["loaded"] = True
    vec_path = _MODELS_DIR / "tfidf_vectorizer.pkl"
    clf_path = _MODELS_DIR / "clause_classifier.pkl"
    risk_path = _MODELS_DIR / "risk_map.json"

    if not (vec_path.exists() and clf_path.exists()):
        _ml_cache["ok"] = False
        return None

    try:
        import joblib  # type: ignore[import-not-found]
        _ml_cache["vectorizer"] = joblib.load(vec_path)
        _ml_cache["clf"] = joblib.load(clf_path)
        _ml_cache["risk_map"] = json.loads(risk_path.read_text(encoding="utf-8")) if risk_path.exists() else {}
        _ml_cache["ok"] = True
    except Exception:
        _ml_cache["ok"] = False
        return None

    return _ml_cache


def _ml_classify(text: str) -> tuple[str, float] | None:
    """Return (clause_type, confidence) from the ML model, or None if unavailable."""
    cache = _load_ml_model()
    if cache is None:
        return None
    vec = cache["vectorizer"].transform([text])
    proba = cache["clf"].predict_proba(vec)[0]
    best_idx = int(proba.argmax())
    confidence = float(proba[best_idx])
    if confidence < _ML_THRESHOLD:
        return None
    return cache["clf"].classes_[best_idx], confidence


def _ml_risk(clause_type: str) -> str:
    cache = _load_ml_model()
    if cache:
        return cache["risk_map"].get(clause_type, _UNKNOWN_RISK)
    return _RISK.get(clause_type, _UNKNOWN_RISK)

# ---------------------------------------------------------------------------
# Keyword rules
# Each entry: (clause_type, [required_any], [required_all], [blocklist])
#   required_any  — at least one must match
#   required_all  — all must match (empty = no constraint)
#   blocklist     — if any match, skip this rule
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, list[str], list[str], list[str]]] = [
    (
        "Termination",
        [r"\bterminat\w*\b", r"\bnotice of termination\b", r"\bterminates?\s+this\s+agreement\b"],
        [],
        [],
    ),
    (
        "Arbitration",
        [r"\barbitrat\w*\b", r"\bconciliation\b"],
        [],
        [r"\bsection\s+\d+\s+of\s+the\s+arbitration\b"],  # avoid pure statute citations
    ),
    (
        "Confidentiality",
        [r"\bconfidential\w*\b", r"\bnon.?disclosure\b", r"\bproprietary information\b"],
        [],
        [],
    ),
    (
        "Indemnification",
        [r"\bindemnif\w*\b", r"\bhold harmless\b", r"\bdefend.*from.*claims?\b"],
        [],
        [],
    ),
    (
        "NonCompete",
        [
            r"\bnon.?compet\w*\b",
            r"\brestraint of trade\b",
            r"\bcompeting business\b",
            r"\bnot.*engag\w*.*compet\w*\b",
        ],
        [],
        [],
    ),
    (
        "IPAssignment",
        [
            r"\bintellectual property\b",
            r"\bip\s+assign\w*\b",
            r"\bassign.*(?:patent|copyright|trademark|invention)\b",
            r"\bwork.?for.?hire\b",
            r"\bowned by.*(?:company|employer|licensor)\b",
        ],
        [],
        [],
    ),
    (
        "LiabilityCap",
        [
            r"\blimit.*liabilit\w*\b",
            r"\bliabilit\w*.*cap\b",
            r"\bin no event.*liab\w*\b",
            r"\bmaximum.*liabilit\w*\b",
            r"\baggregate.*liabilit\w*\b",
            # additional patterns
            r"\bshall not exceed\b",
            r"\btotal.*liabilit\w*\b",
            r"\bliabilit\w*.*shall not exceed\b",
            r"\bmaximum exposure\b",
            r"\bexclud.*liabilit\w*\b",
            r"\bno event.*shall.*liab\w*\b",
            r"\bconsequential\s+damages?\b",
            r"\bindirect\s+damages?\b",
            r"\bincidental\s+damages?\b",
            r"\bspecial\s+damages?\b",
            r"\bnot.*liable.*(?:indirect|consequential|incidental|special)\b",
        ],
        [],
        [],
    ),
    (
        "ForceMajeure",
        [
            r"\bforce majeure\b",
            r"\bact of god\b",
            r"\bevents? beyond.*control\b",
            r"\bcircumstances? beyond.*control\b",
            r"\bneither party.*liable.*delay\b",
        ],
        [],
        [],
    ),
    (
        "PaymentTerms",
        [
            r"\bpayment\b",
            r"\binvoice\b",
            r"\bfee[s]?\b",
            r"\bremittance\b",
            # additional patterns — relaxed, no required_all
            r"\bpayable\b",
            r"\bdue date\b",
            r"\bshall be paid\b",
            r"\blate.*payment\b",
            r"\bpayment.*due\b",
            r"\bnet\s*\d+\b",
            r"\bwithin.*\d+.*days?\b",
            r"\binterest.*per annum\b",
            r"\bper annum\b",
            r"\boverdue\b",
            r"\bshall pay\b",
            r"\bpayment terms?\b",
        ],
        [],   # removed required_all — was too strict
        [],
    ),
    (
        "GoverningLaw",
        [
            r"\bgoverning law\b",
            r"\bgoverned by.*laws? of\b",
            r"\bconstrued in accordance with\b",
            r"\bchoice of law\b",
        ],
        [],
        [],
    ),
    (
        "Jurisdiction",
        [
            r"\bjurisdiction\b",
            r"\bexclusive jurisdiction\b",
            r"\bcourts? at\b",
            r"\bsubmit to.*jurisdiction\b",
        ],
        [],
        [r"\bgoverning law\b"],  # pure governing-law clauses handled above
    ),
    (
        "Renewal",
        [r"\brenew\w*\b", r"\bauto.?renew\w*\b", r"\bextension.*term\b", r"\brenewed.*automaticall\w*\b"],
        [],
        [],
    ),
]

_STRICT_EVIDENCE: dict[str, list[str]] = {
    "Confidentiality": [r"\bconfidential\w*\b", r"\bnon.?disclosure\b", r"\bproprietary information\b"],
    "Indemnification": [r"\bindemnif\w*\b", r"\bhold harmless\b", r"\bdefend\b"],
    "IPAssignment": [
        r"\bintellectual property\b",
        r"\bip\s+assign\w*\b",
        r"\bassign.*(?:patent|copyright|trademark|invention)\b",
        r"\bwork.?for.?hire\b",
        r"\bbackground ip\b",
        r"\bowned by.*(?:company|employer|licensor)\b",
    ],
    "NonCompete": [
        r"\bnon.?compet\w*\b",
        r"\brestraint of trade\b",
        r"\bcompeting business\b",
        r"\bnot.*engag\w*.*compet\w*\b",
    ],
}

_STRUCTURAL_HEADING_PATTERNS = [
    r"\brecitals?\b",
    r"\bparties\b",
    r"\bsignature(?:s| block)?\b",
    r"\bexecution\b",
    r"\bgeneral\b",
    r"\bmiscellaneous\b",
    r"\bpreamble\b",
]


def _keyword_classify(text: str) -> tuple[str, float] | None:
    """
    Returns (clause_type, confidence) if a rule fires, else None.
    Confidence is proportional to number of pattern hits.
    """
    t = text.lower()

    best_type: str | None = None
    best_hits = 0

    for clause_type, required_any, required_all, blocklist in _RULES:
        # Check blocklist first
        if any(re.search(p, t) for p in blocklist):
            continue
        # At least one required_any must match
        hits_any = sum(1 for p in required_any if re.search(p, t))
        if hits_any == 0:
            continue
        # All required_all must match
        if required_all and not all(re.search(p, t) for p in required_all):
            continue

        if hits_any > best_hits:
            best_hits = hits_any
            best_type = clause_type

    if best_type is None:
        return None

    # Confidence: 1 hit → 0.70, 2 hits → 0.82, 3+ hits → 0.90+
    confidence = min(0.70 + (best_hits - 1) * 0.10, 0.95)
    return best_type, confidence


def _is_structural_clause(text: str, heading: str = "") -> bool:
    """Return True for boilerplate sections that should not be forced into a substantive risk label."""
    haystack = f"{heading} {text}".lower()
    return any(re.search(pattern, haystack) for pattern in _STRUCTURAL_HEADING_PATTERNS)


def _supports_clause_type(clause_type: str, text: str, heading: str = "") -> bool:
    """Guard against high-confidence but implausible labels from ML or embeddings."""
    if _is_structural_clause(text, heading):
        return False

    evidence_patterns = _STRICT_EVIDENCE.get(clause_type)
    if evidence_patterns is None:
        return True

    haystack = f"{heading} {text}".lower()
    return any(re.search(pattern, haystack) for pattern in evidence_patterns)


def _accept_prediction(clause_type: str, confidence: float, text: str, heading: str = "") -> tuple[str, float] | None:
    """Keep a prediction only when the clause text supports the label."""
    if not _supports_clause_type(clause_type, text, heading):
        return None
    return clause_type, confidence


# ---------------------------------------------------------------------------
# Embedding fallback
# ---------------------------------------------------------------------------

_emb_cache: dict = {}  # lazy-loaded


def _load_embedding_model(model_name: str = "law-ai/InLegalBERT"):
    if "model" in _emb_cache:
        return _emb_cache["tokenizer"], _emb_cache["model"], _emb_cache["device"]

    import torch  # type: ignore[import-not-found]
    from transformers import AutoTokenizer, AutoModel  # type: ignore[import-not-found]

    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    _emb_cache["tokenizer"] = tokenizer
    _emb_cache["model"] = model
    _emb_cache["device"] = device
    return tokenizer, model, device


def _embed(text: str, tokenizer, model, device) -> object:
    import torch  # type: ignore[import-not-found]
    import numpy as np

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype("float32")
    emb = emb / (emb @ emb) ** 0.5  # L2 normalise
    return emb


def _load_reference_embeddings(clauses_path: Path, tokenizer, model, device):
    """Embed curated clauses once and cache."""
    if "refs" in _emb_cache:
        return _emb_cache["refs"]

    import numpy as np

    rows = [json.loads(l) for l in clauses_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    embeddings = []
    labels = []
    for row in rows:
        emb = _embed(row["clause_text"][:512], tokenizer, model, device)
        embeddings.append(emb)
        labels.append(row["clause_type"])

    _emb_cache["refs"] = (np.vstack(embeddings), labels)
    return _emb_cache["refs"]


def _embedding_classify(
    text: str,
    clauses_path: Path,
    model_name: str = "law-ai/InLegalBERT",
) -> tuple[str, float]:
    import numpy as np

    tokenizer, model, device = _load_embedding_model(model_name)
    ref_embs, ref_labels = _load_reference_embeddings(clauses_path, tokenizer, model, device)

    query_emb = _embed(text[:512], tokenizer, model, device).reshape(1, -1)
    scores = (ref_embs @ query_emb.T).flatten()  # cosine similarities

    best_idx = int(np.argmax(scores))
    confidence = float(scores[best_idx])
    return ref_labels[best_idx], max(confidence, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_clauses(
    clauses: list[dict],
    clauses_jsonl: str | Path | None = None,
    use_embeddings: bool = True,
    model_name: str = "law-ai/InLegalBERT",
) -> list[dict]:
    """
    Label each clause dict with clause_type, risk_level, confidence, method.

    Args:
        clauses:        Output from segmenter.segment_contract().
        clauses_jsonl:  Path to data/processed/clauses.jsonl (for embedding fallback).
                        Auto-detected from repo root if omitted.
        use_embeddings: Set False to disable the embedding fallback (faster, less accurate).
        model_name:     HuggingFace model for embedding fallback.

    Returns:
        Same list with added keys: clause_type, risk_level, confidence, method.
    """
    # Resolve default path to curated clauses
    if clauses_jsonl is None:
        here = Path(__file__).resolve().parent
        clauses_jsonl = here.parent / "data" / "processed" / "clauses.jsonl"
    clauses_jsonl = Path(clauses_jsonl)

    results = []
    for clause in clauses:
        text = clause.get("clause_text", "")
        heading = clause.get("heading", "")
        result = dict(clause)

        if _is_structural_clause(text, heading):
            result["clause_type"] = _UNKNOWN_TYPE
            result["risk_level"] = _UNKNOWN_RISK
            result["confidence"] = 0.0
            result["method"] = "structural"
            results.append(result)
            continue

        # Stage 1: ML classifier
        ml_result = _ml_classify(text)
        accepted = _accept_prediction(ml_result[0], ml_result[1], text, heading) if ml_result is not None else None
        if accepted is not None:
            ctype, conf = accepted
            result["clause_type"] = ctype
            result["risk_level"] = _ml_risk(ctype)
            result["confidence"] = round(conf, 3)
            result["method"] = "ml"

        else:
            # Stage 2: keyword rules
            keyword_result = _keyword_classify(text)
            accepted = _accept_prediction(keyword_result[0], keyword_result[1], text, heading) if keyword_result is not None else None
            if accepted is not None:
                ctype, conf = accepted
                result["clause_type"] = ctype
                result["risk_level"] = _RISK.get(ctype, _UNKNOWN_RISK)
                result["confidence"] = round(conf, 3)
                result["method"] = "keyword"

            # Stage 3: embedding fallback
            elif use_embeddings and clauses_jsonl.exists():
                try:
                    ctype, conf = _embedding_classify(text, clauses_jsonl, model_name)
                    accepted = _accept_prediction(ctype, conf, text, heading)
                    if accepted is not None:
                        ctype, conf = accepted
                        result["clause_type"] = ctype
                        result["risk_level"] = _RISK.get(ctype, _UNKNOWN_RISK)
                        result["confidence"] = round(conf, 3)
                        result["method"] = "embedding"
                    else:
                        result["clause_type"] = _UNKNOWN_TYPE
                        result["risk_level"] = _UNKNOWN_RISK
                        result["confidence"] = 0.0
                        result["method"] = "none"
                except Exception as e:
                    result["clause_type"] = _UNKNOWN_TYPE
                    result["risk_level"] = _UNKNOWN_RISK
                    result["confidence"] = 0.0
                    result["method"] = f"failed:{e}"
            else:
                result["clause_type"] = _UNKNOWN_TYPE
                result["risk_level"] = _UNKNOWN_RISK
                result["confidence"] = 0.0
                result["method"] = "none"

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pipeline.segmenter import segment_contract

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.classifier <contract.pdf|contract.txt>")
        sys.exit(1)

    clauses = segment_contract(path=sys.argv[1])
    labelled = classify_clauses(clauses, use_embeddings=False)

    print(f"Classified {len(labelled)} clauses\n")
    for c in labelled:
        print(f"[{c['clause_id']}] {c['clause_type']} ({c['risk_level']}) conf={c['confidence']} via {c['method']}")
        print(f"  {c['clause_text'][:100]}...")
        print()
