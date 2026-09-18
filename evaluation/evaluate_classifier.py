"""
evaluation/evaluate_classifier.py
==================================
SCOPE: this is a real-world scraped-text spot check, NOT a full-taxonomy
evaluation. It covers only 12 of the 47 clause types the trained classifier
(models/clause_classifier.pkl) actually supports -- exactly the 12 categories
pipeline/curate_clauses.py scrapes Indian-judgment text into. The other 35
CUAD-derived types (LicenseGrant, AuditRights, Insurance, ...) have zero
representation in data/processed/clauses.jsonl and cannot be scored here.

For full 47-class coverage, run evaluate_classifier_full47.py, which scores
the ML model against a held-out slice of the actual CUAD+Indian training data.

This script exists to sanity-check the full production cascade
(ML -> keyword -> embedding) against messy, real Indian-judgment-derived
clause text, which the held-out CUAD/Indian split does not exercise.

Works in two modes:
  --mode all       : use all clauses (scraped labels as ground truth)
  --mode verified  : use only human-verified clauses (after Label Studio)

Embedding fallback (--embeddings):
  loo    (default) : leave-one-out reference pool -- excludes the row being
                      classified from its own nearest-neighbour search, so a
                      clause can't self-match. Honest numbers.
  leaky             : reproduces the old bug where the reference pool was the
                      same clauses.jsonl file used as ground truth, so a
                      clause could match itself at confidence 1.0. Kept only
                      so the leaky-vs-fixed delta can be reproduced/compared.
  off               : disable the embedding fallback entirely (ML+keyword only).

Outputs:
  - Per-type precision, recall, F1
  - Overall macro + weighted averages
  - Confusion matrix
  - evaluation/results/classifier_report.json

Usage:
    python evaluation/evaluate_classifier.py
    python evaluation/evaluate_classifier.py --mode verified
    python evaluation/evaluate_classifier.py --embeddings off
    python evaluation/evaluate_classifier.py --embeddings leaky
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.segmenter import segment_contract
from pipeline.classifier import classify_clauses, _embed, _load_embedding_model, _accept_prediction, _RISK, _UNKNOWN_RISK

CLAUSES_PATH = ROOT / "data" / "processed" / "clauses.jsonl"
RESULTS_DIR  = ROOT / "evaluation" / "results"

SCOPE_NOTE = (
    "SCOPE: real-world scraped-text spot check on 12 of the 47 clause types "
    "the trained classifier supports (see pipeline/curate_clauses.py's scrape "
    "taxonomy). NOT a full-taxonomy evaluation -- run evaluate_classifier_full47.py "
    "for that."
)

# NOTE: only 12 of the classifier's 47 trained types -- see SCOPE_NOTE above.
CLAUSE_TYPES = [
    "Termination", "Arbitration", "Confidentiality", "Indemnification",
    "NonCompete", "ForceMajeure", "IPAssignment", "LiabilityCap",
    "Jurisdiction", "PaymentTerms", "GoverningLaw", "Renewal",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    all_types = sorted(set(y_true) | set(y_pred))

    per_type: dict[str, dict] = {}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1

    total_tp = total_fp = total_fn = 0
    weighted_f1 = 0.0
    support_total = len(y_true)

    for t in all_types:
        tp = confusion[t][t]
        fp = sum(confusion[other][t] for other in all_types if other != t)
        fn = sum(confusion[t][other] for other in all_types if other != t)
        support = sum(confusion[t].values())

        p, r, f = precision_recall_f1(tp, fp, fn)
        per_type[t] = {"precision": p, "recall": r, "f1": f, "support": support}

        total_tp += tp
        total_fp += fp
        total_fn += fn
        weighted_f1 += f * support

    macro_p = round(sum(v["precision"] for v in per_type.values()) / len(per_type), 3)
    macro_r = round(sum(v["recall"]    for v in per_type.values()) / len(per_type), 3)
    macro_f = round(sum(v["f1"]        for v in per_type.values()) / len(per_type), 3)
    weighted_f1_score = round(weighted_f1 / support_total if support_total else 0, 3)

    accuracy = round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 3) if y_true else 0.0

    return {
        "per_type": per_type,
        "macro":    {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "weighted_f1": weighted_f1_score,
        "accuracy": accuracy,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "n_samples": len(y_true),
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_clauses(mode: str) -> list[dict]:
    rows = []
    for line in CLAUSES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if mode == "verified" and not r.get("verified", False):
            continue
        if r.get("clause_type", "") not in CLAUSE_TYPES:
            continue
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Leave-one-out embedding fallback (fixes the self-match leak)
# ---------------------------------------------------------------------------

def _apply_embedding_stage_leave_one_out(
    clauses: list[dict],
    labelled: list[dict],
    model_name: str = "law-ai/InLegalBERT",
) -> list[dict]:
    """
    Resolve clauses that ML+keyword left as method="none" via embedding
    similarity, excluding each query's own row from its reference pool.

    Mirrors pipeline.classifier's stage-3 embedding fallback, but computes a
    per-query reference pool (all clauses minus itself) instead of reusing a
    shared pool that includes the query -- that's what let a clause match
    itself at confidence 1.0 in the old (leaky) evaluation path.
    """
    unresolved_idx = [i for i, l in enumerate(labelled) if l["method"] == "none"]
    if not unresolved_idx:
        return labelled

    import numpy as np

    tokenizer, model, device = _load_embedding_model(model_name)

    print(f"[eval] Embedding {len(clauses)} clauses for leave-one-out fallback "
          f"({len(unresolved_idx)} unresolved by ML+keyword) ...")
    all_embs = np.vstack([_embed(c["clause_text"][:512], tokenizer, model, device) for c in clauses])
    all_labels = [c["clause_type"] for c in clauses]

    for i in unresolved_idx:
        text = clauses[i]["clause_text"]
        heading = clauses[i].get("heading", "")

        mask = np.ones(len(all_labels), dtype=bool)
        mask[i] = False  # leave-one-out: exclude the query's own row
        scores = all_embs[mask] @ all_embs[i]
        masked_labels = [lbl for j, lbl in enumerate(all_labels) if j != i]

        best_idx = int(np.argmax(scores))
        confidence = max(float(scores[best_idx]), 0.0)
        ctype = masked_labels[best_idx]

        accepted = _accept_prediction(ctype, confidence, text, heading)
        if accepted is not None:
            ctype, confidence = accepted
            labelled[i]["clause_type"] = ctype
            labelled[i]["risk_level"] = _RISK.get(ctype, _UNKNOWN_RISK)
            labelled[i]["confidence"] = round(confidence, 3)
            labelled[i]["method"] = "embedding_loo"
        # else: leave as method="none" / Unknown, same as production's guard rejection

    return labelled


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run(mode: str = "all", embeddings: str = "loo") -> dict:
    clauses = load_clauses(mode)

    if not clauses:
        print(f"No clauses found for mode='{mode}'. "
              f"Run Label Studio annotation first for --mode verified.")
        sys.exit(1)

    print(f"Evaluating classifier on {len(clauses)} clauses (mode={mode}, embeddings={embeddings})...")

    # Build fake segmenter-style dicts from ground-truth clauses
    segments = [
        {
            "clause_id":   r.get("contract_id", f"c{i}"),
            "clause_text": r["clause_text"],
            "heading":     "",
            "char_start":  0,
            "char_end":    len(r["clause_text"]),
        }
        for i, r in enumerate(clauses)
    ]

    # "leaky" reproduces the old bug via classify_clauses' built-in embedding
    # stage (shared reference pool = the ground-truth file itself). "loo" and
    # "off" both run ML+keyword only here; "loo" then fills gaps itself below.
    labelled = classify_clauses(segments, use_embeddings=(embeddings == "leaky"))

    if embeddings == "loo":
        labelled = _apply_embedding_stage_leave_one_out(clauses, labelled)

    y_true = [r["clause_type"] for r in clauses]
    y_pred = [l["clause_type"] for l in labelled]

    metrics = compute_metrics(y_true, y_pred)
    metrics["embeddings_mode"] = embeddings
    return metrics


# ---------------------------------------------------------------------------
# Print + save report
# ---------------------------------------------------------------------------

def print_report(metrics: dict, mode: str) -> None:
    print(f"\n{'='*60}")
    print(f"CLASSIFIER EVALUATION -- 12/47-CLASS SPOT CHECK  (mode={mode}, "
          f"embeddings={metrics.get('embeddings_mode', '?')}, n={metrics['n_samples']})")
    print(f"{'='*60}")
    print(SCOPE_NOTE)
    print(f"\n{'Type':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"{'-'*48}")

    for t in CLAUSE_TYPES:
        if t not in metrics["per_type"]:
            continue
        m = metrics["per_type"][t]
        print(f"{t:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>8}")

    print(f"{'-'*48}")
    mac = metrics["macro"]
    print(f"{'Macro avg':<20} {mac['precision']:>6.3f} {mac['recall']:>6.3f} {mac['f1']:>6.3f}")
    print(f"{'Weighted F1':<20} {'':>6} {'':>6} {metrics['weighted_f1']:>6.3f}")
    print(f"{'Accuracy':<20} {'':>6} {'':>6} {metrics['accuracy']:>6.3f}")
    print(f"\nNote: Using scraped labels as ground truth (mode=all).")
    print(f"For human-verified accuracy, run with --mode verified after Label Studio annotation.")


def save_report(metrics: dict, mode: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "classifier_report.json"
    payload = {"scope": SCOPE_NOTE, "mode": mode, **metrics}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "verified"], default="all")
    parser.add_argument(
        "--embeddings", choices=["loo", "leaky", "off"], default="loo",
        help="loo (default): leave-one-out fallback, no self-match leak. "
             "leaky: old buggy shared-pool behavior, for comparison only. "
             "off: ML+keyword stages only.",
    )
    args = parser.parse_args()

    metrics = run(mode=args.mode, embeddings=args.embeddings)
    print_report(metrics, args.mode)
    save_report(metrics, args.mode)


if __name__ == "__main__":
    main()
