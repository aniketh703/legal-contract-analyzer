"""
evaluation/evaluate_classifier.py
==================================
Measures classifier accuracy against ground-truth labels in clauses.jsonl.

Works in two modes:
  --mode all       : use all clauses (scraped labels as ground truth)
  --mode verified  : use only human-verified clauses (after Label Studio)

Outputs:
  - Per-type precision, recall, F1
  - Overall macro + weighted averages
  - Confusion matrix
  - evaluation/results/classifier_report.json

Usage:
    python evaluation/evaluate_classifier.py
    python evaluation/evaluate_classifier.py --mode verified
    python evaluation/evaluate_classifier.py --no-embeddings
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
from pipeline.classifier import classify_clauses

CLAUSES_PATH = ROOT / "data" / "processed" / "clauses.jsonl"
RESULTS_DIR  = ROOT / "evaluation" / "results"

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
# Run evaluation
# ---------------------------------------------------------------------------

def run(mode: str = "all", use_embeddings: bool = False) -> dict:
    clauses = load_clauses(mode)

    if not clauses:
        print(f"No clauses found for mode='{mode}'. "
              f"Run Label Studio annotation first for --mode verified.")
        sys.exit(1)

    print(f"Evaluating classifier on {len(clauses)} clauses (mode={mode})...")

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

    labelled = classify_clauses(segments, use_embeddings=use_embeddings)

    y_true = [r["clause_type"] for r in clauses]
    y_pred = [l["clause_type"] for l in labelled]

    metrics = compute_metrics(y_true, y_pred)
    return metrics


# ---------------------------------------------------------------------------
# Print + save report
# ---------------------------------------------------------------------------

def print_report(metrics: dict, mode: str) -> None:
    print(f"\n{'='*60}")
    print(f"CLASSIFIER EVALUATION  (mode={mode}, n={metrics['n_samples']})")
    print(f"{'='*60}")
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
    payload = {"mode": mode, **metrics}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "verified"], default="all")
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()

    metrics = run(mode=args.mode, use_embeddings=not args.no_embeddings)
    print_report(metrics, args.mode)
    save_report(metrics, args.mode)


if __name__ == "__main__":
    main()
