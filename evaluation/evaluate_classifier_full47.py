"""
evaluation/evaluate_classifier_full47.py
=========================================
Full 47-class evaluation of the trained TF-IDF + Logistic Regression model
(models/clause_classifier.pkl) against a held-out slice of its own training
data -- the CUAD + Indian clauses CSVs pipeline/train_classifier.py trains on.

Unlike evaluate_classifier.py (a 12/47-class real-world spot check on scraped
Indian-judgment text), this measures the ML model itself, in isolation, on
data drawn from the same distribution it was trained on. It does not exercise
the keyword or embedding fallback stages -- those only run in
pipeline.classifier.classify_clauses(), not here.

The held-out split is computed with the exact loading + normalize_type +
train_test_split logic from pipeline/train_classifier.py (random_state=42,
stratify=y, test_size=0.2), then persisted to
evaluation/results/held_out_test_47class.jsonl on first run. Later runs reuse
that file so results stay reproducible even if train_classifier.py is re-run
later on updated source CSVs. Pass --regenerate to force a rebuild.

Usage:
    python evaluation/evaluate_classifier_full47.py
    python evaluation/evaluate_classifier_full47.py --regenerate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.train_classifier import load_combined_source

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "evaluation" / "results"
HELD_OUT_PATH = RESULTS_DIR / "held_out_test_47class.jsonl"
REPORT_PATH = RESULTS_DIR / "tfidf_lr_baseline_47class.json"


def _load_and_split() -> tuple[list[str], list[str]]:
    """Reproduce train_classifier.py's exact loading + split to recover its X_test/y_test."""
    df = load_combined_source(log_prefix="[eval47]")

    X = df["clause_text"]
    y = df["type_norm"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return list(X_test), list(y_test)


def _persist_test_set(X_test: list[str], y_test: list[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HELD_OUT_PATH, "w", encoding="utf-8") as fh:
        for text, label in zip(X_test, y_test):
            fh.write(json.dumps({"clause_text": text, "clause_type": label}) + "\n")
    print(f"[eval47] Persisted {len(X_test):,} held-out rows -> {HELD_OUT_PATH}")


def _load_persisted_test_set() -> tuple[list[str], list[str]]:
    rows = [json.loads(l) for l in HELD_OUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r["clause_text"] for r in rows], [r["clause_type"] for r in rows]


def get_test_set(regenerate: bool) -> tuple[list[str], list[str]]:
    if not regenerate and HELD_OUT_PATH.exists():
        print(f"[eval47] Loading persisted held-out split -> {HELD_OUT_PATH}")
        return _load_persisted_test_set()

    print("[eval47] Regenerating held-out split from source CSVs (pipeline/train_classifier.py logic) ...")
    X_test, y_test = _load_and_split()
    _persist_test_set(X_test, y_test)
    return X_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Rebuild the held-out split from source CSVs instead of reusing the persisted file.",
    )
    args = parser.parse_args()

    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    clf_path = MODELS_DIR / "clause_classifier.pkl"
    if not (vec_path.exists() and clf_path.exists()):
        sys.exit(f"[eval47] Model files not found in {MODELS_DIR}. Run pipeline/train_classifier.py first.")

    vectorizer = joblib.load(vec_path)
    clf = joblib.load(clf_path)

    X_test, y_test = get_test_set(args.regenerate)
    print(f"[eval47] Scoring {len(X_test):,} held-out clauses across {len(set(y_test))} classes ...")

    X_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_vec)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"FULL 47-CLASS EVALUATION  (n={len(X_test)}, classes={len(set(y_test))})")
    print(f"{'='*60}")
    print(f"\nAccuracy: {acc:.1%}\n")
    print(report_text)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Full 47-class evaluation of models/clause_classifier.pkl (ML stage only, "
            "no keyword/embedding fallback) on a held-out slice of the CUAD+Indian "
            "training data, using the same split as pipeline/train_classifier.py."
        ),
        "held_out_source": str(HELD_OUT_PATH),
        "n_test_samples": len(X_test),
        "n_classes": len(set(y_test)),
        "accuracy": round(float(acc), 4),
        "classification_report": report_dict,
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[eval47] Saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
