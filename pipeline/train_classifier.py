"""
pipeline/train_classifier.py
============================
Train TF-IDF + Logistic Regression clause classifier from labeled CSV data.

Run from the project root:
    python pipeline/train_classifier.py

Expects:
    <project-root>/../legal_contract_clauses.csv

Writes to models/:
    tfidf_vectorizer.pkl
    clause_classifier.pkl
    risk_map.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT.parent / "legal_contract_clauses.csv"          # CUAD (US contracts)
INDIAN_CSV_PATH = ROOT / "data" / "indian_clauses.csv"         # supplemental Indian clauses
MODELS_DIR = ROOT / "models"

# Explicit aliases: CSV label -> internal type name used by generator.py
# All other CSV types are normalised to CamelCase automatically.
_TYPE_ALIASES: dict[str, str] = {
    "Cap On Liability": "LiabilityCap",
    "Ip Ownership Assignment": "IPAssignment",
    "Joint Ip Ownership": "IPAssignment",
    "Renewal Term": "RenewalTerm",
    "Notice Period To Terminate Renewal": "RenewalTerm",
    "Termination For Convenience": "Termination",
    "Force Majeure": "ForceMajeure",   # preserve camel-case used in generator.py
    "GST": "GST",
    "DPDP": "DPDP",
    "StampDuty": "StampDuty",
    "SpecificPerformance": "SpecificPerformance",
}


def normalize_type(raw: str) -> str:
    if raw in _TYPE_ALIASES:
        return _TYPE_ALIASES[raw]
    words = re.split(r"[\s\-/]+", raw.strip())
    return "".join(w.capitalize() for w in words if w)


def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"[train] CSV not found: {CSV_PATH}\nExpected path: {CSV_PATH}")

    # --- Load primary CUAD dataset (US contracts) ---
    print(f"[train] Loading {CSV_PATH.name} (CUAD) ...")
    df_cuad = pd.read_csv(CSV_PATH)
    df_cuad = df_cuad.dropna(subset=["clause_text", "clause_type", "risk_level"])
    df_cuad["clause_text"] = df_cuad["clause_text"].astype(str).str.strip()
    df_cuad = df_cuad[df_cuad["clause_text"].str.len() > 10]
    print(f"[train]   CUAD rows: {len(df_cuad):,}")

    # --- Load supplemental Indian clauses ---
    if INDIAN_CSV_PATH.exists():
        print(f"[train] Loading {INDIAN_CSV_PATH.name} (Indian supplemental) ...")
        df_indian = pd.read_csv(INDIAN_CSV_PATH)
        df_indian = df_indian.dropna(subset=["clause_text", "clause_type", "risk_level"])
        df_indian["clause_text"] = df_indian["clause_text"].astype(str).str.strip()
        df_indian = df_indian[df_indian["clause_text"].str.len() > 10]
        print(f"[train]   Indian rows: {len(df_indian):,} ({df_indian['clause_type'].nunique()} types)")
        df = pd.concat([df_cuad, df_indian], ignore_index=True)
    else:
        print(f"[train] No Indian supplemental CSV found at {INDIAN_CSV_PATH}, skipping.")
        df = df_cuad

    print(f"[train] Combined: {len(df):,} rows | {df['clause_type'].nunique()} raw types")

    df["type_norm"] = df["clause_type"].apply(normalize_type)
    print(f"[train] {df['type_norm'].nunique()} normalised types after aliasing")

    # Majority risk level per normalised type (stored uppercase)
    risk_map: dict[str, str] = (
        df.groupby("type_norm")["risk_level"]
        .apply(lambda x: x.str.upper().value_counts().idxmax())
        .to_dict()
    )

    X = df["clause_text"]
    y = df["type_norm"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[train] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("[train] Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=60_000,
        sublinear_tf=True,
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[train] Training Logistic Regression ...")
    clf = LogisticRegression(
        max_iter=1000,
        C=5.0,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[train] Test accuracy: {acc:.1%}\n")
    print(classification_report(y_test, y_pred))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(clf, MODELS_DIR / "clause_classifier.pkl")
    with open(MODELS_DIR / "risk_map.json", "w", encoding="utf-8") as fh:
        json.dump(risk_map, fh, indent=2)

    print(f"\n[train] Saved model files -> {MODELS_DIR}")
    print(f"[train] Types covered: {sorted(y.unique())}")


if __name__ == "__main__":
    main()
