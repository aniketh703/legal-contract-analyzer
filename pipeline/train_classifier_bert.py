"""
pipeline/train_classifier_bert.py
==================================
Fine-tunes law-ai/InLegalBERT for 47-class clause-type sequence classification,
scored on the EXACT same held-out split as the TF-IDF + Logistic Regression
baseline, so the two classification_report JSONs can be diffed directly.

Test set:  loaded verbatim from evaluation/results/held_out_test_47class.jsonl
           (persisted by evaluation/evaluate_classifier_full47.py).
Train set: recomputed from the same CUAD + Indian source CSVs via the
           identical normalize_type() + train_test_split(test_size=0.2,
           random_state=42, stratify=y) logic as pipeline/train_classifier.py,
           then asserted to be the exact complement of the persisted test
           file. If source CSVs or split logic have drifted since that file
           was generated, this assertion fails loudly instead of silently
           training on rows the baseline was tested on.
Labels:    the 47 label names come from models/risk_map.json (sorted for a
           stable id2label mapping) -- the same label set
           pipeline/train_classifier.py produced.

Run from the project root:
    python pipeline/train_classifier_bert.py

Requires (already in requirements.txt): transformers, torch, scikit-learn,
pandas. Does NOT require the `datasets` package -- a plain torch Dataset
is used instead since nothing here needs its extra machinery.

Writes:
    models/inlegalbert_classifier/               fine-tuned model + tokenizer
    evaluation/results/inlegalbert_47class.json  classification_report, same
                                                  shape as tfidf_lr_baseline_47class.json

ponytail: no GPU detected in this environment (torch.cuda.is_available()
was False when this was written). Measured via smoke test: ~22.5s/step at
batch_size=16 on this CPU. Full run is 7,758 train rows -> 485 steps/epoch
x 4 epochs =~ 1,940 steps =~ 12+ hours, plus eval passes. If a CUDA box is
available, just run this script there; nothing here needs changing --
device placement is fully automatic (see module docstring note below).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.train_classifier import load_combined_source

MODEL_NAME = "law-ai/InLegalBERT"
MAX_LENGTH = 256

MODELS_DIR = ROOT / "models"
RISK_MAP_PATH = MODELS_DIR / "risk_map.json"
BERT_MODEL_DIR = MODELS_DIR / "inlegalbert_classifier"
CHECKPOINT_DIR = MODELS_DIR / "inlegalbert_checkpoints"

RESULTS_DIR = ROOT / "evaluation" / "results"
HELD_OUT_TEST_PATH = RESULTS_DIR / "held_out_test_47class.jsonl"
REPORT_PATH = RESULTS_DIR / "inlegalbert_47class.json"


# ---------------------------------------------------------------------------
# Labels -- same 47-class encoding as models/risk_map.json
# ---------------------------------------------------------------------------

def load_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    if not RISK_MAP_PATH.exists():
        sys.exit(f"[bert] {RISK_MAP_PATH} not found. Run pipeline/train_classifier.py first.")
    risk_map = json.loads(RISK_MAP_PATH.read_text(encoding="utf-8"))
    labels = sorted(risk_map.keys())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------------------
# Data -- reconstruct the exact train/test split from pipeline/train_classifier.py
# ---------------------------------------------------------------------------

def _load_persisted_test_set() -> tuple[list[str], list[str]]:
    if not HELD_OUT_TEST_PATH.exists():
        sys.exit(
            f"[bert] {HELD_OUT_TEST_PATH} not found. "
            f"Run evaluation/evaluate_classifier_full47.py first to generate it."
        )
    rows = [json.loads(l) for l in HELD_OUT_TEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r["clause_text"] for r in rows], [r["clause_type"] for r in rows]


def build_train_test_split() -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Test set comes verbatim from the persisted held-out file (the same rows
    the TF-IDF+LR baseline was scored on). Train set is recomputed via the
    identical train_test_split call from pipeline/train_classifier.py, then
    asserted to produce the exact same test partition as the persisted file.
    """
    X_test, y_test = _load_persisted_test_set()

    df = load_combined_source(log_prefix="[bert]")
    X = df["clause_text"]
    y = df["type_norm"]

    X_train, X_test_recomputed, y_train, y_test_recomputed = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = list(X_train), list(y_train)
    X_test_recomputed, y_test_recomputed = list(X_test_recomputed), list(y_test_recomputed)

    if X_test_recomputed != X_test or y_test_recomputed != y_test:
        sys.exit(
            f"[bert] Recomputed test split does not match {HELD_OUT_TEST_PATH}. "
            "Source CSVs or split logic have drifted since evaluate_classifier_full47.py "
            "was last run -- refusing to train, since train/test separation can no "
            "longer be guaranteed. Re-run `evaluate_classifier_full47.py --regenerate` "
            "and re-diff before training."
        )

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Torch Dataset (avoids adding the `datasets` package as a new dependency)
# ---------------------------------------------------------------------------

class ClauseDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[str], tokenizer, label2id: dict[str, int]):
        self.encodings = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        self.label_ids = [label2id[l] for l in labels]

    def __len__(self) -> int:
        return len(self.label_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.label_ids[idx])
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    label2id, id2label = load_label_maps()
    num_labels = len(label2id)
    print(f"[bert] {num_labels} labels loaded from {RISK_MAP_PATH}")

    X_train, y_train, X_test, y_test = build_train_test_split()
    print(f"[bert] Train: {len(X_train):,}  |  Test: {len(X_test):,} "
          f"(test set verified identical to the TF-IDF+LR baseline split)")

    print(f"[bert] Loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    print("[bert] Tokenizing (max_length=256) ...")
    train_ds = ClauseDataset(X_train, y_train, tokenizer, label2id)
    test_ds = ClauseDataset(X_test, y_test, tokenizer, label2id)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=50,
        seed=42,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    print("[bert] Fine-tuning ...")
    trainer.train()

    print(f"[bert] Saving model -> {BERT_MODEL_DIR}")
    trainer.save_model(str(BERT_MODEL_DIR))
    tokenizer.save_pretrained(str(BERT_MODEL_DIR))

    print("[bert] Scoring best checkpoint on held-out test set ...")
    pred_output = trainer.predict(test_ds)
    y_pred_ids = np.argmax(pred_output.predictions, axis=-1)
    y_pred = [id2label[i] for i in y_pred_ids]

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n[bert] Accuracy: {acc:.1%}\n")
    print(report_text)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Full 47-class evaluation of a fine-tuned law-ai/InLegalBERT sequence "
            "classifier, scored on the identical held-out split used for the "
            "TF-IDF+LR baseline (evaluation/results/tfidf_lr_baseline_47class.json) "
            "-- diff the two `classification_report` blocks directly."
        ),
        "model": MODEL_NAME,
        "held_out_source": str(HELD_OUT_TEST_PATH),
        "n_test_samples": len(X_test),
        "n_classes": num_labels,
        "accuracy": round(float(acc), 4),
        "classification_report": report_dict,
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[bert] Saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
