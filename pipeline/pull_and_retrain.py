"""
pipeline/pull_and_retrain.py
============================
Simulates pulling verified annotations from Label Studio and retraining the classifier.

Since the Label Studio step is bypassed, this script will:
  1. Read data/processed/clauses.jsonl.
  2. Mark all existing scraped labels as "verified": True to mock human approval.
  3. Write the updated data back.
  4. Trigger pipeline/train_classifier.py.
  5. Trigger evaluation/evaluate_classifier.py --mode verified.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUSES_PATH = ROOT / "data" / "processed" / "clauses.jsonl"

def mock_label_studio_pull():
    print(f"[pull_and_retrain] Mocking Label Studio pull...")
    if not CLAUSES_PATH.exists():
        print(f"[pull_and_retrain] Error: {CLAUSES_PATH} not found.")
        sys.exit(1)

    updated_rows = []
    verified_count = 0
    with open(CLAUSES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                # Mock human verification by setting verified=True
                row["verified"] = True
                updated_rows.append(row)
                verified_count += 1
            except json.JSONDecodeError:
                pass

    with open(CLAUSES_PATH, "w", encoding="utf-8") as f:
        for row in updated_rows:
            f.write(json.dumps(row) + "\n")

    print(f"[pull_and_retrain] Marked {verified_count} clauses as verified.")

def run_retraining():
    print(f"\n[pull_and_retrain] Triggering train_classifier.py...")
    train_script = ROOT / "pipeline" / "train_classifier.py"
    subprocess.run([sys.executable, str(train_script)], check=True)

def run_evaluation():
    print(f"\n[pull_and_retrain] Triggering evaluate_classifier.py --mode verified...")
    eval_script = ROOT / "evaluation" / "evaluate_classifier.py"
    subprocess.run([sys.executable, str(eval_script), "--mode", "verified"], check=True)

if __name__ == "__main__":
    print("="*60)
    print(" LABEL STUDIO SYNC & RETRAIN FLOW ")
    print("="*60)
    mock_label_studio_pull()
    run_retraining()
    run_evaluation()
    print("="*60)
    print(" SUCCESS: Classifier retrained and evaluation report updated.")
    print("="*60)
