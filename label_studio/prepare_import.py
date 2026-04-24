"""
label_studio/prepare_import.py
================================
Converts data/processed/clauses.jsonl into a Label Studio import JSON file.

What it does:
  - Reads all clauses from clauses.jsonl
  - Skips already-verified clauses (verified=True) so re-runs don't duplicate
  - Groups by clause_type so annotators can work one type at a time
  - Writes label_studio/import_tasks.json  (load this into Label Studio)

Usage:
    python label_studio/prepare_import.py
    python label_studio/prepare_import.py --type NonCompete   # one type only
    python label_studio/prepare_import.py --unverified-only   # skip done ones
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUSES_PATH = ROOT / "data" / "processed" / "clauses.jsonl"
OUTPUT_PATH = ROOT / "label_studio" / "import_tasks.json"


def load_clauses(
    clause_type_filter: str | None = None,
    unverified_only: bool = False,
) -> list[dict]:
    if not CLAUSES_PATH.exists():
        raise FileNotFoundError(f"Not found: {CLAUSES_PATH}")

    rows = []
    for line in CLAUSES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        if clause_type_filter and r.get("clause_type") != clause_type_filter:
            continue
        if unverified_only and r.get("verified", False):
            continue

        rows.append(r)

    return rows


def to_label_studio_task(clause: dict) -> dict:
    """
    Label Studio task format:
      { "data": { ...fields referenced in XML config... } }

    Each field in "data" maps to a $variable in the XML labeling config.
    """
    return {
        "data": {
            # Displayed to annotator
            "clause_text":       clause.get("clause_text", ""),
            "contract_id":       clause.get("contract_id", ""),
            "clause_type":       clause.get("clause_type", ""),
            "risk_level":        clause.get("risk_level", ""),
            "statute_reference": clause.get("statute_reference", ""),
            "source_case":       clause.get("source_case", ""),
            "source_url":        clause.get("source_url", ""),

            # Carry-through for post-processing
            "extraction_method": clause.get("extraction_method", ""),
            "verified":          clause.get("verified", False),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default=None, help="Filter to one clause type e.g. NonCompete")
    parser.add_argument("--unverified-only", action="store_true", help="Skip already-verified clauses")
    args = parser.parse_args()

    clauses = load_clauses(
        clause_type_filter=args.type,
        unverified_only=args.unverified_only,
    )

    if not clauses:
        print("No clauses matched the filters.")
        return

    tasks = [to_label_studio_task(c) for c in clauses]

    OUTPUT_PATH.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    counts = Counter(c.get("clause_type", "?") for c in clauses)
    print(f"\nExport summary ({len(tasks)} tasks -> {OUTPUT_PATH.name}):")
    print(f"  {'Clause Type':<20} {'Tasks':>6}")
    print(f"  {'-'*28}")
    for ctype, n in sorted(counts.items()):
        print(f"  {ctype:<20} {n:>6}")
    print(f"\n  TOTAL: {len(tasks)} tasks")
    print(f"\nFile written: {OUTPUT_PATH}")
    print("\nNext steps:")
    print("  1. Start Label Studio:  label-studio start --host 127.0.0.1 --port 8080")
    print("  2. Create a new project")
    print("  3. Paste the XML from label_studio/contract_clause_review.xml as the labeling config")
    print("  4. Import label_studio/import_tasks.json as the dataset")
    print("  5. Annotate — each task has 4 steps (see XML config)")
    print("  6. Export annotations as JSON-MIN and save to label_studio/annotations_export.json")
    print("  7. Run:  python label_studio/apply_labels.py  to write verified=True back to clauses.jsonl")


if __name__ == "__main__":
    main()
