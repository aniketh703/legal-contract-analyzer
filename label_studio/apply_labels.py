"""
label_studio/apply_labels.py
==============================
Reads Label Studio exported annotations and writes corrections back
to data/processed/clauses.jsonl, setting verified=True on each done row.

Export from Label Studio:
    Project -> Export -> JSON-MIN -> save as label_studio/annotations_export.json

Usage:
    python label_studio/apply_labels.py
    python label_studio/apply_labels.py --export path/to/export.json --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUSES_PATH = ROOT / "data" / "processed" / "clauses.jsonl"
DEFAULT_EXPORT = Path(__file__).parent / "annotations_export.json"


def load_clauses() -> dict[str, dict]:
    """Load clauses keyed by contract_id."""
    clauses: dict[str, dict] = {}
    for line in CLAUSES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            clauses[r["contract_id"]] = r
        except (json.JSONDecodeError, KeyError):
            continue
    return clauses


def parse_annotation(result_list: list[dict]) -> dict:
    """
    Extract the annotator's choices from a Label Studio annotation result list.
    Returns a dict of field updates to apply to the clause.
    """
    updates: dict = {}

    for item in result_list:
        name = item.get("from_name", "")
        value = item.get("value", {})

        if name == "text_source":
            choices = value.get("choices", [])
            src = choices[0] if choices else ""
            # Mark for exclusion if not verbatim contract text
            if src in ("court_discourse", "truncated_or_intro_only"):
                updates["_exclude"] = True
            else:
                updates["_exclude"] = False

        elif name == "clause_type_fixed":
            choices = value.get("choices", [])
            if choices and choices[0] not in ("no_change", "other_or_unsure"):
                updates["clause_type"] = choices[0]

        elif name == "risk_level_fixed":
            choices = value.get("choices", [])
            if choices and choices[0] != "no_change":
                updates["risk_level"] = choices[0]

        elif name == "statute_reference":
            texts = value.get("text", [])
            if texts and texts[0].strip():
                updates["statute_reference"] = texts[0].strip()

    return updates


def apply(export_path: Path, dry_run: bool = False) -> None:
    if not export_path.exists():
        raise FileNotFoundError(f"Annotation export not found: {export_path}")

    export = json.loads(export_path.read_text(encoding="utf-8"))
    clauses = load_clauses()

    updated = 0
    excluded = 0
    not_found = 0

    for task in export:
        data = task.get("data", {})
        contract_id = data.get("contract_id", "")

        if contract_id not in clauses:
            not_found += 1
            continue

        annotations = task.get("annotations", [])
        if not annotations:
            continue

        # Use the first (or most recent) annotation
        result_list = annotations[0].get("result", [])
        updates = parse_annotation(result_list)

        if not updates:
            continue

        clause = clauses[contract_id]

        if updates.pop("_exclude", False):
            # Remove from dataset
            del clauses[contract_id]
            excluded += 1
            continue

        clause.update(updates)
        clause["verified"] = True
        updated += 1

    # Write back
    if not dry_run:
        out = "\n".join(json.dumps(r, ensure_ascii=False) for r in clauses.values()) + "\n"
        CLAUSES_PATH.write_text(out, encoding="utf-8")
        print(f"Written {len(clauses)} clauses back to {CLAUSES_PATH.name}")

    print(f"\nAnnotation apply summary:")
    print(f"  Updated + verified : {updated}")
    print(f"  Excluded (noise)   : {excluded}")
    print(f"  Not found in file  : {not_found}")
    print(f"  Remaining clauses  : {len(clauses)}")

    if dry_run:
        print("\n[DRY RUN] No files written.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", default=str(DEFAULT_EXPORT), help="Path to Label Studio JSON-MIN export")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    apply(Path(args.export), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
