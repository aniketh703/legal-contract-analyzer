"""
fix_segmentation.py
===================
Two things in this file:

  PART A - diagnose_duplicates()
    Run this FIRST to see exactly which section IDs are colliding and why.
    It reads your existing chunk_registry.json and ica_1872_clean.txt.

  PART B - segment_into_sections_v2()
    Drop-in replacement for the function in build_knowledge_base.py.
    Fixes the 231-segmented / 153-unique-IDs problem by:
      1. Merging sub-sections (73A, 73B) under their own stable IDs
      2. Deduplicating by keeping merged text for repeated IDs
      3. Preserving deterministic section order
      4. Printing a full collision report so you can verify the fix

Usage:
  # Step 1: Run diagnostics first
  python fix_segmentation.py --diagnose

  # Step 2: Once happy, rebuild with the fixed segmenter
"""

import argparse
import json
import re
from collections import Counter, OrderedDict, defaultdict


def diagnose_duplicates(
    registry_path: str = "knowledge_base/chunk_registry.json",
    clean_text_path: str = "knowledge_base/statutes/ica_1872_clean.txt",
):
    print("=" * 60)
    print("DUPLICATE SECTION ID DIAGNOSTIC")
    print("=" * 60)

    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)
    print(f"\nchunk_registry.json unique IDs : {len(registry)}")

    with open(clean_text_path, encoding="utf-8") as f:
        clean_text = f.read()

    section_pattern = re.compile(
        r"(?:^|\n)"
        r"(?:Section\s+)?"
        r"(\d{1,3}[A-Z]?)"
        r"(?:[\.\-\u2014\s]+)"
        r"([A-Z][^\n]{5,80}?)"
        r"(?:\.|\n|\u2014)",
        re.MULTILINE,
    )

    matches = list(section_pattern.finditer(clean_text))
    print(f"Total regex matches             : {len(matches)}")
    print(f"Sections dropped (duplicates)   : {len(matches) - len(registry)}")

    counts = Counter(m.group(1) for m in matches)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    print(f"\nSection numbers appearing >1 time: {len(duplicates)}")
    print("\nTop 20 most-duplicated section numbers:")
    print(f"  {'Num':<8} {'Count':<8} {'Example titles from text'}")
    print(f"  {'-'*8} {'-'*8} {'-'*40}")

    title_map = defaultdict(list)
    for m in matches:
        title_map[m.group(1)].append(m.group(2).strip())

    for num, count in sorted(duplicates.items(), key=lambda x: -x[1])[:20]:
        titles = title_map[num]
        print(f"  S{num:<7} x{count:<7} {titles[0][:45]}")
        for t in titles[1:]:
            print(f"  {'':<16} {t[:45]}")

    print("\n--- First 60 lines of ica_1872_clean.txt ---")
    lines = clean_text.split("\n")
    for i, line in enumerate(lines[:60]):
        if line.strip():
            print(f"  {i+1:>4}: {line[:100]}")

    print("\n--- Lines containing 'Section 27' or '27' ---")
    for i, line in enumerate(lines):
        if re.search(r"\b27\b", line) and len(line.strip()) > 5:
            print(f"  {i+1:>4}: {line[:120]}")

    print("\n--- Lines containing 'Section 56' or 'force majeure' ---")
    for i, line in enumerate(lines):
        if re.search(r"\b56\b|force majeure", line, re.IGNORECASE) and len(line.strip()) > 5:
            print(f"  {i+1:>4}: {line[:120]}")

    print("\n--- Lines containing 'Section 73' ---")
    for i, line in enumerate(lines):
        if re.search(r"\b73\b", line) and len(line.strip()) > 5:
            print(f"  {i+1:>4}: {line[:120]}")

    print("\n" + "=" * 60)
    print("ACTION: Copy 2-3 actual header lines from above into the chat")
    print("so the fixed regex can be tuned precisely if needed.")
    print("=" * 60)


def segment_into_sections_v2(text: str, statute_prefix: str = "ICA") -> list[dict]:
    print("\n[3/5] Segmenting into sections (v2 - dedup fix)...")

    section_pattern = re.compile(
        r"(?:^|\n)"
        r"(?:Section\s+)?"
        r"(\d{1,3}[A-Z]?)"
        r"(?:\s*[\.\-\u2014]+\s*)"
        r"([\"'\u2018\u2019\u201c\u201d]*[A-Z][^\n]{5,80})",
        re.MULTILINE,
    )

    matches = list(section_pattern.finditer(text))
    print(f"      Raw matches found       : {len(matches)}")

    section_spans: dict[str, dict] = OrderedDict()
    for i, match in enumerate(matches):
        sec_num = match.group(1).strip()
        sec_title = match.group(2).strip().lstrip("\"'\u2018\u2019\u201c\u201d").rstrip(".")
        sec_id = f"{statute_prefix}_S{sec_num}"

        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        if sec_id not in section_spans:
            section_spans[sec_id] = {
                "id": sec_id,
                "section_number": sec_num,
                "title": sec_title,
                "bodies": [body],
                "source": _statute_name(statute_prefix),
            }
        else:
            section_spans[sec_id]["bodies"].append(body)

    sections = []
    merged_count = 0
    short_dropped = 0
    collision_report = []

    for sec_id, entry in section_spans.items():
        bodies = entry["bodies"]
        if len(bodies) > 1:
            merged_count += 1
            collision_report.append(
                f"  {sec_id}: merged {len(bodies)} occurrences (title: {entry['title'][:40]}...)"
            )

        combined_body = "\n\n".join(b for b in bodies if len(b.strip()) > 10)
        if len(combined_body.strip()) < 30:
            short_dropped += 1
            continue

        if len(combined_body) > 3000:
            combined_body = combined_body[:3000] + "..."

        sections.append(
            {
                "id": sec_id,
                "section_number": entry["section_number"],
                "title": entry["title"],
                "text": f"Section {entry['section_number']}. {entry['title']}. {combined_body}",
                "source": entry["source"],
            }
        )

    print(f"      Unique section IDs       : {len(sections)}")
    print(f"      Merged (duplicate IDs)   : {merged_count}")
    print(f"      Dropped (too short)      : {short_dropped}")

    if collision_report:
        print(f"\n      Merge decisions (first {min(10, len(collision_report))}):")
        for line in collision_report[:10]:
            print(line)
        if len(collision_report) > 10:
            print(f"      ... and {len(collision_report) - 10} more merges")

    if statute_prefix == "ICA":
        kg_critical = [
            "ICA_S27",
            "ICA_S56",
            "ICA_S73",
            "ICA_S74",
            "ICA_S124",
            "ICA_S125",
            "ICA_S9",
            "ICA_S28",
            "ICA_S32",
            "ICA_S55",
            "ICA_S64",
            "ICA_S1",
        ]
        found_ids = {s["id"] for s in sections}
        print("\n      KG-critical section check:")
        all_present = True
        for kg_id in kg_critical:
            present = kg_id in found_ids
            status = "PRESENT" if present else "MISSING - CHECK PDF"
            flag = "" if present else " <-- ACTION NEEDED"
            print(f"        {kg_id:<15} {status}{flag}")
            if not present:
                all_present = False
        if all_present:
            print("      All KG-critical sections found.")
        else:
            print("\n      WARNING: Some KG sections missing.")
            print("      Run --diagnose to inspect raw header lines.")

    if sections:
        print(f"\n      Sample IDs: {[s['id'] for s in sections[:8]]}")

    return sections


def _statute_name(prefix: str) -> str:
    return {
        "ICA": "Indian Contract Act 1872",
        "ArbiAct": "Arbitration and Conciliation Act 1996",
        "ITAct": "Information Technology Act 2000",
        "CPC": "Code of Civil Procedure 1908",
    }.get(prefix, prefix)


INTEGRATION_INSTRUCTIONS = """
INTEGRATION STEPS
=================
1) In build_knowledge_base.py add:
   from fix_segmentation import segment_into_sections_v2

2) Replace:
   sections = segment_into_sections(clean)
   with:
   sections = segment_into_sections_v2(clean, statute_prefix="ICA")

3) Re-run:
   python build_knowledge_base.py --pdf "data/raw/indian_contract_act_1872.pdf"
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run duplicate diagnostic on existing chunk_registry.json",
    )
    parser.add_argument("--registry", default="knowledge_base/chunk_registry.json")
    parser.add_argument("--clean-text", default="knowledge_base/statutes/ica_1872_clean.txt")
    args = parser.parse_args()

    if args.diagnose:
        diagnose_duplicates(args.registry, args.clean_text)
        print(INTEGRATION_INSTRUCTIONS)
    else:
        print("Run with --diagnose or import segment_into_sections_v2 into build_knowledge_base.py.")
        print(INTEGRATION_INSTRUCTIONS)
