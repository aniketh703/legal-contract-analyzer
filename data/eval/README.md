# Gold eval set — clauses from Indian Kanoon judgments

Independent ground truth for the v2 comparison (see `overview.md` §4). Each row is one
contract clause quoted **verbatim** in a court judgment, plus what *the court* said about it.

## Why this exists

`data/processed/clauses.jsonl` can't be the ground truth: its 212 rows were auto-extracted
(186 `auto_from_raw`), 0 are human-verified, and `statute_reference` is empty in every row.
The "expected statute" per clause type (`EXPECTED_STATUTE` in `evaluation/evaluate_retriever.py`)
is a hand-written dict by the same author as the statute map, so the eval and the system share
the same assumptions. Court-stated sections break that circularity.

## Inclusion rules

- The clause is **quoted verbatim** in the judgment (look for "reads as under", "extracted
  hereunder", "reproduced below", "reads thus"). No paraphrase, no headnote, no Kanoon summary.
- The **court itself** applies a named section to that clause. A section appearing only in a
  citation list or headnote doesn't count.
- A commercial/employment contract clause. Skip pure tax, service-law and criminal matters.
- Not in `exclude_kanoon_doc_ids.txt` (93 judgments the classifier/retriever data already used).
- At most 2 rows per judgment. The clause must make sense on its own (≤ ~150 words).
- Fill the fields by **reading**; don't have an LLM label them, or the eval inherits its blind spots.

## Columns (`kanoon_gold_template.csv`)

| Column | Content |
|---|---|
| `id` | `KAN-001`, `KAN-002`, ... |
| `url` | `https://indiankanoon.org/doc/<id>/` |
| `case_name`, `court` | e.g. `SC`, `HC-Delhi`, `HC-Bombay` |
| `judgment_date` | `YYYY-MM-DD` (year only if that's all you can find) |
| `clause_text` | verbatim clause, keep the clause number if given |
| `clause_type` | one of the 47 types in `models/risk_map.json`, else `OTHER` |
| `sections_applied` | sections the court applied to this clause, `;`-separated, as `<ACT>_S<n>` |
| `outcome` | `unenforceable` \| `enforceable` \| `partly_enforceable` (read down / narrowed) |
| `holding` | 1-2 sentences: what the court held about this clause |
| `risk_facts` | 1-3 atomic facts a correct analysis must state, separated by ` \|\| ` |
| `source_para` | paragraph number(s) where the clause and its treatment appear |
| `notes` | anything odd |

`risk_facts` are phrased as statements of law about the clause, each independently checkable
(these become the per-fact pass/fail checks for the missed-risk rate). Format only, not data:
`A restraint that operates after employment ends is void under S.27 ICA || A restraint operating
only during employment is not caught by S.27`.

Act prefixes: `ICA_S27` (Contract Act 1872), `ARB_S34` (Arbitration and Conciliation Act 1996),
`SRA_S14` (Specific Relief), `STAMP_S35`, `COPY_S18`, `CPC_S20`, `LIM_S3`, `CPA_S2`, `COMP_S3`,
`IT_S43A`, `DPDP_S4`. Any new short prefix is fine.

## Scope is computed, not entered

The retriever index holds ICA (185 sections), `ARB_S8`, `ARB_S11`, `IT_S43A`, `IT_S79`. A row is
**in scope** when every section in `sections_applied` is in the index; otherwise it is an
out-of-scope row used to test refusal. Index gaps (absent from the index): ICA S.16, S.76-123
(repealed Sale of Goods part), S.151, 161, 185, 186, 213. Record what the court applied even
when it's missing here; those rows expose real KB gaps.

## Targets

| | Minimum | Target |
|---|---|---|
| Total rows | 30 | 60 |
| In-scope, by type (S.27 restraints, S.74 damages/penalty, S.73 breach/termination, S.56 force majeure / frustration, S.124-125 indemnity, liability limits, S.28 arbitration / jurisdiction, payment / time, novation / renewal) | >= 3 each | >= 5 each |
| Out-of-scope (other statute governs: IP assignment, specific performance, stamp duty, A&C Act, consumer / competition, data protection) | 6 | 12-15 |
| Outcome `enforceable` / `partly_enforceable` | >= 30% | >= 40% of in-scope |
| Judgments dated 2024-2026 | >= 1/3 | >= 1/2 |
| Landmark / famous cases | <= 1/3 | <= 1/3 |

## Hard rules

- **Never merge these rows into** `data/processed/clauses.jsonl`, `data/indian_clauses.csv`, the
  classifier's embedding reference pool, or any training data. This set only scores.
- Keep this directory out of any glob that feeds the classifier.
- Keep `source_para` and `url` so every row can be re-audited against the judgment.
