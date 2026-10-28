# Gold eval set — clauses from Indian Kanoon judgments

Independent ground truth for the v2 comparison (see `overview.md` §4). Each row is one
contract clause quoted **verbatim** in a court judgment, plus what *the court* said about it.
Save your rows as `data/eval/kanoon_gold.csv` (start from `kanoon_gold_template.csv`).

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
- `sections_applied`, `outcome` and `holding` come from **your own reading** of the judgment.
  `clause_type` and `risk_facts` may be left blank: they are drafted from your `holding` and you
  confirm them before anything is scored.

## Columns (`kanoon_gold_template.csv`)

| Column | Required | Content |
|---|---|---|
| `url` | yes | `https://indiankanoon.org/doc/<id>/` |
| `clause_text` | yes | verbatim clause, keep the clause number if given |
| `sections_applied` | yes | section(s) the court applied to this clause, `;`-separated, as `<ACT>_S<n>` |
| `outcome` | yes | `unenforceable` \| `enforceable` \| `partly_enforceable` (read down / narrowed) |
| `holding` | yes | 1-2 sentences: what the court held about this clause |
| `id` | no | `KAN-001`, ... (auto-assigned if blank) |
| `case_name`, `court`, `judgment_date` | no, but wanted | e.g. `SC`, `HC-Delhi`; date as `YYYY-MM-DD` (year alone is fine) |
| `clause_type` | no | one of the 47 types in `models/risk_map.json`, else `OTHER` |
| `risk_facts` | no | 1-3 atomic facts a correct analysis must state, separated by ` \|\| ` |
| `source_para` | no | paragraph number(s) where the clause and its treatment appear |
| `notes` | no | anything odd |

`risk_facts` are statements of law about the clause, each independently checkable (the per-fact
pass/fail in the hand review). Format only, not data: `A restraint that operates after employment
ends is void under S.27 ICA || A restraint operating only during employment is not caught by S.27`.

Act prefixes: `ICA_S27` (Contract Act 1872), `ARB_S34` (Arbitration and Conciliation Act 1996),
`SRA_S14` (Specific Relief), `STAMP_S35`, `COPY_S18`, `CPC_S20`, `LIM_S3`, `CPA_S2`, `COMP_S3`,
`IT_S43A`, `DPDP_S4`. Any new short prefix is fine. A bare "Section 27" is rejected: say which Act.
The outcome column also accepts `void`, `valid`, `upheld`, `partial`, `read down` and maps them.

## Scope is computed, not entered

The retriever index holds ICA (193 sections), `ARB_S8`, `ARB_S11`, `IT_S43A`, `IT_S79`. For each row
the loader compares `sections_applied` with the index:

| Scope | Meaning | Used for |
|---|---|---|
| `in` | every applied section is in the index | retrieval metrics, and "should not refuse" |
| `mixed` | some are | retrieval metrics only (scored against the in-index sections) |
| `out` | none are | "should refuse" test |

Index gaps (absent from the index): ICA S.76-123 (the repealed Sale of Goods part) and repealed
sections such as S.239-266. Record what the court applied even when it's missing here; those rows
expose real KB gaps.

## Check your file

```
python evaluation/gold_set.py data/eval/kanoon_gold.csv
```

Prints every error (a row with an error is not scored), warnings, and progress against the targets
below. Catches: missing/odd fields, duplicate ids or clause text, judgments on the exclude list, and
clauses that overlap `data/processed/clauses.jsonl` (the classifier's reference pool).

## Targets

| | Minimum | Target |
|---|---|---|
| Total usable rows | 30 | 60 |
| In-scope, by section group (S.27, S.74, S.73, S.56, S.124-125, S.28/arbitration, S.55, S.62, liability limits) | >= 3 each | >= 5 each |
| Out-of-scope (other statute governs: IP assignment, specific performance, stamp duty, A&C Act, consumer / competition, data protection) | 6 | 12-15 |
| Outcome `enforceable` / `partly_enforceable` | >= 30% of in-scope | >= 40% |
| Judgments dated 2024-2026 | >= 1/3 | >= 1/2 |
| Landmark / famous cases | <= 1/3 | <= 1/3 |

## Scoring v1 (and later v2)

```
python evaluation/score_analyzer.py --gold data/eval/kanoon_gold.csv --review-sheet review.csv
```

Add `--no-embeddings` where torch / the InLegalBERT download isn't available (the numbers then
approximate the live pipeline). Automatic metrics: hit@1/3/5, recall@k, MRR, clause-type accuracy,
citation precision against the court's sections, unsupported-citation rate, refusal behaviour,
p50/p95 latency and cost. v2 is scored by the same code through the same `Analyzer` interface:

```
python evaluation/score_analyzer.py --analyzer v2 --gold data/eval/kanoon_gold.csv --review-sheet review_v2.csv
python evaluation/score_analyzer.py --analyzer v2 --llm-backend fake --gold ...   # dry run, no credentials
```

v2 needs `GOOGLE_CLOUD_PROJECT` and Application Default Credentials (see `pipeline/llm_client.py`).
It writes `evaluation/results/v2_<model>_baseline.json` and adds `metrics.checks` (gate and model
refusals, invalid JSON, dropped risks, ungrounded quotes).

## Hand-review checks

`--review-sheet` writes a stratified sample (default 30) with these blank columns. Read the
analyzer's output next to the court's holding and mark each:

| Check | Pass when |
|---|---|
| `faithful` | every legal claim in the output is supported by the retrieved section text or the court's holding; nothing invented |
| `missed_risk_facts` | count of `risk_facts` the output does not state or clearly imply (pass = 0) |
| `unsupported_citation` | no cited section is missing from the retrieved context (pass = N) |
| `contradicts_court` | the output never says the clause is valid when the court held it void, or the reverse (pass = N) |
| `overflags_valid_clause` | for `enforceable` rows, the output doesn't call the clause void or illegal (pass = N) |

A refused output needs no marks: it claims nothing (faithful, no unsupported citation, no
contradiction, no over-flag) and misses every risk fact. `missed_risk_facts` is only scored on
in-scope and mixed rows, so a correct refusal of an out-of-scope clause isn't penalised.

## v1 vs v2 (Step 4)

```
python evaluation/compare_analyzers.py run   --gold data/eval/kanoon_gold.csv --out-dir evaluation/results/compare \
       --contracts path/to/contract1.pdf path/to/contract2.pdf
# fill in evaluation/results/compare/review_blind.csv
python evaluation/compare_analyzers.py score --out-dir evaluation/results/compare
```

`run` scores both analyzers and writes **one blinded sheet**: the same 30 clauses (stratified by
scope and outcome) for both, each output under an anonymous id (R001, R002, ...), the two outputs
of a clause next to each other in random order. Section texts are in `review_sections.md`.
Which analyzer wrote which output is in `review_key.json`: **don't open it until you've finished
the sheet**. Blinding is partial (template and LLM outputs read differently), but you never see a
label. `--contracts` segments a few real contracts to turn cost per clause into cost per contract.

`score` writes `comparison.md` (the results table) and `comparison.json`: hand-review pass rates
with 95% Wilson intervals, exact McNemar p-values on the paired clauses, and the automatic
metrics (hit@k, unsupported-citation rate, refusals, court-outcome agreement for v2, p50/p95
latency, cost per clause and per contract). With 30 clauses an interval is roughly ±15 points,
so read small differences as ties.

## Hard rules

- **Never merge these rows into** `data/processed/clauses.jsonl`, `data/indian_clauses.csv`, the
  classifier's embedding reference pool, or any training data. This set only scores.
- Keep this directory out of any glob that feeds the classifier.
- Keep `source_para` and `url` so every row can be re-audited against the judgment.
