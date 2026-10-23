# Project Overview — Legal Contract Analyzer

> Living document. Update this whenever scope, architecture, or status changes —
> it's the single place to see what this project *is*, not just what's in README.md
> (user-facing setup/usage) or HANDOFF.md (point-in-time status snapshots).
>
> Last updated: 23 Oct 2026

---

## 1. What we're building

An end-to-end pipeline that takes an **Indian commercial contract** (PDF or TXT),
and produces a clause-by-clause **risk report grounded in actual Indian law** —
not generic NLP output, but specific citations to the sections of the
**Indian Contract Act, 1872** that govern each clause.

**Who it's for:** startups and SMEs without in-house counsel, who currently have
no affordable way to get a first-pass legal risk read on a contract before it's
signed. Also doubles as a research vehicle: the evaluation-rigor work (below) is
substantial enough to be its own contribution, independent of the product.

**Context:** ECD502 Project Phase II, VNIT Nagpur, Executive M.Tech. in Applied AI.
Vustepalle Aniketh (MT24AAI187), supervised by Dr. Mansi A. Radke.
Mid Evaluation-1 presented 19 Sep 2026. Mid Evaluation-2: 21 Nov 2026. End-Term:
4–5 Jan 2027.

**Repo:** `aniketh703/legal-contract-analyzer` (GitHub), branch `master` is trunk.

---

## 2. Architecture

```
Contract (PDF/TXT)
      │
      ▼
 Segmenter            splits into clause dicts (OCR fallback via Tesseract
                       for scanned PDFs)
      │
      ▼
 Classifier            3-stage cascade:
                        1. TF-IDF + Logistic Regression — 47 clause types,
                           trained on CUAD (US) + hand-curated Indian clauses
                        2. Keyword rules — fallback
                        3. InLegalBERT embedding similarity — fallback
      │
      ▼
 Retriever              3-layer hybrid over Indian Contract Act 1872 (185 sections):
                        1. Statute map — hardcoded clause-type → section, guaranteed
                        2. FAISS — semantic search
                        3. BM25 — lexical search
                        fused via Reciprocal Rank Fusion (k=60)
      │
      ▼
 Generator              template-based HTML report (no LLM) + an "AI Narrative
                       Summary" box (distilbart-cnn-12-6 summarization of the
                       high/medium-risk clauses)
      │
      ▼
 Flask App              serves the report; routes: /, /health, /analyse, /demo,
                       /report, /export/json
```

**Also exposed as:** an MCP server (`mcp_server/`) wrapping the same pipeline as
3 agent tools — `analyze_contract`, `lookup_statute_section`, `list_clause_types`
— for any MCP-compatible client (Claude Desktop, Claude Code).

**Key design choices:**
- No LLM in the core pipeline — fully offline, reproducible, auditable.
- CUAD (US contracts) used only for clause-*type* detection (pattern-based, transfers
  across jurisdictions); all India-specific legal grounding is in `retriever.py` /
  `generator.py`.
- Two distinct uses of InLegalBERT, not to be confused: (a) pretrained, used live in
  Stage 3 of the classifier for embedding similarity; (b) a separately *fine-tuned*
  classifier (`pipeline/train_classifier_bert.py`), evaluated but **not yet wired into
  the live pipeline** — see "Features planned."

---

## 3. Features implemented

### Core pipeline
- [x] Segmenter — PDF/TXT clause extraction, OCR fallback for scanned PDFs
- [x] Classifier — 3-stage cascade, 47 clause types
- [x] Retriever — 3-layer hybrid, **100% Hit@5** (statute map + BM25)
- [x] Generator — template HTML report + AI narrative summary
- [x] Flask app — all routes, Dockerized, `docker-compose.yml`
- [x] CI — GitHub Actions, runs the test suite on push/PR (test-only, no deploy step)
- [x] 131 passing tests (`tests/`)

### Evaluation rigor (the research contribution)
- [x] Found and fixed a **self-match leak** in the real-world spot-check eval
      (embedding fallback's reference pool included the row being scored) —
      honest real-world accuracy is **42.9%**, not the previously-reported 78.8%
- [x] Built a clean, leak-free **47-class held-out benchmark**
      (`evaluation/evaluate_classifier_full47.py`) — TF-IDF+LR baseline: **82.5% acc /
      0.740 macro-F1 / 0.824 weighted-F1** on 1,940 held-out examples
- [x] Completed an **InLegalBERT fine-tune** on the identical split (Colab T4 GPU,
      28 min): **85.6% acc / 0.708 macro-F1 / 0.847 weighted-F1** — wins on accuracy
      and weighted-F1, **loses on macro-F1** (0.00 F1 on 5 thin classes, incl. `DPDP`,
      India's 2023 data-protection law). Reported honestly, not as a flat "BERT wins."
- [x] Extracted shared CSV-loading logic (`load_combined_source()`) so the baseline
      trainer, the 47-class evaluator, and the fine-tune script can't silently drift
      apart on what "the training data" means

### MCP server
- [x] 3 tools, verified end-to-end with a real MCP client (not just code review)
- [x] Found + fixed a thread-unsafe stdout-redirect bug (root-caused to
      `retriever.py`'s logging, not papered over with a fragile lock)
- [x] Found + fixed a bug where a generic error message discarded the specific
      "this isn't a legal contract" rejection reason

### Research output
- [x] IEEE conference paper draft (`paper/`, IEEEtran, compiles clean, 5 pages,
      8 citations — all verified via WebSearch, none fabricated)

### Docs & presentation
- [x] README.md, HANDOFF.md, `docs/since_last_review.md`, `docs/live_demo_script.md`,
      `docs/source_of_truth.md` (auto-generated from README+HANDOFF+eval JSON)
- [x] Colab notebook for the InLegalBERT fine-tune (`notebooks/`)
- [x] Mid Eval-1 slide deck — built into VNIT's official 11-section template
      structure, native vector diagrams (architecture, eval-leak fix, timeline,
      MCP server), results chart validated against the dataviz skill's palette rules
- [x] Branch consolidation — PR #1 (InLegalBERT fine-tune + eval fixes + MCP server +
      paper) **merged into `master`** 23 Oct 2026

### In progress (this session)
- [ ] **Public deployment to Hugging Face Spaces** — Dockerfile fixed for Spaces'
      port convention (7860, was hardcoded 5000), `|| true` removed from the
      HF-model pre-download step so a real failure fails the build loudly,
      `deploy/space_readme_header.md` + `deploy/push_to_space.sh` added,
      `label-studio` dropped from `requirements.txt` (dev-only, never imported
      at runtime). **Still needed:** you create the Space on huggingface.co
      (needs your account), first push, verify `/health` + `/demo` live, point
      README/HANDOFF at the live URL.

---

## 4. Features planned (committed, not yet started)

From HANDOFF.md's "Next Steps," toward Mid Eval-2 (21 Nov) and End-Term (4–5 Jan):

- [ ] **Wire the fine-tuned InLegalBERT classifier into the live pipeline**
      (replacing or supplementing the TF-IDF+LR stage) — with explicit guardrails
      for its known thin-class blind spots (the 5 zero-F1 classes), since blindly
      swapping it in would silently regress exactly the classes that matter most
      (e.g. `DPDP`)
- [ ] **Label Studio human annotation** — 212 tasks already prepared
      (`label_studio/import_tasks.json`), never completed. Would replace the
      scraped/unverified labels the real-world spot-check currently relies on.
- [ ] **Expand real-world spot-check coverage** beyond the current 12/47 classes
      (more scraped/curated data needed for thin classes like `LiabilityCap`,
      `Indemnification`)
- [ ] **Submit the IEEE paper** to a target venue (draft is venue-agnostic
      IEEEtran two-column format; pick the exact CFP with Dr. Radke)
- [ ] **Finish the HF Spaces deployment** (see above) and keep it live through
      Mid Eval-2

---

## 5. Features thought of (floated, not committed — revisit if useful)

- **Continuous deployment**: GitHub Action that auto-pushes to the HF Space on
  every merge to `master` (needs an `HF_TOKEN` repo secret). Skipped for now —
  manual re-push is fine at this project's update frequency.
- **Light API hardening**: `MAX_CONTENT_LENGTH` on the Flask app, now that
  `/analyse` is about to be public and unauthenticated.
- A formal OpenAPI spec for the Flask routes — considered and explicitly passed
  on when the MCP server was built, since the routes already function as a
  de facto REST API and the MCP tools cover the "structured agent access" need.

---

## 6. Checklist — where things stand right now

- [x] Core pipeline working end-to-end, tested, Dockerized
- [x] Evaluation rigor story complete (leak found+fixed, clean benchmark, honest
      InLegalBERT comparison)
- [x] MCP server built, bug-fixed, verified
- [x] IEEE paper drafted and compiling
- [x] Mid Eval-1 presented (19 Sep 2026) with a template-compliant deck
- [x] All work merged to `master` (PR #1, 23 Oct 2026)
- [ ] **Live public deployment** — in progress right now
- [ ] InLegalBERT wired into production (blocked on deciding the thin-class
      guardrail approach)
- [ ] Human-annotated eval set (Label Studio)
- [ ] Paper submitted
- [ ] Mid Eval-2 (21 Nov 2026)
- [ ] End-Term (4–5 Jan 2027)

---

## 7. Taking this to the next level — ideas beyond current scope

Roughly ordered by "cheap and high-signal" → "bigger lift":

- **Model card / evaluation transparency page** — a public page (or a `/evaluate`
  route) that shows the honest benchmark numbers, the leak-fix story, and the
  InLegalBERT per-class breakdown live from the JSON artifacts already in the
  repo. Turns the evaluation-rigor work into something a visitor can verify
  themselves, not just read about.
- **Explainability** — SHAP/LIME (or just top-weighted TF-IDF features) surfaced
  per classification in the report, so a user can see *why* a clause was flagged
  HIGH risk, not just that it was.
- **Multi-statute support** — right now grounding is ICA 1872 only; the DPDP Act
  2023 is already a recognized clause type but has no dedicated statute KB the
  way ICA does. Building that out would close the gap the InLegalBERT result
  itself pointed at.
- **Contract redlining / suggested edits** — beyond flagging risk, propose
  alternative clause language (template-based, not LLM-generated, to stay
  consistent with the project's no-LLM-in-the-pipeline principle).
- **Batch / API-key access** — a simple API-key-gated batch endpoint for
  processing multiple contracts, useful if this ever gets used by an actual
  small firm rather than one-off uploads.
- **Slack/Teams bot** — thin wrapper over the MCP server or the Flask API;
  "paste a contract, get a risk summary in-channel."
- **Contract version diffing** — upload two versions of a contract, highlight
  which risk flags changed between drafts (useful for negotiation rounds).
- **A real evaluation dashboard** — instead of static JSON files, a small page
  that tracks accuracy/macro-F1/weighted-F1 across every model version tried,
  so the "honest evaluation over impressive evaluation" principle has a
  permanent home instead of living in README prose.
- **CI/CD to the live Space** — once manual re-deploys start feeling like
  friction, the GitHub Action auto-sync from the "thought of" list above.
