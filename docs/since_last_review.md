# Since Last Review

> Written for ECD502 Project Phase II — Mid Evaluation-1, 19 Sep 2026.
> Scope: everything that changed between the previously reviewed state and this review.

## What was shown previously

As of commit `1451a57` (1 Jun 2026, "Add RAG pipeline, Docker support, classifiers, and test coverage"): a complete, working end-to-end pipeline.

- **Segmenter** — splits an uploaded contract (PDF/TXT) into clause dicts.
- **Classifier** — 3-stage cascade: TF-IDF + Logistic Regression ML model (42 types) → keyword rules fallback → InLegalBERT embedding-similarity fallback.
- **Retriever** — 3-layer hybrid over the Indian Contract Act 1872 (185 sections): statute map (guaranteed lookup) → FAISS (semantic) → BM25 (lexical), fused with Reciprocal Rank Fusion. 100% Hit@5.
- **Generator** — template-based HTML risk report, statute-aware, no LLM required.
- **Flask app** — `/`, `/health`, `/analyse`, `/demo`, `/report`, `/export/json`, Dockerized, CI'd.
- **Tests** — 116/116 passing.
- Headline classifier numbers reported at the time: 78.8% → 83.2% accuracy (keyword-only baseline vs. ML + Indian supplemental data).

## What's new since then

Two pieces of work, both from a single branch (`finetune-inlegalbert`, committed 25 Jul 2026) that had not yet been folded into the reviewed codebase — now merged in for this review.

### 1. Found and fixed a real evaluation bug

The classifier's real-world spot-check evaluation (`evaluation/evaluate_classifier.py`) had a **self-match leak**: its embedding-fallback stage searched a reference pool that included the very row being scored, so a clause could match itself at 100% confidence and inflate the accuracy number. Fixed by switching to leave-one-out reference pooling.

**Result: the honest real-world accuracy is 42.9%, not the old 78.8%.** This is not a regression — it's the same model, correctly measured. The old number was wrong.

### 2. Built a clean, leak-free 47-class benchmark

Added `evaluation/evaluate_classifier_full47.py`: scores the ML model alone (no fallback stages) against a properly held-out 20% split across **all 47 clause types** the model supports (vs. the old evaluation's 12-of-47 coverage), persisting the exact split to `evaluation/results/held_out_test_47class.jsonl` so later comparisons stay apples-to-apples.

**Result: 82.5% accuracy / macro-F1 0.740 / weighted-F1 0.824** on 1,940 held-out examples. This is the number that matters for what comes next.

### 3. Fine-tuned InLegalBERT to test a domain-specific transformer against the baseline

Wrote `pipeline/train_classifier_bert.py`: fine-tunes [`law-ai/InLegalBERT`](https://huggingface.co/law-ai/InLegalBERT) — a BERT model pretrained on Indian legal text — as a sequence classifier, scored on the *identical* held-out split as the clean 47-class benchmark above, so the two results diff directly.

The script was written but never executed at the time — no GPU was available, and it's estimated at 12+ hours on CPU. For this review, it was run on a free Google Colab T4 GPU (~15-40 min) instead.

**Result:** InLegalBERT beats the baseline on accuracy (85.6% vs. 82.5%) and weighted-F1 (0.847 vs. 0.824), but *loses* on macro-F1 (0.708 vs. 0.740) — it scores 0.00 F1 on 5 of the thinnest classes (support ≤ 10, including `DPDP`, India's 2023 data-protection law). Not a flat win: see README "Key findings" #3 for the full breakdown.

## Why this is the right "actual work done" story

The Mid Evaluation-1 rubric is literally "Actual work done, Partial results." Rather than add a new surface feature, this cycle went into the project's *evaluation rigor* — finding a bug that was making the model look better than it is, replacing it with an honest and reproducible benchmark, and using that benchmark to run a real experiment (does a legal-domain transformer beat a TF-IDF baseline on this task?). That's a genuine research contribution, not just more code.

## What's next (toward Mid Evaluation-2, 21 Nov 2026)

- Decide whether to wire the fine-tuned InLegalBERT model into `pipeline/classifier.py` as a production stage (replacing or supplementing the TF-IDF+LR stage) if it meaningfully beats the baseline.
- Consider running Label Studio annotation (212 tasks already prepared, never completed) to get a human-verified evaluation set — the current real-world spot check still relies on scraped, unverified labels.
- Expand the real-world spot-check coverage beyond 12/47 classes if more scraped data can be curated for the thin classes (LiabilityCap, Indemnification — see `HANDOFF.md` "Failed Attempts").
