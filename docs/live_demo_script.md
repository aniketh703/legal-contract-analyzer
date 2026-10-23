# Live Demo Script — Mid Evaluation-1

10 minutes total per student. Budget the live demo to **2-3 minutes** — it's a supporting beat inside the talk, not the whole talk. Everything below assumes you're screen-sharing a terminal + browser.

## Before the call (do this the night before, not live)

```bash
cd legal-contract-analyzer
git checkout master   # after the PR is merged
python app/main.py
```

- Confirm it starts cleanly and prints `Running on http://127.0.0.1:5000`.
- Open `http://127.0.0.1:5000/health` in a browser tab — confirm `{"status":"ok"}`. Leave this tab open; it's your fastest "is it alive" check if something looks frozen mid-call.
- Leave the terminal and an empty browser tab ready. **Don't** leave the server running for hours before the call — restart it fresh 10-15 min before your slot so a stale process/port conflict doesn't bite you live.

## During the call

**1. Set up the ask (10 sec)**
> "I'll show the pipeline end-to-end on a sample contract, then walk through what changed in the evaluation since last time."

**2. Run the demo (30 sec)**

Open `http://127.0.0.1:5000/demo` in the browser tab. This runs the built-in sample Master Service Agreement (9 clause types: Termination, Confidentiality, Indemnification, NonCompete, LiabilityCap, ForceMajeure, Arbitration, GoverningLaw, PaymentTerms) through the full pipeline with no upload needed — the safest, fastest demo path since there's no file-picker or network variability.

*(Backup if `/demo` is slow or the wifi is shaky: have `app/uploads/test_contract.txt` ready to upload via the `/` upload page instead — same idea, just pick "Try a sample contract" or the real MSA file.)*

**3. Narrate the report (60-90 sec)**

Scroll through the generated HTML report and point out, in this order:

1. **Clause segmentation** — the contract got split into distinct numbered clauses.
2. **Classification** — each clause is tagged with a type (e.g. "Indemnification") and a risk level badge (HIGH / MEDIUM / LOW).
3. **Statute citations** — pick one HIGH-risk clause (e.g. Indemnification or LiabilityCap) and show the cited Indian Contract Act 1872 section(s) (e.g. S73, S74) with the plain-English explanation — this is the retriever's statute-map + FAISS + BM25 hybrid doing its job.
4. **Disclaimer** — point out the legal disclaimer is present (shows you thought about responsible deployment, not just a model demo).

**4. Bridge to the evaluation story (source of your "actual work done")**

> "That pipeline is what I showed last time. What's new this cycle is entirely on the evaluation side — I found a bug that was inflating our accuracy number, fixed it, built a proper benchmark, and used it to test whether a legal-domain transformer beats our baseline."

Switch to the slide deck / terminal for the baseline-vs-InLegalBERT numbers (see `docs/since_last_review.md` and the README "Key findings" section for the exact figures).

## If something breaks live

- **Server won't start**: check nothing else is bound to port 5000 (`lsof -i :5000` or just restart). Have a **pre-saved `report.html` screenshot** as a static fallback slide so you're never fully blocked.
- **`/demo` hangs**: the embedding fallback (InLegalBERT) can be slow on first call (cold model load). If it's taking >15s, narrate while it loads rather than sitting in silence — or fall back to the screenshot.
- **Wifi drops mid-call**: this is exactly why the screenshot fallback exists — never let a live-demo failure eat into your 10-minute slot recovering from it.

## Optional: show the evaluation numbers live instead of on a slide

```bash
python3 -c "
import json
b = json.load(open('evaluation/results/tfidf_lr_baseline_47class.json'))
n = json.load(open('evaluation/results/inlegalbert_47class.json'))
print('TF-IDF+LR   :', b['accuracy'], b['classification_report']['macro avg']['f1-score'])
print('InLegalBERT :', n['accuracy'], n['classification_report']['macro avg']['f1-score'])
"
```

Only do this if you're confident about timing — a slide with the same numbers pre-formatted is safer under time pressure.
