# Legal Contract Analyzer

An end-to-end pipeline that accepts an Indian commercial contract (PDF or TXT), segments it into clauses, classifies each clause by type and risk level, retrieves the relevant sections of the **Indian Contract Act 1872**, and generates an HTML risk report with plain-English explanations.

---

## Architecture

```
User uploads contract (PDF / TXT)
        │
        ▼
  segmenter.py        → splits document into clause dicts
        │
        ▼
  classifier.py       → adds clause_type + risk_level
  (Stage 1: TF-IDF + LR ML model — 42 types, trained on CUAD + Indian data
   Stage 2: keyword rules fallback
   Stage 3: InLegalBERT embedding fallback)
        │
        ▼
  retriever.py        → adds retrieved_sections from ICA 1872 KB
  (statute map +        3-layer hybrid: statute map → FAISS → BM25
   FAISS + BM25 +       185 ICA sections indexed
   RRF fusion)
        │
        ▼
  generator.py        → builds HTML risk report
  (template-based,      statute-aware, varies by retrieved sections
   no LLM required)
        │
        ▼
  Flask app           → serves report at /analyse (POST)
```

### Three data sources

| Source | Used for |
|--------|---------|
| **CUAD dataset** (US commercial contracts, ~9,400 labeled clauses) | Primary classifier training — 39 clause types, text-pattern recognition |
| **Hand-crafted Indian clauses** (151 examples, 3 types) | Supplemental training — Arbitration, Confidentiality, Indemnification in Indian legal drafting style |
| **Indian Contract Act 1872 PDF** | Knowledge base — 185 legal sections retrieved to explain risk |

> **Note on training data:** The CUAD dataset originates from US commercial contracts. It is used solely for *clause type detection* — identifying what kind of clause a piece of text is. This task is largely language-pattern-based and transfers well across jurisdictions. The Indian-law risk analysis (ICA statute mapping, risk explanations, recommended actions) is handled entirely by `retriever.py` and `generator.py`, which are built specifically for the Indian Contract Act 1872.

---

## Repository layout

```
app/                  Flask web app (main.py + templates/)
pipeline/
  classifier.py       3-stage classifier: ML → keyword rules → embedding
  train_classifier.py train TF-IDF + LR model from labeled CSV data
  segmenter.py        clause segmentation
  retriever.py        ICA 1872 statute retrieval (statute map + FAISS + BM25)
  generator.py        HTML report generation
data/
  raw/                330 Indian Kanoon judgment JSONs (git-ignored)
  processed/          clauses.jsonl — curated clause examples
  indian_clauses.csv  151 hand-crafted Indian contract examples (Arbitration / Confidentiality / Indemnification)
  create_indian_clauses.py  script that generates indian_clauses.csv
knowledge_base/       FAISS index + BM25 index + chunk_registry.json (185 ICA sections)
evaluation/           evaluate_classifier.py, evaluate_retriever.py, results/
label_studio/         import_tasks.json (212 tasks), prepare_import.py
models/               tfidf_vectorizer.pkl, clause_classifier.pkl, risk_map.json
tests/                116 unit tests (pytest)
notebooks/            exploration notebooks
build_knowledge_base.py   one-time KB build script
requirements.txt
```

---

## Setup (Windows)

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

Optional: `faiss-cpu` enables semantic retrieval; without it the retriever falls back to statute map + BM25. OCR for scanned PDFs needs `pytesseract`, Pillow, and system [Tesseract](https://github.com/tesseract-ocr/tesseract).

### 3. Build the knowledge base (one-time)

Requires the **Indian Contract Act 1872** PDF placed anywhere accessible:

```powershell
python build_knowledge_base.py --pdf path\to\indian_contract_act_1872.pdf
```

This produces `knowledge_base/chunk_registry.json`, `knowledge_base/faiss_index.bin`, and `knowledge_base/bm25_corpus.pkl` (185 ICA sections indexed).

---

## Running the web app

```powershell
python app/main.py
```

Open **http://127.0.0.1:5000** in your browser.

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Upload UI |
| `/health` | GET | JSON liveness check (`{"status":"ok"}`) |
| `/analyse` | POST | Upload PDF/TXT → HTML risk report |
| `/demo` | GET | Built-in sample contract (no upload) |
| `/report` | GET | Last `report.html` |
| `/export/json` | GET | Download last `analysis.json` |

Each successful `/analyse` or `/demo` run saves `report.html` and `analysis.json` in the project root (git-ignored). Reports include a legal disclaimer and export links.

**Debug mode:** Off by default. Set `FLASK_DEBUG=1` for auto-reload and the Flask debugger.

**Scanned PDFs:** If embedded text extraction yields very little text, the segmenter attempts OCR when `pytesseract` and system Tesseract are installed.

### Production deployment (optional)

For anything beyond local dev, use a WSGI server instead of `app.run()`:

```powershell
pip install gunicorn
set FLASK_DEBUG=0
gunicorn -w 2 -b 0.0.0.0:5000 "app.main:app"
```

On Windows, use `waitress` if `gunicorn` is unavailable: `pip install waitress` then `waitress-serve --listen=127.0.0.1:5000 app.main:app`.

---

## Retraining the classifier

The ML classifier is pre-trained and saved in `models/`. To retrain (e.g. after adding more labeled data):

```powershell
# Regenerate the Indian supplemental CSV (if you edited create_indian_clauses.py)
python data/create_indian_clauses.py

# Retrain on CUAD + Indian supplemental data
python pipeline/train_classifier.py
```

The script prints a full per-type classification report and saves three files:
- `models/tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer
- `models/clause_classifier.pkl` — trained Logistic Regression model
- `models/risk_map.json` — data-driven risk level per clause type

---

## Running the pipeline from the command line

```powershell
# Re-scrape Indian Kanoon data
python -m pipeline.scraper --mode all --max-per-type 25 --resume

# Re-curate clauses from raw JSONs
python -m pipeline.curate_clauses

# Regenerate Label Studio import tasks
python label_studio\prepare_import.py
```

---

## Evaluation

```powershell
# Classifier accuracy + per-type F1
venv\Scripts\python evaluation\evaluate_classifier.py

# Retriever Hit@5 (BM25 vs. statute map + BM25)
venv\Scripts\python evaluation\evaluate_retriever.py
```

Results are written to `evaluation/results/classifier_report.json` and `evaluation/results/retriever_report.json`.

---

## Tests

```powershell
venv\Scripts\pytest tests\ -v
```

CI runs the same suite on push/PR (see `.github/workflows/ci.yml`). Offline env vars `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` avoid Hugging Face downloads in CI.

---

## Key findings

### Retriever

| Strategy | Hit@5 |
|----------|-------|
| BM25 only | 23.0% |
| Statute map + BM25 | **100%** |

The statute map layer (a hardcoded clause-type → ICA section mapping) guarantees the most relevant section is always retrieved, removing dependence on embedding similarity for known clause types.

### Classifier (TF-IDF + Logistic Regression, with keyword fallback)

Overall accuracy: **83.2%** | Weighted F1: **0.83** | Clause types covered: **42**

Training data: 9,598 labeled clauses (9,447 CUAD + 151 Indian supplemental), 80/20 train/test split.

| Clause type | F1 | Notes |
|---|---|---|
| GoverningLaw | 0.99 | |
| Arbitration | 0.95 | boosted by Indian supplemental data |
| Confidentiality | 0.95 | boosted by Indian supplemental data |
| Indemnification | 0.95 | boosted by Indian supplemental data |
| AuditRights | 0.96 | |
| Insurance | 0.96 | |
| Parties | 0.96 | |
| LiabilityCap | 0.87 | |
| RenewalTerm | 0.89 | |
| RevenueProfitSharing | 0.89 | |
| CovenantNotToSue | 0.90 | |
| Termination | 0.84 | |
| AntiAssignment | 0.86 | |
| IPAssignment | 0.85 | |

> **Previous baseline (keyword rules only):** 78.8% accuracy, 12 clause types.  
> **Current (ML + Indian supplemental):** 83.2% accuracy, 42 clause types.

---

## Design decisions

- **No LLM in the pipeline** — the generator is template-based, keeping the project fully offline and reproducible.
- **3-layer hybrid retriever** — statute map (guaranteed lookup) → FAISS (semantic) → BM25 (lexical), fused with Reciprocal Rank Fusion.
- **Label Studio annotation skipped** — 212 import tasks are prepared (`label_studio/import_tasks.json`) but annotation was not completed; classifier validation uses scraped labels.
