# Legal Contract Analyzer

End-to-end pipeline to extract clause-like text from Indian Kanoon judgments, filter out court discourse, and build a labeled dataset + retrieval components for contract clause analysis.

## Repository layout

- `app/`: application code (UI/API/helpers)
- `pipeline/`: ingestion + processing pipelines
- `data/`: datasets (raw/interim/external are ignored; keep processed samples small)
- `models/`: model code / saved artifacts (small files only)
- `knowledge_base/`: KB build outputs (small files only)
- `evaluation/`: evaluation scripts and results
- `notebooks/`: experiments
- `label_studio/`: Label Studio labeling configs and helpers
- `tests/`: tests

## Quickstart (Windows)

Create and activate a venv, then install requirements:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Start Label Studio (optional):

```powershell
label-studio start --host 127.0.0.1 --port 8080
```

