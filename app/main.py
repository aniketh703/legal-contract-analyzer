"""
app/main.py
===========
Flask web app for the Legal Contract Analyzer.

Routes:
    GET  /           — Upload page
    GET  /health     — Liveness check (JSON)
    POST /analyse    — Run pipeline, return HTML report
    GET  /report     — Last report.html
    GET  /export/json — Last analysis.json download
    GET  /demo       — Built-in sample contract

Run:
    python app/main.py
    # or from project root:
    flask --app app.main run --port 5000

Production (set FLASK_DEBUG=0 or unset; use a WSGI server):
    pip install gunicorn
    gunicorn -w 2 -b 127.0.0.1:5000 "app.main:app"
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, render_template, request, Response, send_file

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.pipeline import run_pipeline
from pipeline.document_gate import CONTRACT_GATE_MESSAGE

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit

ALLOWED_EXTENSIONS = {".pdf", ".txt"}

_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_TASKS: dict[str, dict] = {}


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Liveness probe for deployments and smoke checks."""
    return jsonify({"status": "ok"}), 200


@app.route("/analyse", methods=["POST"])
def analyse():
    file = request.files.get("contract")

    if not file or not file.filename:
        return Response("No file uploaded.", status=400)

    if not _allowed(file.filename):
        return Response("Only PDF and TXT files are supported.", status=400)

    suffix = Path(file.filename).suffix.lower()

    # Save to a temp file so the pipeline can read it
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        html, clauses = run_pipeline(
            path=tmp_path,
            contract_name=file.filename,
            use_embeddings=True,
            artifact_dir=ROOT,
        )
        if not clauses:
            if html == CONTRACT_GATE_MESSAGE or (
                html and "does not appear to be a legal contract" in html
            ):
                return Response(html, status=422)
            return Response(
                "No clauses could be extracted. Check that the file contains readable contract text.",
                status=422,
            )
        return Response(html, mimetype="text/html")
    except Exception as e:
        app.logger.error(f"Pipeline error: {e}", exc_info=True)
        return Response(f"Analysis failed: {e}", status=500)
    finally:
        os.unlink(tmp_path)


def _run_pipeline_async(task_id: str, file_path: str, filename: str):
    try:
        html, clauses = run_pipeline(
            path=file_path,
            contract_name=filename,
            use_embeddings=True,
            artifact_dir=ROOT,
            use_llm_summary=True,  # Optional P2 feature enabled here
        )
        if not clauses:
            _TASKS[task_id] = {"status": "error", "message": "No clauses extracted. Check that the file contains readable contract text."}
        else:
            _TASKS[task_id] = {"status": "done", "html": html}
    except Exception as e:
        app.logger.error(f"Async Pipeline error: {e}", exc_info=True)
        _TASKS[task_id] = {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)

@app.route("/analyse/async", methods=["POST"])
def analyse_async():
    file = request.files.get("contract")

    if not file or not file.filename:
        return jsonify({"error": "No file uploaded."}), 400

    if not _allowed(file.filename):
        return jsonify({"error": "Only PDF and TXT files are supported."}), 400

    suffix = Path(file.filename).suffix.lower()

    # Save to a temp file so the pipeline can read it
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    task_id = str(uuid.uuid4())
    _TASKS[task_id] = {"status": "running"}
    _EXECUTOR.submit(_run_pipeline_async, task_id, tmp_path, file.filename)
    
    return jsonify({"task_id": task_id, "status": "running"}), 202


@app.route("/status/<task_id>")
def status(task_id: str):
    task = _TASKS.get(task_id)
    if not task:
        return jsonify({"error": "Task not found."}), 404
        
    if task["status"] == "running":
        return jsonify({"status": "running"})
    elif task["status"] == "error":
        return jsonify({"status": "error", "message": task["message"]})
    elif task["status"] == "done":
        # Usually, a real queue would return a URL to the report.
        # We can just indicate it is done, or return the HTML.
        return jsonify({"status": "done", "message": "Analysis complete. View /report or /export/json."})



DEMO_CONTRACT = """
1. TERM AND TERMINATION
1.1 This Agreement shall commence on the Effective Date and continue for twelve (12) months.
1.2 Either party may terminate this Agreement upon thirty (30) days written notice in the event of a material breach that remains uncured.

2. CONFIDENTIALITY
2.1 Each party agrees to keep confidential all proprietary information disclosed by the other party and shall not disclose such information to any third party without prior written consent.

3. INDEMNIFICATION
3.1 The Service Provider shall indemnify and hold harmless the Client from any claims, damages, losses or expenses arising from breach or gross negligence.

4. NON-COMPETE
4.1 The Service Provider shall not engage in any competing business activities for a period of two (2) years after termination of this Agreement.

5. LIMITATION OF LIABILITY
5.1 In no event shall either party be liable for indirect or consequential damages.
5.2 The aggregate liability shall not exceed the total fees paid in the preceding six (6) months.

6. FORCE MAJEURE
6.1 Neither party shall be liable for delays caused by events beyond their reasonable control, including acts of God, war, or natural disasters.

7. ARBITRATION
7.1 All disputes shall be resolved by arbitration under the Arbitration and Conciliation Act, 1996, seated at New Delhi.

8. GOVERNING LAW
8.1 This Agreement shall be governed by the laws of India.

9. PAYMENT TERMS
9.1 The Client shall pay all invoices within thirty (30) days of receipt.
9.2 Late payments shall attract interest at 18% per annum from the due date.
"""


@app.route("/report")
def report():
    """Serve the last generated report.html from project root."""
    path = ROOT / "report.html"
    if not path.exists():
        return Response("No report available yet. Run an analysis first.", status=404)
    return send_file(path, mimetype="text/html")


@app.route("/export/json")
def export_json():
    """Download the last analysis as JSON."""
    path = ROOT / "analysis.json"
    if not path.exists():
        return Response("No analysis available yet. Run an analysis first.", status=404)
    return send_file(
        path,
        mimetype="application/json",
        as_attachment=True,
        download_name="contract_analysis.json",
    )


@app.route("/demo")
def demo():
    """Run pipeline on a built-in sample contract and return the report."""
    html, _ = run_pipeline(
        text=DEMO_CONTRACT,
        contract_name="Master Service Agreement (Demo)",
        use_embeddings=True,
        artifact_dir=ROOT,
    )
    return Response(html, mimetype="text/html")


def _flask_debug() -> bool:
    return os.environ.get("FLASK_DEBUG", "0").strip().lower() in ("1", "true", "yes")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=_flask_debug())
