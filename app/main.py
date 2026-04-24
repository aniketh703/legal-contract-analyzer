"""
app/main.py
===========
Flask web app for the Legal Contract Analyzer.

Routes:
    GET  /           — Upload page
    POST /analyse    — Run pipeline, return HTML report

Run:
    python app/main.py
    # or from project root:
    flask --app app.main run --port 5000
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, Response, send_file

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.pipeline import run_pipeline

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


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
        )
        if not clauses:
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
    return send_file(ROOT / "report.html", mimetype="text/html")


@app.route("/demo")
def demo():
    """Run pipeline on a built-in sample contract and return the report."""
    html, _ = run_pipeline(
        text=DEMO_CONTRACT,
        contract_name="Master Service Agreement (Demo)",
        use_embeddings=True,
    )
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
