"""
pipeline/disclaimer.py
======================
Shared legal disclaimer text for UI and generated reports.
"""

LEGAL_DISCLAIMER_TEXT = (
    "This tool provides automated, informational contract analysis only. "
    "It is not legal advice and does not create an attorney-client relationship. "
    "Always consult a qualified lawyer before signing or relying on any contract."
)

LEGAL_DISCLAIMER_HTML = (
    '<p class="legal-disclaimer">'
    "<strong>Disclaimer:</strong> "
    + LEGAL_DISCLAIMER_TEXT
    + "</p>"
)
