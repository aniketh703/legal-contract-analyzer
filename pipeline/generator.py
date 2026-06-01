"""
pipeline/generator.py
=====================
Generates an HTML report from enriched clause dicts (retriever output).

Each clause gets a card with:
  - Original clause text
  - Type badge + risk badge
  - Retrieved ICA statute sections (pills)
  - Plain-English risk explanation (statute-aware template)
  - Recommended action

Bottom of report: overall contract summary.

Usage:
    from pipeline.generator import generate_report
    html = generate_report(enriched_clauses, contract_name="Agreement.pdf")
    Path("report.html").write_text(html, encoding="utf-8")
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from pipeline.disclaimer import LEGAL_DISCLAIMER_HTML

# ---------------------------------------------------------------------------
# Risk colour scheme
# ---------------------------------------------------------------------------

_RISK_COLOUR = {
    "HIGH":   {"bg": "#fee2e2", "border": "#ef4444", "text": "#991b1b", "badge_bg": "#ef4444"},
    "MEDIUM": {"bg": "#fef9c3", "border": "#eab308", "text": "#713f12", "badge_bg": "#eab308"},
    "LOW":    {"bg": "#dcfce7", "border": "#22c55e", "text": "#14532d", "badge_bg": "#22c55e"},
}

_DEFAULT_COLOUR = {"bg": "#f1f5f9", "border": "#94a3b8", "text": "#1e293b", "badge_bg": "#64748b"}

# ---------------------------------------------------------------------------
# Per-clause-type explanation templates
# Supports {statute_line} placeholder — filled from retrieved sections.
# ---------------------------------------------------------------------------

_EXPLANATIONS: dict[str, str] = {
    "NonCompete": (
        "This clause restricts competitive activity after the agreement ends. "
        "{statute_line}"
        "Non-compete clauses are frequently challenged under Indian law and may be "
        "wholly or partially unenforceable depending on their scope and duration."
    ),
    "ForceMajeure": (
        "This clause limits liability for failures caused by extraordinary events outside a party's control. "
        "{statute_line}"
        "Ensure the definition of qualifying events is neither too narrow (leaving gaps) "
        "nor too broad (excusing foreseeable delays)."
    ),
    "Termination": (
        "This clause governs how and when the agreement can be ended. "
        "{statute_line}"
        "Key risks include asymmetric termination rights, inadequate notice periods, "
        "and ambiguity around what constitutes a curable breach."
    ),
    "Indemnification": (
        "This clause requires one party to compensate the other for specified losses or liabilities. "
        "{statute_line}"
        "Watch for uncapped indemnities, broad definitions of 'losses', and whether "
        "indemnity is triggered by negligence alone or requires wilful breach."
    ),
    "LiabilityCap": (
        "This clause limits the maximum financial exposure of a party. "
        "{statute_line}"
        "Verify that the cap is proportionate to the contract value and that "
        "exclusions for fraud or gross negligence are explicitly preserved."
    ),
    "Confidentiality": (
        "This clause protects sensitive information shared between the parties. "
        "{statute_line}"
        "Check the scope of 'confidential information', the duration of the obligation, "
        "and whether standard exceptions (public domain, prior knowledge) are included."
    ),
    "Arbitration": (
        "This clause routes disputes to private arbitration rather than courts. "
        "{statute_line}"
        "This is standard for Indian commercial agreements, though parties may still negotiate "
        "the appointment mechanics, seat, governing rules, number of arbitrators, and whether "
        "interim relief from courts is preserved."
    ),
    "IPAssignment": (
        "This clause transfers intellectual property rights from one party to another. "
        "{statute_line}"
        "Ensure the assignment is limited to work created under this agreement and "
        "that pre-existing IP or background IP is explicitly carved out."
    ),
    "PaymentTerms": (
        "This clause sets out payment obligations, timelines, and consequences of late payment. "
        "{statute_line}"
        "Confirm payment timelines are realistic, late-payment interest is specified, "
        "and the invoicing process is clearly defined."
    ),
    "GoverningLaw": (
        "This clause specifies which country's or state's law governs the agreement. "
        "{statute_line}"
        "Ensure the chosen law is neutral or favourable, and is consistent with the "
        "jurisdiction clause if one is present."
    ),
    "Jurisdiction": (
        "This clause specifies which courts have authority to hear disputes. "
        "{statute_line}"
        "Exclusive jurisdiction clauses can significantly limit where you can seek relief — "
        "verify the chosen forum is accessible and practical for your situation."
    ),
    "Renewal": (
        "This clause governs how and when the agreement is extended. "
        "{statute_line}"
        "Auto-renewal clauses with short opt-out windows are a common source of "
        "unintended commitment — ensure notice periods are calendar-marked."
    ),
    # --- Types sourced from CSV training data ---
    "LicenseGrant": (
        "This clause grants permission to use intellectual property or software under specified conditions. "
        "{statute_line}"
        "Review scope (field of use, territory, sublicensing rights), exclusivity, and whether "
        "the grant survives termination."
    ),
    "AuditRights": (
        "This clause grants one party the right to inspect the other's books, records, or systems. "
        "{statute_line}"
        "Negotiate reasonable notice periods, limited scope, and cost allocation for audit findings."
    ),
    "AntiAssignment": (
        "This clause restricts transferring rights or obligations under the agreement to a third party. "
        "{statute_line}"
        "Verify whether change-of-control events trigger the restriction and whether consent "
        "cannot be unreasonably withheld."
    ),
    "Insurance": (
        "This clause requires one or both parties to maintain specified insurance coverage. "
        "{statute_line}"
        "Confirm coverage types, minimum limits, and the obligation to name the other party "
        "as an additional insured."
    ),
    "Exclusivity": (
        "This clause restricts one or both parties from dealing with competitors during the agreement. "
        "{statute_line}"
        "Assess scope, duration, and whether exclusivity is reciprocal or one-sided."
    ),
    "ChangeOfControl": (
        "This clause defines rights triggered by a change in ownership or control of a party. "
        "{statute_line}"
        "Common triggers include acquisition, merger, or sale of substantially all assets. "
        "Verify whether the other party gains termination rights on a change of control."
    ),
    "UncappedLiability": (
        "This clause explicitly removes limits on financial exposure for certain categories of loss. "
        "{statute_line}"
        "Uncapped liability for IP infringement or fraud is market-standard, but broad uncapped "
        "exposure should be carefully negotiated."
    ),
    "RenewalTerm": (
        "This clause governs automatic or optional extension of the agreement term. "
        "{statute_line}"
        "Auto-renewal with short opt-out windows is a common source of unintended commitment — "
        "ensure notice periods are calendar-marked."
    ),
    "ExpirationDate": (
        "This clause sets the fixed end date of the agreement. "
        "{statute_line}"
        "Confirm the date is correctly calculated and that obligations surviving expiration "
        "are explicitly listed."
    ),
    "MinimumCommitment": (
        "This clause requires a party to purchase or use a minimum volume or value. "
        "{statute_line}"
        "Verify the consequences of a shortfall and whether the commitment resets annually "
        "or is cumulative over the term."
    ),
    "LiquidatedDamages": (
        "This clause pre-agrees the compensation payable for a specific breach. "
        "{statute_line}"
        "Under ICA Section 74, courts may reduce penalties that are disproportionate — "
        "ensure the amount is a genuine pre-estimate of loss."
    ),
    "CovenantNotToSue": (
        "This clause is a promise not to initiate legal proceedings against the other party. "
        "{statute_line}"
        "Distinguish from a release of claims — a covenant not to sue preserves underlying "
        "rights but restricts enforcement."
    ),
    "RofrRofoRofn": (
        "This clause grants a party priority rights to purchase, offer, or negotiate before "
        "a deal is offered to others. "
        "{statute_line}"
        "Confirm trigger events, exercise periods, and pricing mechanics."
    ),
    "PostTerminationServices": (
        "This clause requires services to continue for a defined period after the agreement ends. "
        "{statute_line}"
        "Ensure the scope, duration, and pricing of post-termination obligations are clearly defined."
    ),
    "Parties": (
        "This clause identifies the legal entities entering into the agreement. "
        "{statute_line}"
        "Verify full legal names, jurisdiction of incorporation, and authorised signatories."
    ),
    "EffectiveDate": (
        "This clause specifies when the agreement comes into force. "
        "{statute_line}"
        "Confirm the effective date aligns with any conditions precedent and the intended "
        "commencement of obligations."
    ),
    "Unknown": (
        "This clause could not be automatically classified. "
        "Manual review is recommended to determine its type, risk level, and applicable statute."
    ),
}

_ACTIONS: dict[str, str] = {
    "NonCompete":      "Negotiate the scope and duration. Seek legal advice on enforceability under ICA S.27.",
    "ForceMajeure":    "Review the list of qualifying events and ensure obligations resume promptly after the event ends.",
    "Termination":     "Confirm notice periods are symmetric and that breach cure rights are clearly defined.",
    "Indemnification": "Negotiate a liability cap on indemnities and exclude consequential loss where possible.",
    "LiabilityCap":    "Verify the cap amount is proportionate and that carve-outs for fraud/IP infringement exist.",
    "Confidentiality": "Confirm the duration is reasonable (2–5 years post-term) and exceptions are standard.",
    "Arbitration":     "Agree on a neutral seat and confirm interim relief from courts is not excluded.",
    "IPAssignment":    "Carve out pre-existing IP and background IP in writing before signing.",
    "PaymentTerms":    "Confirm invoice timelines, late-payment interest rate, and dispute resolution for contested invoices.",
    "GoverningLaw":    "Ensure governing law is consistent with the jurisdiction clause and is acceptable to both parties.",
    "Jurisdiction":    "Verify the chosen forum is accessible. Consider adding an alternative for urgent interim relief.",
    "Renewal":         "Set a calendar reminder before the opt-out deadline to avoid automatic renewal.",
    "Unknown":         "Have a qualified legal professional review this clause before signing.",
    # --- Types sourced from CSV training data ---
    "LicenseGrant":            "Review exclusivity, sublicensing rights, and field-of-use restrictions carefully.",
    "AuditRights":             "Negotiate notice periods (10–30 days), limit scope to relevant records, cap audit frequency.",
    "AntiAssignment":          "Confirm change-of-control is addressed and that consent cannot be unreasonably withheld.",
    "Insurance":               "Verify required coverage types, minimum limits, and additional-insured obligations.",
    "Exclusivity":             "Assess whether exclusivity is reciprocal and negotiate carve-outs for existing relationships.",
    "ChangeOfControl":         "Confirm trigger definition and whether termination rights on change of control are mutual.",
    "UncappedLiability":       "Negotiate to limit uncapped exposure to fraud, wilful misconduct, and IP infringement only.",
    "RenewalTerm":             "Set a calendar reminder before the opt-out deadline to avoid automatic renewal.",
    "ExpirationDate":          "List all surviving obligations and confirm the expiry date is correctly calculated.",
    "MinimumCommitment":       "Model downside scenarios and confirm remedies for shortfall are proportionate.",
    "LiquidatedDamages":       "Verify the amount is a genuine loss pre-estimate; disproportionate penalties may be reduced by courts.",
    "CovenantNotToSue":        "Clarify whether this is a full release or covenant only, and confirm the scope of claims covered.",
    "RofrRofoRofn":            "Define trigger events, matching periods, and pricing mechanics precisely.",
    "PostTerminationServices": "Fix scope, duration, and pricing of wind-down services in writing before signing.",
    "Parties":                 "Verify full legal names, jurisdiction of incorporation, and signatory authority.",
    "EffectiveDate":           "Confirm alignment with conditions precedent and the intended start of obligations.",
}

# ---------------------------------------------------------------------------
# Statute-aware explanation builder
# ---------------------------------------------------------------------------

_STATUTE_LINES: dict[str, str] = {
    "ICA_S10":  "Under <strong>ICA Section 10</strong>, all agreements must satisfy essential contract requirements to be enforceable. ",
    "ICA_S11":  "Under <strong>ICA Section 11</strong>, parties must be competent to contract. ",
    "ICA_S27":  "Under <strong>ICA Section 27</strong>, agreements in restraint of trade are void in India unless a narrow exception applies. ",
    "ICA_S28":  "Under <strong>ICA Section 28</strong>, clauses that restrict legal proceedings are generally void, with exceptions for arbitration. ",
    "ICA_S55":  "Under <strong>ICA Section 55</strong>, time-of-the-essence provisions have specific enforceability requirements. ",
    "ICA_S56":  "Under <strong>ICA Section 56</strong>, agreements to do impossible acts are void — this is the statutory basis for force majeure in India. ",
    "ICA_S62":  "Under <strong>ICA Section 62</strong>, novation, rescission, and alteration of contracts are recognised. ",
    "ICA_S73":  "Under <strong>ICA Section 73</strong>, compensation is available for losses naturally arising from a breach. ",
    "ICA_S74":  "Under <strong>ICA Section 74</strong>, liquidated damages and penalty clauses are subject to court scrutiny on reasonableness. ",
    "ICA_S124": "Under <strong>ICA Section 124</strong>, a contract of indemnity is defined as a promise to save the other party from loss. ",
    "ICA_S125": "Under <strong>ICA Section 125</strong>, the indemnified party may recover all damages, costs, and sums paid in a suit. ",
    "ICA_S23":  "Under <strong>ICA Section 23</strong>, contracts with unlawful consideration or objects are void. ",
    "ICA_S37":  "Under <strong>ICA Section 37</strong>, assignment of a contract requires the other party's consent unless usage permits otherwise. ",
    "ICA_S39":  "Under <strong>ICA Section 39</strong>, refusal to perform a contract may amount to a breach entitling the other party to remedies. ",
    "ICA_S215": "Under <strong>ICA Section 215</strong>, a principal's rights when an agent acts on their own account are prescribed. ",
    "ICA_S222": "Under <strong>ICA Section 222</strong>, agents are entitled to indemnification for consequences of lawful acts. ",
}


def _build_explanation(clause_type: str, retrieved_sections: list[dict]) -> str:
    template = _EXPLANATIONS.get(clause_type, _EXPLANATIONS["Unknown"])

    # Build statute line from retrieved sections that have a known blurb
    statute_parts = []
    for sec in retrieved_sections:
        sid = sec.get("section_id", "")
        if sid in _STATUTE_LINES:
            statute_parts.append(_STATUTE_LINES[sid])

    statute_line = statute_parts[0] if statute_parts else ""
    return template.replace("{statute_line}", statute_line)


# ---------------------------------------------------------------------------
# HTML primitives
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f8fafc;
    color: #1e293b;
    padding: 2rem;
    line-height: 1.6;
}
h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }
.meta { color: #64748b; font-size: 0.85rem; margin-bottom: 2rem; }
.clause-card {
    background: #fff;
    border-radius: 10px;
    border-left: 5px solid #94a3b8;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}
.card-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
}
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.clause-id { font-size: 0.78rem; color: #94a3b8; margin-left: auto; }
.clause-text {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #475569;
    margin-bottom: 0.9rem;
    font-style: italic;
    max-height: 120px;
    overflow-y: auto;
}
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 0.35rem;
}
.statute-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.9rem; }
.pill {
    background: #e0f2fe;
    color: #0369a1;
    border-radius: 4px;
    padding: 0.18rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: default;
}
.pill.guaranteed { background: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }
.explanation { font-size: 0.9rem; color: #334155; margin-bottom: 0.75rem; }
.action-box {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    font-size: 0.85rem;
    color: #166534;
}
.action-box strong { margin-right: 0.4rem; }

/* Summary */
.summary-card {
    background: #1e293b;
    color: #f1f5f9;
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin-top: 2rem;
}
.summary-card h2 { font-size: 1.15rem; margin-bottom: 1rem; }
.summary-grid { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.summary-stat { text-align: center; }
.summary-stat .number { font-size: 2rem; font-weight: 800; }
.summary-stat .label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; }
.high-num   { color: #f87171; }
.medium-num { color: #fbbf24; }
.low-num    { color: #4ade80; }
.verdict {
    margin-top: 0.75rem;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 500;
}
.verdict.high   { background: #7f1d1d; color: #fecaca; }
.verdict.medium { background: #713f12; color: #fef9c3; }
.verdict.low    { background: #14532d; color: #dcfce7; }
.legal-disclaimer {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #9a3412;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}
.report-exports {
    margin-top: 0.75rem;
    font-size: 0.82rem;
}
.report-exports a {
    color: #93c5fd;
    margin-right: 1rem;
    text-decoration: none;
    cursor: pointer;
}
.report-exports a:hover { text-decoration: underline; }

/* Print styling for PDF export */
@media print {
    body { background: #fff; color: #000; padding: 0; }
    .clause-card { box-shadow: none; border: 1px solid #ccc; page-break-inside: avoid; }
    .summary-card { background: #fff; color: #000; border: 1px solid #ccc; page-break-inside: avoid; }
    .report-exports, .action-box { display: none !important; }
    .verdict.high { background: #fff; color: #7f1d1d; border: 2px solid #7f1d1d; }
    .verdict.medium { background: #fff; color: #713f12; border: 2px solid #713f12; }
    .verdict.low { background: #fff; color: #14532d; border: 2px solid #14532d; }
    .pill { border: 1px solid #0369a1; }
    .pill.guaranteed { border: 1px solid #1d4ed8; font-weight: bold; }
}
"""


def _badge(label: str, bg_colour: str) -> str:
    return f'<span class="badge" style="background:{bg_colour}">{label}</span>'


def _clause_card_html(clause: dict) -> str:
    risk = clause.get("risk_level", "MEDIUM")
    colours = _RISK_COLOUR.get(risk, _DEFAULT_COLOUR)
    ctype = clause.get("clause_type", "Unknown")
    sections = clause.get("retrieved_sections", [])

    explanation = _build_explanation(ctype, sections)
    action = _ACTIONS.get(ctype, _ACTIONS["Unknown"])

    # Statute pills
    pills_html = ""
    if sections:
        pills = []
        for sec in sections:
            sid = sec.get("section_id", "")
            title = sec.get("title", "")
            src = sec.get("source", "")
            cls = "pill guaranteed" if src == "statute_map" else "pill"
            tooltip = title[:60] if title else sid
            pills.append(f'<span class="{cls}" title="{tooltip}">{sid}</span>')
        pills_html = f"""
        <div class="section-label">Relevant ICA Statutes</div>
        <div class="statute-pills">{''.join(pills)}</div>"""

    type_badge = _badge(ctype, "#6366f1")
    risk_badge = _badge(risk, colours["badge_bg"])

    clause_text_escaped = (
        clause.get("clause_text", "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return f"""
<div class="clause-card" style="border-left-color:{colours['border']};background:{colours['bg']}">
  <div class="card-header">
    {type_badge}
    {risk_badge}
    <span class="clause-id">{clause.get('clause_id','')}</span>
  </div>
  <div class="clause-text">{clause_text_escaped}</div>
  {pills_html}
  <div class="section-label">Analysis</div>
  <div class="explanation">{explanation}</div>
  <div class="action-box"><strong>Recommended action:</strong>{action}</div>
</div>"""


def _summary_html(clauses: list[dict], contract_name: str) -> str:
    high   = sum(1 for c in clauses if c.get("risk_level") == "HIGH")
    medium = sum(1 for c in clauses if c.get("risk_level") == "MEDIUM")
    low    = sum(1 for c in clauses if c.get("risk_level") == "LOW")
    total  = len(clauses)

def _summary_html(clauses: list[dict], contract_name: str, use_llm_summary: bool = False) -> str:
    counts = {"high": 0, "medium": 0, "low": 0}
    for c in clauses:
        level = c.get("risk_level", "MEDIUM").lower()
        counts[level] = counts.get(level, 0) + 1
    
    total = len(clauses)
    high_risk_score = counts["high"] * 3 + counts["medium"] * 1

    if high_risk_score >= 6:
        verdict_cls = "high"
        verdict_text = f"This contract contains significant risk areas (Score: {high_risk_score}). Some HIGH risk items are materially important financial or operational terms, not necessarily unfair or unusual. Legal review is strongly recommended before signing."
    elif high_risk_score >= 3:
        verdict_cls = "medium"
        verdict_text = "This contract has moderate risk areas. Consider negotiating flagged clauses before signing, especially where they affect financial exposure, exit rights, or operational control."
    else:
        verdict_cls = "low"
        verdict_text = "This contract appears relatively low-risk. A final legal review is still advisable."

    lease_watchlist_html = ""
    lease_indicators = {"Lease", "Termination", "Renewal", "PaymentTerms", "GoverningLaw", "Jurisdiction"}
    clause_types = {c.get("clause_type", "Unknown") for c in clauses}
    if "Lease" in contract_name or len(lease_indicators.intersection(clause_types)) >= 3:
        lease_watchlist_html = """
  <div style="margin-top: 0.9rem; padding: 0.9rem 1rem; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #cbd5e1; font-size: 0.88rem;">
    <strong style="color:#e2e8f0;">Commercial lease watchlist:</strong> confirm fit-out restoration obligations, who bears structural compliance upgrades, whether CAM escalations are capped, and whether force majeure affects rent suspension or only performance timelines.
  </div>"""

    # Optional LLM Narrative Summary
    llm_summary_html = ""
    if use_llm_summary and clauses:
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
            import warnings
            warnings.filterwarnings("ignore")
            summary_text = " ".join([c["clause_text"] for c in clauses if c.get("risk_level", "low") in ["HIGH", "MEDIUM"]])
            if len(summary_text.split()) > 50:
                summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
                truncated_text = " ".join(summary_text.split()[:700])
                res = summarizer(truncated_text, max_length=130, min_length=30, do_sample=False)
                summary_val = res[0]["summary_text"]
                llm_summary_html = f"<div style='margin-top: 1rem; padding: 1rem; background: #f8fafc; border-left: 4px solid #3b82f6; color:#1e293b;'><strong>AI Narrative Summary:</strong> {summary_val}</div>"
        except Exception as e:
            llm_summary_html = f"<div style='margin-top: 1rem; color: #ef4444;'><em>Failed to generate LLM summary: {e}</em></div>"

    # Clause type breakdown
    from collections import Counter
    type_counts = Counter(c.get("clause_type", "Unknown") for c in clauses)
    type_rows = "".join(f"<tr><td>{t}</td><td>{n}</td></tr>" for t, n in type_counts.most_common())

    return f"""
<div class="summary-card">
  <h2>Contract Summary — {contract_name}</h2>
  <div class="summary-grid">
    <div class="summary-stat">
      <div class="number">{total}</div>
      <div class="label">Total Clauses</div>
    </div>
    <div class="summary-stat">
      <div class="number high-num">{counts['high']}</div>
      <div class="label">High Risk</div>
    </div>
    <div class="summary-stat">
      <div class="number medium-num">{counts['medium']}</div>
      <div class="label">Medium Risk</div>
    </div>
    <div class="summary-stat">
      <div class="number low-num">{counts['low']}</div>
      <div class="label">Low Risk</div>
    </div>
  </div>
  {llm_summary_html}
  <table style="border-collapse:collapse;font-size:0.85rem;width:100%;margin-top:1rem;margin-bottom:0.75rem">
    <thead>
      <tr style="color:#94a3b8">
        <th style="text-align:left;padding:0.2rem 0.5rem">Clause Type</th>
        <th style="text-align:left;padding:0.2rem 0.5rem">Count</th>
      </tr>
    </thead>
    <tbody style="color:#e2e8f0">
      {type_rows}
    </tbody>
  </table>
  <div class="verdict {verdict_cls}">{verdict_text}</div>
    {lease_watchlist_html}
    <div style="margin-top:0.75rem;font-size:0.8rem;color:#94a3b8;line-height:1.5">
        This assessment is commercial and structural, not formal legal advice.
    </div>
  <div class="report-exports">
        <a data-app-path="/export/json" href="/export/json" download>Download JSON analysis</a>
        <a data-app-path="/report" href="/report" target="_blank" rel="noopener">Open HTML report</a>
        <a href="#" onclick="window.print(); return false;">Save as PDF</a>
  </div>
    <script>
        (function () {{
            var openerOrigin = window.opener && window.opener.location && window.opener.location.origin;
            var currentOrigin = window.location && window.location.origin;
            var origin = (openerOrigin && openerOrigin !== "null")
                ? openerOrigin
                : ((currentOrigin && currentOrigin !== "null") ? currentOrigin : "");

            if (!origin) return;

            document.querySelectorAll(".report-exports a[data-app-path]").forEach(function (link) {{
                var path = link.getAttribute("data-app-path");
                if (path) link.href = origin + path;
            }});
        }})();
    </script>
  <div style="margin-top:0.75rem;font-size:0.75rem;color:#64748b">
    Generated by Legal Contract Analyzer &middot; {datetime.now().strftime('%d %b %Y, %H:%M')} &middot; Statutes: Indian Contract Act 1872
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    clauses: list[dict],
    contract_name: str = "Contract",
    output_path: str | Path | None = None,
    use_llm_summary: bool = False
) -> str:
    """
    Generate an HTML analysis report for a list of enriched clause dicts.

    Args:
        clauses:       Output from retriever.retrieve_statutes().
        contract_name: Display name shown in the report header.
        output_path:   If provided, writes the HTML to this file path.
        use_llm_summary: If True, uses transformers to generate a summary.

    Returns:
        HTML string.
    """
    date_str = datetime.now().strftime("%d %B %Y")
    total = len(clauses)
    high  = sum(1 for c in clauses if c.get("risk_level") == "HIGH")

    cards_html = "\n".join(_clause_card_html(c) for c in clauses)
    summary = _summary_html(clauses, contract_name)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Contract Analysis — {contract_name}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>Contract Risk Analysis</h1>
  <div class="meta">
    {contract_name} &nbsp;&middot;&nbsp; {total} clauses analysed
    &nbsp;&middot;&nbsp; {high} high-risk &nbsp;&middot;&nbsp; {date_str}
  </div>

  {LEGAL_DISCLAIMER_HTML}

  {cards_html}
  {summary}
</body>
</html>"""

    if output_path is not None:
        Path(output_path).write_text(html, encoding="utf-8")
        print(f"[generator] Report saved -> {output_path}")

    return html


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pipeline.segmenter import segment_contract
    from pipeline.classifier import classify_clauses
    from pipeline.retriever import retrieve_statutes

    contract_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else "report.html"

    if contract_path:
        clauses = segment_contract(path=contract_path)
    else:
        # Built-in demo contract
        DEMO = """
1. TERMINATION
1.1 Either party may terminate this Agreement upon thirty (30) days written notice in the event of a material breach.

2. NON-COMPETE
2.1 The employee shall not engage in any competing business activities for two (2) years after termination.

3. INDEMNIFICATION
3.1 The Service Provider shall indemnify and hold harmless the Client from any claims or expenses arising from breach or negligence.

4. FORCE MAJEURE
4.1 Neither party shall be liable for delays caused by events beyond their reasonable control.

5. ARBITRATION
5.1 All disputes shall be resolved by arbitration under the Arbitration and Conciliation Act, 1996, seated at New Delhi.

6. GOVERNING LAW
6.1 This Agreement is governed by the laws of India.
"""
        clauses = segment_contract(text=DEMO, doc_id="demo")

    classified = classify_clauses(clauses, use_embeddings=False)
    enriched   = retrieve_statutes(classified)
    generate_report(enriched, contract_name=contract_path or "Demo Contract", output_path=out_path)
    print(f"Open {out_path} in a browser to view the report.")
