import re
from pathlib import Path

ROOT = Path(r"c:\Users\Ani\OneDrive\Desktop\legal contract Analyzer\rag-legal-contract-analyzer")
target_file = ROOT / "data" / "create_indian_clauses.py"

content = target_file.read_text(encoding="utf-8")

NEW_CLAUSES = """    # -------------------------------------------------------------------------
    # GST (15 examples) — risk_level: low
    # -------------------------------------------------------------------------
    ("All payments made under this Agreement are exclusive of Goods and Services Tax (GST) unless otherwise stated.", "GST", "low"),
    ("The Client shall pay the GST at the prevailing rate upon receipt of a valid tax invoice.", "GST", "low"),
    ("Any GST or similar tax payable on the services shall be borne by the Purchaser.", "GST", "low"),
    ("The Service Provider shall issue a GST-compliant invoice for all fees and charges.", "GST", "low"),
    ("If any GST is levied on the transaction, it shall be in addition to the stated consideration.", "GST", "low"),
    ("The Vendor warrants that it is registered under the GST Act and will remit all collected taxes.", "GST", "low"),
    ("In the event of a change in GST rates, the prices shall be adjusted accordingly.", "GST", "low"),
    ("Both Parties shall cooperate to claim input tax credits under the GST regime.", "GST", "low"),
    ("The Contractor shall indemnify the Employer against any loss of input tax credit due to the Contractor's non-compliance.", "GST", "low"),
    ("Taxes, including GST, shall be shown separately on all invoices.", "GST", "low"),
    ("The Buyer shall reimburse the Seller for any GST liability incurred in relation to this Agreement.", "GST", "low"),
    ("Where GST is applicable, the amount payable shall be increased by the GST amount.", "GST", "low"),
    ("The Supplier must provide a valid GST invoice within 30 days of supply.", "GST", "low"),
    ("Any penalty for late payment of GST by the Provider shall be borne by the Provider.", "GST", "low"),
    ("All prices are quoted exclusive of GST and other applicable statutory levies.", "GST", "low"),

    # -------------------------------------------------------------------------
    # StampDuty (15 examples) — risk_level: low
    # -------------------------------------------------------------------------
    ("The stamp duty payable on this Agreement shall be borne equally by both Parties.", "StampDuty", "low"),
    ("The Lessee shall bear the cost of stamp duty and registration charges for this Lease.", "StampDuty", "low"),
    ("All stamp duty, registration fees, and similar charges in relation to this deed shall be paid by the Purchaser.", "StampDuty", "low"),
    ("The Borrower is responsible for paying all stamp duties in connection with the loan documents.", "StampDuty", "low"),
    ("Any penalty for inadequate stamp duty on this Agreement shall be the responsibility of the First Party.", "StampDuty", "low"),
    ("This Agreement shall be stamped in accordance with the Indian Stamp Act, 1899.", "StampDuty", "low"),
    ("The costs of stamping and registering this instrument shall be to the account of the Buyer.", "StampDuty", "low"),
    ("Each Party shall bear its own legal costs, but stamp duty shall be shared 50:50.", "StampDuty", "low"),
    ("The Promotor shall pay the applicable stamp duty on the allotment letter.", "StampDuty", "low"),
    ("If this document is deemed inadequately stamped, the defaulting party shall pay the deficit and penalty.", "StampDuty", "low"),
    ("The stamp duty for this Leave and License agreement will be paid by the Licensee.", "StampDuty", "low"),
    ("All duties, including stamp duty and registration fees, are payable by the Developer.", "StampDuty", "low"),
    ("The cost of stamp paper and franking shall be borne by the Client.", "StampDuty", "low"),
    ("The Parties agree that the stamp duty on this supplementary agreement shall be paid by the Vendor.", "StampDuty", "low"),
    ("In case of adjudication of stamp duty, the fees shall be paid by the Purchaser.", "StampDuty", "low"),

    # -------------------------------------------------------------------------
    # SpecificPerformance (15 examples) — risk_level: high
    # -------------------------------------------------------------------------
    ("The Parties agree that damages would not be an adequate remedy and the non-breaching Party shall be entitled to specific performance.", "SpecificPerformance", "high"),
    ("In the event of a breach, the aggrieved Party may seek an order for specific performance of the obligations under this Agreement.", "SpecificPerformance", "high"),
    ("The Purchaser shall have the right to enforce specific performance of the Seller's obligation to transfer the shares.", "SpecificPerformance", "high"),
    ("Nothing in this clause shall prejudice the right of the Disclosing Party to seek specific performance of the confidentiality obligations.", "SpecificPerformance", "high"),
    ("The Parties acknowledge that the subject matter of this Agreement is unique and specific performance is the appropriate remedy for breach.", "SpecificPerformance", "high"),
    ("The remedy of specific performance shall be available in addition to any other remedies available at law or in equity.", "SpecificPerformance", "high"),
    ("In case of default by the Developer, the Landowner shall be entitled to seek specific performance of the development obligations.", "SpecificPerformance", "high"),
    ("The Buyer may, at its sole discretion, choose to seek specific performance instead of terminating the Agreement.", "SpecificPerformance", "high"),
    ("The defaulting Party waives the defense that damages are an adequate remedy in any action for specific performance.", "SpecificPerformance", "high"),
    ("The non-defaulting Party shall be entitled to an injunction and/or specific performance to prevent a breach of this Agreement.", "SpecificPerformance", "high"),
    ("The Parties submit that specific performance is the only adequate remedy for a breach of the exclusivity provisions.", "SpecificPerformance", "high"),
    ("Any court of competent jurisdiction may order specific performance of the transfer of intellectual property rights.", "SpecificPerformance", "high"),
    ("Specific performance shall be a cumulative remedy and shall not preclude the recovery of damages.", "SpecificPerformance", "high"),
    ("The Seller acknowledges that the Property is unique and the Buyer shall be entitled to specific performance.", "SpecificPerformance", "high"),
    ("The obligations of the Service Provider are such that specific performance may be enforced by the Client.", "SpecificPerformance", "high"),

    # -------------------------------------------------------------------------
    # DPDP (15 examples) — risk_level: high
    # -------------------------------------------------------------------------
    ("The Service Provider shall process personal data strictly in compliance with the Digital Personal Data Protection Act, 2023.", "DPDP", "high"),
    ("The Data Fiduciary and Data Processor agree to adhere to all obligations under the DPDP Act.", "DPDP", "high"),
    ("Any breach of personal data shall be reported in accordance with the Digital Personal Data Protection Act.", "DPDP", "high"),
    ("The Processor shall implement reasonable security safeguards to prevent personal data breach as required by the DPDP Act.", "DPDP", "high"),
    ("The Parties shall ensure that verifiable consent is obtained from the Data Principal prior to processing personal data.", "DPDP", "high"),
    ("The Vendor acts as a Data Processor and shall only process personal data on the documented instructions of the Data Fiduciary.", "DPDP", "high"),
    ("In the event of a data breach, the Processor shall notify the Fiduciary within 24 hours as per DPDP guidelines.", "DPDP", "high"),
    ("The Client warrants that it has collected the personal data lawfully and obtained the necessary consents under the DPDP Act.", "DPDP", "high"),
    ("The Data Processor shall assist the Data Fiduciary in fulfilling obligations regarding Data Principals' rights.", "DPDP", "high"),
    ("No personal data shall be transferred outside India except in compliance with the provisions of the DPDP Act.", "DPDP", "high"),
    ("The Service Provider shall securely erase all personal data upon termination of the Agreement as required by the DPDP Act.", "DPDP", "high"),
    ("Both Parties agree to cooperate with the Data Protection Board of India in any inquiry or investigation.", "DPDP", "high"),
    ("The Processing of children's data shall be done only after obtaining verifiable parental consent as per the DPDP Act.", "DPDP", "high"),
    ("The Vendor shall indemnify the Client against any penalties imposed by the Data Protection Board due to the Vendor's breach.", "DPDP", "high"),
    ("The Data Processor shall not engage any sub-processor without the prior written authorisation of the Data Fiduciary.", "DPDP", "high"),
]"""

# Replace the closing bracket of the ROWS list
new_content = content.replace("]\n\n# Write to CSV", NEW_CLAUSES + "\n\n# Write to CSV")

target_file.write_text(new_content, encoding="utf-8")
print("Successfully appended new clauses.")
