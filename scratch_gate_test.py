import sys
from pathlib import Path

ROOT = Path(r"c:\Users\Ani\OneDrive\Desktop\legal contract Analyzer\rag-legal-contract-analyzer")
sys.path.insert(0, str(ROOT))

from pipeline.document_gate import assess_document

test_text = """
# Create comprehensive list of real Indian prescription medicines # Based on research from 1mg, PharmEasy, Apollo Pharmacy, and medical references

new_drugs = [ # PAIN ANALGESICS & ANTI-INFLAMMATORY { "Medicine Name": "Dolo 650 Tablet", "Prescription": "Prescription Required", "Type of Sell": "15.0 tablets in 1 strip", "Manufacturer": "Micro Labs Ltd", "Salt": "Paracetamol (650mg),", "MRP": "₹34.65", "Status": "", "Uses": "Pain relief,Treatment of Fever,", "Alternate Medicines": "Calpol 650mg Tablet,Febrinil 650mg Tablet,Pyrigesic 650mg Tablet,", "Side Effects": "Stomach pain,Nausea,Vomiting,", "How to Use": "Take this medicine in the dose and duration as advised by your doctor. Swallow it as a whole. Do not chew, crush or break it. Dolo 650 Tablet is to be taken with food.", "Chemical Class": "P-Aminophenol Derivative", "Habit Forming": "No", "Therapeutic Class": "PAIN ANALGESICS", "Action Class": "Analgesic & Antipyretic-PCM", "How It Works": "Dolo 650 Tablet is an analgesic (pain reliever) and antipyretic (fever reducer). It works by blocking the release of certain chemical messengers that cause pain and fever." }, { "Medicine Name": "Combiflam Tablet", "Prescription": "Prescription Required", "Type of Sell": "20.0 tablets in 1 strip", "Manufacturer": "Sanofi India Ltd", "Salt": "Ibuprofen (400mg) + Paracetamol (325mg),", "MRP": "₹45.50", "Status": "", "Uses": "Pain relief,Treatment of Fever,", "Alternate Medicines": "Ibugesic Plus Tablet,Flexon Tablet,Brufen Plus Tablet,", "Side Effects": "Nausea,Vomiting,Stomach pain,Heartburn,Diarrhea,", "How to Use": "Take this medicine in the dose and duration as advised by your doctor. Swallow it as a whole. Do not chew, crush or break it. Combiflam Tablet is to be taken with food.", "Chemical Class": "Propionic acid derivative", "Habit Forming": "No", "Therapeutic Class": "PAIN ANALGESICS", "Action Class": "NSAID's- Non-Selective COX 1&2 Inhibitors", "How It Works": "Combiflam Tablet is a combination of two medicines: Ibuprofen and Paracetamol. It works by blocking the release of certain chemical messengers that cause fever, pain and inflammation." }, { "Medicine Nam
"""

is_contract, msg = assess_document(test_text)
print(f"Is Contract: {is_contract}")
print(f"Message: {msg}")

from pipeline.document_gate import _contract_score, _normalise
norm = _normalise(test_text)
score = _contract_score(norm, test_text)
print(f"Contract score: {score}")

from pipeline.document_gate import _CLAUSE_NUMBER_RE, _SECTION_HEADING_RE, _ARTICLE_CAPS_RE, _NUMBERED_ARTICLE_RE
print(f"Clause number hits: {len(_CLAUSE_NUMBER_RE.findall(test_text))}")
print(f"Section heading hits: {bool(_SECTION_HEADING_RE.search(test_text))}")
print(f"Article caps hits: {bool(_ARTICLE_CAPS_RE.search(test_text))}")
print(f"Numbered article hits: {bool(_NUMBERED_ARTICLE_RE.search(test_text))}")
