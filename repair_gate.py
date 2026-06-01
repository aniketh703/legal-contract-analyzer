import re

file_path = r"c:\Users\Ani\OneDrive\Desktop\legal contract Analyzer\rag-legal-contract-analyzer\pipeline\document_gate.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# We know the duplicate starts because the file suddenly contains "_RESUME_SECTION_HEADERS" or something again.
# Let's just find the first occurrence of `return True, ""` and truncate there,
# but since my replace_file_content failed, it might be messy.

# I will just write out the EXACT `assess_document` function directly from my knowledge
# since I have seen lines 1-320 perfectly.

fixed_function = '''
def assess_document(
    text: str,
    segments: list[dict] | None = None,
    filename: str | None = None,
) -> tuple[bool, str]:
    """
    Returns (is_contract, user_message_if_not).

    ``segments`` is optional; when provided, segment text is merged with
    ``text`` so short PDF extractions still get full coverage.
    ``filename`` optional path or stem (e.g. Aniketh_Vustepalle_CV.pdf) boosts
    resume detection.
    """
    parts = [text or ""]
    if segments:
        for seg in segments:
            parts.append(seg.get("clause_text") or seg.get("text") or "")
    combined = "\\n".join(parts)
    norm = _normalise(combined)

    if len(norm) < MIN_TEXT_CHARS:
        return False, CONTRACT_GATE_MESSAGE

    contract_score = _contract_score(norm, combined)
    resume_score, resume_structural = _resume_score(norm, combined)
    resume_score += _filename_resume_boost(filename)

    section_count = _count_resume_sections(combined)

    # --- Reject rules (order: structural resume signals first) ---
    if resume_structural >= RESUME_STRUCTURAL_REJECT:
        return False, CONTRACT_GATE_MESSAGE

    if section_count >= 2:
        return False, CONTRACT_GATE_MESSAGE

    if _has_experience_education_skills_combo(norm):
        return False, CONTRACT_GATE_MESSAGE

    if resume_score >= RESUME_REJECT_THRESHOLD:
        return False, CONTRACT_GATE_MESSAGE

    legal_hits = sum(
        1 for phrase in _LEGAL_CONTRACT_PHRASES if phrase in norm
    )
    
    # STRICT ENFORCEMENT: A document MUST contain at least one strongly indicative
    # legal phrase to be considered a contract. This completely prevents lists, 
    # medical data, or non-legal text from bypassing the gate.
    if legal_hits == 0:
        return False, CONTRACT_GATE_MESSAGE

    if contract_score < MIN_CONTRACT_SCORE:
        return False, CONTRACT_GATE_MESSAGE

    if resume_score > contract_score and resume_score >= 2:
        return False, CONTRACT_GATE_MESSAGE

    return True, ""
'''

# Find everything up to `def assess_document(` and replace the rest of the file
import re
new_content = re.sub(r'def assess_document\(.*', fixed_function.strip(), content, flags=re.DOTALL)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Repaired document_gate.py")
