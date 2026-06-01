"""
tests/test_document_gate.py
============================
Unit tests for the contract document gate heuristic.
"""
import pytest

from pipeline.document_gate import assess_document, CONTRACT_GATE_MESSAGE
from tests.conftest import SIMPLE_CONTRACT, DEMO_CONTRACT


SAMPLE_RESUME = """
John Doe
linkedin.com/in/johndoe

Experience
Software Engineer at PanTerra Networks
Jan 2020 - Present
Led platform integrations and API design.

Education
Bachelor of Science in Computer Science
Master of Science, Information Systems

Skills
Python, Java, React, Leadership, Communication

Summary
Motivated engineer with internship experience at a growth-stage startup.

Honors
Dean's List

Publications
None listed.

Page 1 of 3
"""

# Snippets from user report (Aniketh CV) — must not pass as contract
ANIKETH_CV_SNIPPET = """
SUMMARY Visual/UI Designer with strong focus on user-centered design and prototyping.
EXPERIENCE Associate - UI/UX Designer PanTerra Networks November 2025 - Present
EDUCATION Executive M.Tech Computer Science 2024 - 2025
SKILLS Design Tools: Figma Adobe XD Sketch
CERTIFICATIONS Create High-Fidelity Designs and Prototypes
PROJECT RAG-Based Legal Contract Analyzer full-stack contract review tool
aniketh@example.com +91 9876543210 linkedin.com/in/aniketh github.com/anikethv
May 2025 - November 2025 UI Intern at Design Studio
"""


class TestAssessDocument:
    def test_demo_contract_passes(self):
        ok, msg = assess_document(DEMO_CONTRACT)
        assert ok is True
        assert msg == ""

    def test_simple_contract_passes(self):
        ok, msg = assess_document(SIMPLE_CONTRACT)
        assert ok is True
        assert msg == ""

    def test_resume_rejects(self):
        ok, msg = assess_document(SAMPLE_RESUME)
        assert ok is False
        assert msg == CONTRACT_GATE_MESSAGE

    def test_aniketh_cv_snippet_rejects(self):
        ok, msg = assess_document(ANIKETH_CV_SNIPPET)
        assert ok is False
        assert msg == CONTRACT_GATE_MESSAGE

    def test_aniketh_cv_snippet_rejects_with_cv_filename(self):
        ok, msg = assess_document(
            ANIKETH_CV_SNIPPET,
            filename="Aniketh_Vustepalle_CV.pdf",
        )
        assert ok is False
        assert msg == CONTRACT_GATE_MESSAGE

    def test_project_contract_name_does_not_pass_without_legal_phrasing(self):
        """Portfolio project title must not satisfy the gate alone."""
        text = (
            "SUMMARY Designer\n"
            "EXPERIENCE Lead at Acme 2023 - 2024\n"
            "EDUCATION B.A. Design\n"
            "SKILLS Figma\n"
            "PROJECT Legal Contract Analyzer and agreement tooling for demos\n"
            "aniketh@mail.com linkedin.com/in/user\n"
        )
        ok, msg = assess_document(text)
        assert ok is False

    def test_empty_rejects(self):
        ok, msg = assess_document("   \n\n  ")
        assert ok is False
        assert msg == CONTRACT_GATE_MESSAGE

    def test_whitespace_only_rejects(self):
        ok, msg = assess_document("\t\t")
        assert ok is False

    def test_gate_message_mentions_resumes(self):
        assert "resumes" in CONTRACT_GATE_MESSAGE.lower()
        assert "linkedin" in CONTRACT_GATE_MESSAGE.lower()

    def test_filename_cv_boost_rejects_borderline(self):
        short_cv = (
            "SUMMARY Designer with visual systems experience.\n"
            "EXPERIENCE Company 2024 - 2025\n"
            "EDUCATION Degree program completed with honors.\n"
            "SKILLS Figma prototyping and user research methods applied daily.\n"
            "PROJECT Legal Contract Analyzer\n"
        )
        ok, msg = assess_document(short_cv, filename="profile_cv.pdf")
        assert ok is False
        assert msg == CONTRACT_GATE_MESSAGE
