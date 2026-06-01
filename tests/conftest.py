"""
tests/conftest.py
=================
Shared fixtures for all test modules.
"""
import sys
from pathlib import Path

# Ensure project root is on path for all tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

# ---------------------------------------------------------------------------
# Sample contracts
# ---------------------------------------------------------------------------

SIMPLE_CONTRACT = """
1. TERM AND TERMINATION
1.1 This Agreement commences on the Effective Date and continues for twelve (12) months.
1.2 Either party may terminate this Agreement upon thirty (30) days written notice in the event of a material breach that remains uncured.

2. CONFIDENTIALITY
2.1 Each party shall keep confidential all proprietary information of the other party and shall not disclose such information to any third party without prior written consent.

3. INDEMNIFICATION
3.1 The Service Provider shall indemnify and hold harmless the Client from any claims, damages, losses or expenses arising from breach of this Agreement or negligence.

4. NON-COMPETE
4.1 The employee shall not engage in any competing business activities for a period of two (2) years after termination of this Agreement.

5. FORCE MAJEURE
5.1 Neither party shall be liable for delays caused by events beyond their reasonable control including acts of God, war, or natural disasters.

6. ARBITRATION
6.1 All disputes shall be resolved by arbitration under the Arbitration and Conciliation Act, 1996, seated at New Delhi.

7. GOVERNING LAW
7.1 This Agreement shall be governed by and construed in accordance with the laws of India.
"""

MINIMAL_CONTRACT = """
1.1 Either party may terminate this Agreement upon thirty (30) days written notice.
2.1 All disputes shall be resolved by arbitration under the Arbitration Act, 1996.
"""

UNPARSEABLE_TEXT = "   \n\n\n   "

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


@pytest.fixture
def simple_contract():
    return SIMPLE_CONTRACT


@pytest.fixture
def minimal_contract():
    return MINIMAL_CONTRACT


@pytest.fixture
def unparseable_text():
    return UNPARSEABLE_TEXT
