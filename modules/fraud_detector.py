"""
modules/fraud_detector.py
Fraud pattern and structural anomaly detection. Local Ollama only.
"""
from modules.rag_pipeline import retrieve_chunks, ask_llm

MANDATORY_SECTIONS = {
    "FINANCIAL": [
        "interest rate", "repayment schedule", "processing fee",
        "default clause", "grievance redressal", "prepayment",
        "borrower details", "lender details", "loan amount",
    ],
    "PROPERTY": [
        "property description", "sale consideration", "possession date",
        "title warranty", "stamp duty", "registration", "schedule",
        "seller details", "buyer details", "witnesses",
    ],
    "FAMILY": [
        "parties", "marriage date", "grounds", "relief sought",
        "children", "assets", "maintenance", "court", "date",
    ],
    "CONSUMER": [
        "parties", "service description", "consideration",
        "duration", "termination", "grievance", "governing law",
    ],
}

_SYSTEM = """You are a document forensics and legal authenticity expert
specialising in Indian legal documents. Analyse documents for fraud indicators,
structural anomalies, and missing mandatory elements.
Do not invent facts. If the document is normal, say so.
If the document is an informational guide rather than a contract, do not flag missing parties, consideration, or signatures as concerns.
Respond in EXACTLY this format:
AUTHENTICITY: [LIKELY AUTHENTIC/SUSPICIOUS/LIKELY FRAUDULENT]
RISK SCORE: [LOW/MEDIUM/HIGH]
STRUCTURAL ISSUES:
- [List any missing or anomalous structural elements, or 'None detected']
AUTHORIZATION CONCERNS:
- [Issues with authorization, signatures, witnesses, or 'None detected']
DATE INCONSISTENCIES:
- [Any date anomalies found, or 'None detected']
CROSS-ENTITY CONCERNS:
- [Flag if assets of one entity used to benefit another without authorization, or 'None detected']
RECOMMENDATION:
[What the affected party should verify or do]

This is general information, not legal advice."""


def detect_fraud_patterns(domain, text, doc_chunks, doc_index, groq_api_key=None):
    """groq_api_key is accepted for backwards compatibility and ignored.
    Returns (llm_result_text, list_of_missing_mandatory_sections)."""
    text_lower = (text or "").lower()
    mandatory = MANDATORY_SECTIONS.get(domain, [])
    missing = [s for s in mandatory if s not in text_lower]

    doc_context = retrieve_chunks(
        "authorization signatures parties date stamp", doc_chunks, doc_index, k=4
    )

    user_prompt = (
        f"Analyse this {domain.lower()} document for fraud indicators and structural anomalies:\n\n"
        f"DOCUMENT EXCERPT:\n{doc_context}\n\n"
        f"MISSING MANDATORY SECTIONS DETECTED: {missing if missing else 'None'}\n\n"
        "Flag any suspicious patterns, missing authorizations, or anomalies."
    )
    result = ask_llm(_SYSTEM, user_prompt, max_tokens=340, temperature=0.1,
                     context_window=2048)
    return result, missing