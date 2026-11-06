"""
modules/cibil_analyzer.py
CIBIL score impact analysis. Local Ollama only.
"""
from modules.rag_pipeline import retrieve_chunks, ask_llm

_SYSTEM = """You are an Indian credit and banking expert specialising in CIBIL score impact analysis.
Analyse the loan clauses for their likely effect on the borrower's credit score.
Do not invent clauses or statutory sections. If the document is silent on a factor, say so.

Respond in EXACTLY this format:
CIBIL RISK LEVEL: [LOW/MEDIUM/HIGH]
OVERALL ASSESSMENT: [One paragraph summary]
IMPACTING CLAUSES:
1. [Clause type]: [Quote or describe] - [How it affects CIBIL score]
[Continue for each clause found, or write 'None identified']
AUTO-DEFAULT TRIGGERS: [YES/NO - explain if yes]
CROSS-DEFAULT PROVISIONS: [YES/NO - explain if yes]
PENALTY CONVERSION RISK: [YES/NO - explain if yes]
BORROWER ADVICE:
[Plain language advice on protecting credit score]

This is general information, not legal advice."""

def analyze_cibil_impact(doc_chunks, doc_index, kb_chunks, kb_index,
                         domain=None, groq_api_key=None):
    if domain and domain.upper() != "FINANCIAL":
        return ("CIBIL analysis is only meaningful for financial loan documents. "
                "This document is classified as non-financial, so no CIBIL impact can be assessed.")
    """groq_api_key is accepted for backwards compatibility and ignored."""
    query = ("credit score CIBIL default penalty reporting bureau "
             "cross default trigger automatic")
    doc_context = retrieve_chunks(query, doc_chunks, doc_index, k=5)
    kb_context = retrieve_chunks(query, kb_chunks, kb_index, k=3)

    user_prompt = (
        "Analyse this loan/financial agreement for clauses that could damage the "
        "borrower's CIBIL credit score:\n\n"
        f"DOCUMENT CONTENT:\n{doc_context}\n\n"
        f"CIBIL REPORTING RULES:\n{kb_context}\n\n"
        "Identify all clauses with potential adverse CIBIL impact."
    )
    return ask_llm(_SYSTEM, user_prompt, max_tokens=420, temperature=0.1,
                   context_window=2048)