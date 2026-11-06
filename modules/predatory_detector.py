"""
modules/predatory_detector.py
Predatory and exploitative clause detection. Local Ollama only.
"""
from modules.rag_pipeline import retrieve_chunks, ask_llm

_SYSTEM = """You are an Indian consumer protection and legal expert.
Analyse financial and legal documents for predatory, exploitative, or fraudulent clauses that harm the weaker party.
Do not invent clauses, statutes, or case names that are not present in the document or the legal protections text.

Respond in EXACTLY this format:
VERDICT: [SAFE/SUSPICIOUS/LIKELY PREDATORY]
RISK LEVEL: [LOW/MEDIUM/HIGH]
RED FLAGS:
- [List each red flag found, or write 'None detected']
EXPLOITATIVE CLAUSES:
- [Quote specific problematic clauses with explanation]
LEGAL VIOLATIONS:
- [Specific Indian law sections potentially violated, or 'None identified']
RECOMMENDATION:
[Clear actionable advice for the affected party]

This is general information, not legal advice."""


def detect_predatory_clauses(domain, doc_chunks, doc_index,
                             kb_chunks, kb_index, groq_api_key=None):
    """Detect predatory and fraudulent clauses. groq_api_key kept for backwards compatibility, ignored."""
    query = ("predatory unfair exploitative clause hidden trap "
             "misleading deceptive borrower consumer rights")
    doc_context = retrieve_chunks(query, doc_chunks, doc_index, k=5)
    kb_context = retrieve_chunks(query, kb_chunks, kb_index, k=3)

    user_prompt = (
        f"Analyse this {domain.lower()} document for predatory or exploitative clauses:\n\n"
        f"DOCUMENT CONTENT:\n{doc_context}\n\n"
        f"RELEVANT LEGAL PROTECTIONS:\n{kb_context}\n\n"
        "Identify any clauses designed to exploit or harm the weaker party."
    )
    return ask_llm(_SYSTEM, user_prompt, max_tokens=520, temperature=0.2, context_window=2048)