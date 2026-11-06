"""
modules/plain_language.py
Plain-language explanation of a whole document. Local Ollama only.
"""
from modules.rag_pipeline import retrieve_chunks, ask_llm

_SYSTEM = """You are explaining a legal document to an ordinary Indian citizen with no legal background.
Use simple everyday language. Avoid legal jargon. Write as if explaining to a friend or family member.
Do not invent clauses, sections, or legal provisions that are not present in the document context.

Structure your response as:
WHAT IS THIS DOCUMENT?
[2-3 sentences in very simple language]

WHAT ARE YOU AGREEING TO?
[Bullet points of key obligations in simple language]

WHAT ARE YOUR RIGHTS?
[Bullet points of protections you have]

WHAT SHOULD YOU WATCH OUT FOR?
[3-5 red flags in simple language]

WHAT TO DO BEFORE SIGNING?
[3-4 practical steps anyone can take]

This is general information, not legal advice."""


def generate_plain_language(domain, doc_chunks, doc_index, groq_api_key=None):
    """groq_api_key is accepted for backwards compatibility and ignored."""
    doc_context = retrieve_chunks(
        "main purpose terms obligations rights parties", doc_chunks, doc_index, k=6
    )
    user_prompt = (
        f"Explain this {domain.lower()} document in very simple plain language "
        f"that any ordinary person would understand:\n\n{doc_context}\n\n"
        "Make it clear, simple, and practical."
    )
    return ask_llm(_SYSTEM, user_prompt, max_tokens=520, temperature=0.2,
                   context_window=2048)