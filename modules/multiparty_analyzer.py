"""
modules/multiparty_analyzer.py
Analyses the document from each affected party's perspective. Local Ollama only.
"""
from modules.rag_pipeline import retrieve_chunks, ask_llm

PARTY_CONFIGS = {
    "FINANCIAL": {
        "parties": ["Lender/Bank", "Borrower", "Guarantor (if any)"],
        "context": "loan financial agreement",
    },
    "PROPERTY": {
        "parties": ["Seller/Vendor", "Buyer/Purchaser", "Developer (if applicable)"],
        "context": "property sale deed agreement",
    },
    "FAMILY": {
        "parties": ["Petitioner/Party A", "Respondent/Party B",
                    "Child's Best Interest"],
        "context": "family law divorce custody agreement",
    },
    "CONSUMER": {
        "parties": ["Business/Service Provider", "Consumer",
                    "Third Parties (if any)"],
        "context": "consumer service agreement",
    },
}

_SYSTEM_TEMPLATE = """You are analysing a {context} from the perspective of the {party}.
Be specific about what benefits and risks this party faces.
Be honest. If the document heavily favours one party, say so clearly.
Do not invent clauses.

Respond in EXACTLY this format:
PARTY: {party}
FAVOURABLE CLAUSES:
- [List clauses that benefit this party, or 'None identified']
UNFAVOURABLE CLAUSES:
- [List clauses that harm or burden this party, or 'None identified']
HIDDEN RISKS:
- [List non-obvious risks this party faces, or 'None detected']
POWER BALANCE: [BALANCED/SLIGHTLY UNFAIR/HEAVILY UNFAIR to this party]
ADVICE: [One clear recommendation for this party]

This is general information, not legal advice."""


def analyze_multiparty(domain, doc_chunks, doc_index, groq_api_key=None):
    """groq_api_key is accepted for backwards compatibility and ignored."""
    config = PARTY_CONFIGS.get(domain, PARTY_CONFIGS["FINANCIAL"])
    doc_context = retrieve_chunks(
        "rights obligations benefits risks parties", doc_chunks, doc_index, k=6
    )

    results = {}
    for party in config["parties"]:
        system = _SYSTEM_TEMPLATE.format(context=config["context"], party=party)
        user_prompt = (
            f"Analyse this document from the perspective of: {party}\n\n"
            f"DOCUMENT CONTENT:\n{doc_context}\n\n"
            f"What are the benefits, risks, and concerns for {party}?"
        )
        results[party] = ask_llm(
            system, user_prompt,
            max_tokens=340, temperature=0.1, context_window=2048,
        )
    return results