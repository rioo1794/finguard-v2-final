"""
modules/compliance_checker.py
Runs per-domain compliance checks against retrieved Indian law. Local Ollama only.
"""
import re
from modules.rag_pipeline import retrieve_chunks, ask_llm

FINANCIAL_CHECKS = [
    ("APR Disclosure", "interest rate expressed as annual percentage rate APR"),
    ("Hidden Charges", "hidden charges fees not disclosed upfront"),
    ("Cooling-off Period", "cooling off period withdrawal right"),
    ("Grievance Redressal", "grievance redressal mechanism complaint"),
    ("Auto-renewal Consent", "automatic renewal without explicit consent"),
    ("Prepayment Penalty", "prepayment foreclosure penalty charges"),
    ("Data Collection", "data collection beyond stated purpose"),
    ("Recovery Practices", "recovery agent harassment coercive methods"),
]

PROPERTY_CHECKS = [
    ("Title Clarity", "clear title ownership seller authority to sell"),
    ("Encumbrance", "encumbrance mortgage charge existing loan on property"),
    ("NOC Requirements", "no objection certificate NOC society bank"),
    ("Registration", "registration stamp duty compulsory registration"),
    ("Possession Date", "possession handover delivery date timeline"),
    ("Force Majeure", "force majeure delay extension unreasonable"),
    ("Penalty Clauses", "penalty cancellation forfeiture excessive"),
    ("SEBI LODR", "board authorization disclosure related party pledge"),
]

FAMILY_CHECKS = [
    ("Maintenance Amount", "maintenance alimony amount reasonable fair"),
    ("Child Welfare", "child welfare best interest custody visitation"),
    ("Asset Division", "matrimonial property division fair equitable"),
    ("Stridhan Protection", "stridhan wife property exclusive right"),
    ("Visitation Rights", "visitation access non-custodial parent rights"),
    ("Modification Clause", "modification review change future circumstances"),
]

CONSUMER_CHECKS = [
    ("Unfair Terms", "unfair contract terms significant imbalance"),
    ("Hidden Charges", "hidden charges fees not disclosed"),
    ("Subscription Trap", "automatic renewal subscription cancellation"),
    ("Arbitration Clause", "arbitration clause consumer forum waiver"),
    ("Warranty Terms", "warranty guarantee defect replacement"),
    ("Data Privacy", "data sharing third party consent privacy"),
]

DOMAIN_CHECKS = {
    "FINANCIAL": FINANCIAL_CHECKS,
    "PROPERTY": PROPERTY_CHECKS,
    "FAMILY": FAMILY_CHECKS,
    "CONSUMER": CONSUMER_CHECKS,
}

_SYSTEM = """You are an Indian legal compliance expert specialising in {domain} law.
Analyse the document excerpt against the relevant Indian legal requirement.

Respond in EXACTLY this format:
STATUS: [PASS/FAIL/UNCLEAR]
FINDING: [One sentence explanation with specific reference]
CLAUSE: [Quote the problematic text if found, else write 'None identified']
RECOMMENDATION: [One actionable suggestion]
If the DOCUMENT EXCERPT does not contain text relevant to the check, return UNCLEAR, not FAIL.
Only return FAIL when you can quote a specific clause that violates the requirement.
Absence of a topic in a document is not evidence of non-compliance unless the topic is mandatory for that document type.
Do not invent clauses or provisions. If the excerpt is silent on the point, use UNCLEAR.
This is general information, not legal advice."""

_STATUS_PAT = re.compile(r"STATUS\s*:\s*\**\s*(PASS|FAIL|UNCLEAR)", re.IGNORECASE)


def _parse_status(text):
    m = _STATUS_PAT.search(text or "")
    return m.group(1).upper() if m else "UNCLEAR"


def run_compliance_check(domain, doc_chunks, doc_index, kb_chunks, kb_index,
                         groq_api_key=None):
    """groq_api_key is accepted for backwards compatibility and ignored."""
    checks = DOMAIN_CHECKS.get(domain, FINANCIAL_CHECKS)
    system = _SYSTEM.format(domain=domain.lower())
    results = []

    for check_name, check_query in checks:
        doc_context = retrieve_chunks(check_query, doc_chunks, doc_index)
        kb_context = retrieve_chunks(check_query, kb_chunks, kb_index)
        user_prompt = (
            f"Check this document for: {check_name}\n\n"
            f"DOCUMENT EXCERPT:\n{doc_context}\n\n"
            f"RELEVANT LEGAL REQUIREMENT:\n{kb_context}\n\n"
            f"Is this document compliant with Indian law on {check_name}?"
        )
        result = ask_llm(system, user_prompt, max_tokens=180, temperature=0.1,
                         context_window=1536)
        # If the model says it couldn't identify a clause, treat as UNCLEAR, not FAIL.
        if "CLAUSE: None identified" in result or "CLAUSE: None" in result:
            status = "UNCLEAR"
        else:
            status = _parse_status(result)
        results.append({
            "check": check_name,
            "result": result,
            "status": _parse_status(result),
        })
    return results