# Domain classification using keyword + semantic matching
# No ML training needed — rule-based for demo
# Final project: replace with fine-tuned DistilBERT

DOMAIN_KEYWORDS = {
    "FINANCIAL": [
        "loan", "borrower", "lender", "interest rate", "emi",
        "principal", "collateral", "mortgage", "credit", "nbfc",
        "repayment", "default", "apr", "processing fee", "sanction",
        "disbursal", "pledge", "hypothecation", "guarantee", "security",
        "rbi", "sebi", "lodr", "dividend", "securities", "shares",
        "listed company", "promoter", "audit committee", "disclosure",
        "mutual fund", "nav", "nfo", "unit holder", "idcw",
        "expense ratio", "exit load", "nifty", "asset management",
        "scheme information", "amc", "trustee company"
    ],
    "PROPERTY": [
        "sale deed", "property", "land", "plot", "flat", "apartment",
        "buyer", "seller", "vendor", "purchaser", "immovable",
        "registration", "stamp duty", "title", "encumbrance",
        "possession", "conveyance", "lease", "rent", "tenant",
        "landlord", "noc", "no objection", "transfer", "survey number",
        "khasra", "khata", "survey", "municipal", "patta"
    ],
    "FAMILY": [
        "divorce", "matrimonial", "husband", "wife", "spouse",
        "custody", "child", "minor", "guardian", "maintenance",
        "alimony", "stridhan", "dowry", "marriage", "separation",
        "mutual consent", "petition", "respondent", "petitioner",
        "visitation", "access", "welfare", "cruelty", "desertion",
        "family court", "district court", "matrimonial home"
    ],
    "CONSUMER": [
        "consumer", "service", "product", "warranty", "guarantee",
        "defect", "deficiency", "complaint", "redressal", "forum",
        "unfair trade", "misleading", "advertisement", "return policy",
        "refund", "exchange", "subscription", "terms of service",
        "privacy policy", "ecommerce", "online", "app", "platform",
        "seller", "marketplace", "delivery", "complaint number"
    ]
}

FINANCIAL_SUBTYPES = {
    "MUTUAL_FUND": [
        "mutual fund", "nav", "nfo", "amc", "sebi mf",
        "scheme", "unit holder", "idcw", "redemption proceeds",
        "asset management", "trustee", "benchmark index",
        "expense ratio", "exit load", "nifty", "sensex"
    ],
    "LOAN": [
        "borrower", "lender", "emi", "interest rate", "loan amount",
        "repayment", "default", "nbfc", "collateral", "disbursal",
        "processing fee", "prepayment", "loan agreement"
    ],
    "SECURITIES": [
        "lodr", "listed company", "promoter", "pledge",
        "board of directors", "audit committee", "sebi lodr",
        "related party", "material event", "stock exchange"
    ]
}

def get_financial_subtype(text):
    text_lower = text.lower()
    subtype_scores = {
        st: sum(1 for kw in kws if kw in text_lower)
        for st, kws in FINANCIAL_SUBTYPES.items()
    }
    return max(subtype_scores, key=subtype_scores.get)

DOMAIN_INFO = {
    "FINANCIAL": {
        "icon": "💰",
        "description": "Loan Agreement / Financial Document / Securities",
        "color": "#ff9800"
    },
    "PROPERTY": {
        "icon": "🏠",
        "description": "Property / Land / Real Estate Document",
        "color": "#4caf50"
    },
    "FAMILY": {
        "icon": "👨‍👩‍👧",
        "description": "Family Law / Divorce / Custody Document",
        "color": "#9c27b0"
    },
    "CONSUMER": {
        "icon": "🛍️",
        "description": "Consumer Agreement / Service Contract",
        "color": "#2196f3"
    }
}

def classify_domain(text):
    """
    Classify document domain based on keyword frequency
    Returns: (domain, confidence, scores)
    """
    text_lower = text.lower()
    scores = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        scores[domain] = count

    total = sum(scores.values())
    if total == 0:
        return "FINANCIAL", 0.25, scores

    # Get top domain
    top_domain = max(scores, key=scores.get)
    confidence = scores[top_domain] / total if total > 0 else 0.25

    # Normalize scores to percentages
    score_pcts = {
        d: round((s / total) * 100, 1)
        for d, s in scores.items()
    }

    subtype = get_financial_subtype(text) if top_domain == "FINANCIAL" else None
    return top_domain, confidence, score_pcts, subtype

def domain_key(domain):
    """Lowercase domain string for clause_classifier / rag_pipeline lookups."""
    return (domain or "").strip().lower()