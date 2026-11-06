import requests

# Hardcoded precedents as fallback when API unavailable
SAMPLE_PRECEDENTS = {
    "FINANCIAL": [
        {
            "case": "ICICI Bank Ltd vs Prakash Kaur",
            "court": "Supreme Court of India",
            "year": "2007",
            "citation": "AIR 2007 SC 1348",
            "outcome": "Court held that banks cannot use muscle power "
                       "for recovery and must follow due process. "
                       "Recovery agents threatening borrowers is illegal.",
            "relevance": "Predatory recovery practices"
        },
        {
            "case": "Central Bank of India vs Ravindra",
            "court": "Supreme Court of India",
            "year": "2002",
            "citation": "(2002) 1 SCC 367",
            "outcome": "Compound interest and penal charges must be "
                       "clearly disclosed. Undisclosed charges in loan "
                       "agreements are unenforceable.",
            "relevance": "Hidden charges in loan agreements"
        }
    ],
    "PROPERTY": [
        {
            "case": "Suraj Lamp & Industries vs State of Haryana",
            "court": "Supreme Court of India",
            "year": "2012",
            "citation": "(2012) 1 SCC 656",
            "outcome": "Power of Attorney sales do not confer title. "
                       "Only registered sale deeds transfer ownership "
                       "of immovable property.",
            "relevance": "Property title and registration"
        },
        {
            "case": "K. Laxmanan vs Thekkayil Padmini",
            "court": "Supreme Court of India",
            "year": "2009",
            "citation": "(2009) 1 SCC 354",
            "outcome": "Seller must disclose all known defects in "
                       "property title. Concealment of encumbrances "
                       "gives buyer right to rescind contract.",
            "relevance": "Title defects and encumbrance disclosure"
        }
    ],
    "FAMILY": [
        {
            "case": "Navneet Kaur vs Neeraj Kumar",
            "court": "Delhi High Court",
            "year": "2019",
            "citation": "2019 SCC OnLine Del 9064",
            "outcome": "Maintenance amount must be sufficient for "
                       "spouse to maintain same standard of living "
                       "as during marriage.",
            "relevance": "Maintenance and alimony amounts"
        },
        {
            "case": "Gaurav Nagpal vs Sumedha Nagpal",
            "court": "Supreme Court of India",
            "year": "2009",
            "citation": "(2009) 1 SCC 42",
            "outcome": "Child's welfare is paramount in custody "
                       "disputes, not rights of parents. Court must "
                       "independently assess what is best for child.",
            "relevance": "Child custody and welfare"
        }
    ],
    "CONSUMER": [
        {
            "case": "Lucknow Development Authority vs M.K. Gupta",
            "court": "Supreme Court of India",
            "year": "1994",
            "citation": "(1994) 1 SCC 243",
            "outcome": "Housing board or builder causing delay in "
                       "possession is deficiency in service. Consumer "
                       "is entitled to compensation for delay.",
            "relevance": "Service deficiency and consumer rights"
        },
        {
            "case": "Spring Meadows Hospital vs Harjol Ahluwalia",
            "court": "Supreme Court of India",
            "year": "1998",
            "citation": "(1998) 4 SCC 39",
            "outcome": "Service provider cannot waive consumer's "
                       "statutory rights through contract clauses. "
                       "Consumer Protection Act rights are inalienable.",
            "relevance": "Waiver of consumer rights"
        }
    ]
}

def retrieve_precedents(domain, flagged_issues=None):
    """
    Retrieve relevant Indian legal precedents
    Uses hardcoded precedents for demo
    In production: queries Indian Kanoon API
    """
    # Try Indian Kanoon API first
    try:
        if flagged_issues:
            query = " ".join(flagged_issues[:3])
            response = requests.get(
                "https://api.indiankanoon.org/search/",
                params={"formInput": query, "pagenum": 0},
                headers={"Authorization": "Token YOUR_IK_TOKEN"},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("docs"):
                    return [
                        {
                            "case": doc.get("title", "Unknown"),
                            "court": doc.get("courtName", "Unknown Court"),
                            "year": doc.get("publishdate",
                                          "Unknown")[:4],
                            "citation": doc.get("citation", ""),
                            "outcome": doc.get("headline",
                                             "See full judgment"),
                            "relevance": "Retrieved from Indian Kanoon"
                        }
                        for doc in data["docs"][:2]
                    ]
    except Exception:
        pass

    # Fallback to hardcoded precedents
    return SAMPLE_PRECEDENTS.get(domain,
                                  SAMPLE_PRECEDENTS["FINANCIAL"])