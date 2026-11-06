import pymupdf
import csv
import re
import os
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────
OUTPUT_DIR = "datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LABEL CATEGORIES ────────────────────────────────────────────
FINANCIAL_LABELS = [
    "HIDDEN_CHARGE",
    "APR_MISREPRESENTATION",
    "PREDATORY_RECOVERY",
    "AUTO_RENEWAL_TRAP",
    "DATA_MISUSE",
    "CIBIL_IMPACT",
    "CROSS_DEFAULT",
    "MISSING_GRIEVANCE",
    "COMPLIANT",
    "NEUTRAL"
]

CONSUMER_LABELS = [
    "UNFAIR_TERM",
    "RIGHTS_WAIVER",
    "HIDDEN_CHARGES",
    "SUBSCRIPTION_TRAP",
    "ARBITRATION_TRAP",
    "COMPLIANT",
    "NEUTRAL"
]

PROPERTY_LABELS = [
    "TITLE_DEFECT",
    "DELAYED_POSSESSION",
    "UNAUTHORIZED_CHARGE",
    "SPECIFIC_PERFORMANCE_ISSUE",
    "UNFAIR_FORFEITURE",
    "REGULATORY_VIOLATION",
    "COMPLIANT",
    "NEUTRAL"
]

FAMILY_LABELS = [
    "CUSTODY_DISPUTE",
    "MAINTENANCE_CLAIM",
    "DIVORCE_GROUND",
    "DOMESTIC_VIOLENCE_INDICATOR",
    "ALIMONY_TERM",
    "PROCEDURAL",
    "NEUTRAL"
]


# ── EXTRACTION FUNCTION ──────────────────────────────────────────
def scan_existing():
    """
    Look at all four domain CSVs and return:
      found  : {source_document name -> csv path} for every document already extracted
      max_id : {csv path -> highest clause number used so far}
    """
    found, max_id = {}, {}
    for prefix in ("FINANCIAL", "CONSUMER", "PROPERTY", "FAMILY"):
        path = os.path.join(OUTPUT_DIR, f"{prefix.lower()}_clauses.csv")
        max_id[path] = 0
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 2:
                    continue
                found[row[1]] = path
                m = re.search(r"_(\d+)$", row[0])
                if m:
                    max_id[path] = max(max_id[path], int(m.group(1)))
    return found, max_id


def extract_clauses_from_pdf(pdf_path, domain, doc_name):
    """
    Extract clauses from a PDF and append them to the domain CSV.
    Safe to re-run: documents already present in ANY domain CSV are skipped,
    so running the script twice no longer duplicates every clause.
    """
    output_file = os.path.join(OUTPUT_DIR, f"{domain.lower()}_clauses.csv")

    # ---- skip documents that were already extracted ----
    existing_docs, max_id = scan_existing()
    if doc_name in existing_docs:
        print(f"\nSkipping: {doc_name}  (already in {os.path.basename(existing_docs[doc_name])})")
        return 0

    print(f"\nProcessing: {doc_name}")

    try:
        doc = pymupdf.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return 0

    # Clean text
    full_text = re.sub(r'\s+', ' ', full_text)
    full_text = re.sub(r'[^\x00-\x7F]+', ' ', full_text)

    # Split into clauses - multiple strategies
    clauses = []

    # Strategy 1: Split by numbered clauses (1. 2. 3. etc)
    numbered = re.split(r'(?=\d+\.\s+[A-Z])', full_text)
    for clause in numbered:
        clause = clause.strip()
        word_count = len(clause.split())
        if 15 < word_count < 400:
            clauses.append(clause)

    # Strategy 2: Split by sentences if numbered didn't work well
    if len(clauses) < 10:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', full_text)
        # Group sentences into clauses of 2-4 sentences
        grouped = []
        current = []
        for sent in sentences:
            current.append(sent)
            if len(current) >= 3:
                grouped.append(' '.join(current))
                current = []
        if current:
            grouped.append(' '.join(current))

        clauses = [
            c.strip() for c in grouped
            if 15 < len(c.split()) < 400
        ]

    # Remove duplicates within this document
    seen = set()
    unique_clauses = []
    for clause in clauses:
        fingerprint = clause[:50].lower().strip()  # first 50 chars as fingerprint
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_clauses.append(clause)

    clauses = unique_clauses
    print(f"  Extracted {len(clauses)} clauses")

    # Write to CSV
    write_header = (not os.path.exists(output_file)) or os.path.getsize(output_file) == 0
    start = max_id[output_file] + 1  # continue numbering after the highest existing ID

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "clause_id", "source_document", "clause_text", "domain", "label",
                "is_predatory", "annotator_1", "annotator_2", "annotator_3",
                "annotator_4", "final_label", "notes"
            ])

        for i, clause in enumerate(clauses):
            clause_id = f"{domain[:3]}_{start + i:04d}"
            writer.writerow([
                clause_id, doc_name, clause, domain,
                "", "", "", "", "", "", "", ""  # label ... notes: filled later
            ])

    print(f"  Saved to {output_file}")
    return len(clauses)


financial_docs = [
    ("raw_documents/rbi_nbfc_sample_loan_agreement.pdf",
     "RBI NBFC Sample Loan Agreement"),
    ("raw_documents/kredotbee_terms.pdf",
     "KreditBee Terms and Conditions"),
    ("raw_documents/moneytap_terms.pdf",
     "MoneyTap Terms and Conditions"),
    ("raw_documents/dor_fin_rec_289.pdf",
     "RBI DOR Financial Recommendation 289"),
    ("raw_documents/dor_cre_rec_5.pdf",
     "RBI DOR Credit Recommendation 5"),
    ("raw_documents/NT108AA12014D00734F849B0B871142686CC0.pdf",
     "RBI Circular - PSS Act Monetary Penalty & Compounding Framework"),
]

consumer_docs = [
    ("raw_documents/swiggy_terms.pdf",
     "Swiggy Terms of Service"),
    ("raw_documents/zomato_terms.pdf",
     "Zomato Terms of Service"),
    ("raw_documents/amazon_terms.pdf",
     "Amazon India Terms"),
    ("raw_documents/health_insurance_basics.pdf",
     "Health Insurance Policy Basics"),
]

property_docs = [
    ("raw_documents/lucina_land_vs_union_of_india.PDF",
     "Lucina Land Development vs Union of India"),
    ("raw_documents/Ida_vs_gupta.pdf",
     "Lucknow Development Authority vs MK Gupta"),
    ("raw_documents/veerasri_vs_PD.pdf",
     "Veera Srivastava vs Parsvnath Development"),
    ("raw_documents/Sky Land International Pvt. Ltd. vs Kavita P Lalwani on 25 May, 2012.pdf",
     "Sky Land International Pvt Ltd vs Kavita P Lalwani"),
    ("raw_documents/Rohit Kochhar vs Vipul Infrastructure Developers Ltd on 26 November, 2024.pdf",
     "Rohit Kochhar vs Vipul Infrastructure Developers Ltd"),
    ("raw_documents/M_S.Aditya Real Estates vs P.Baskar on 7 March, 2018.pdf",
     "M/S Aditya Real Estates vs P Baskar"),
    ("raw_documents/Parkash Singh And Others vs Joint Development Commissioner on 8 November, 2013.pdf",
     "Parkash Singh & Others vs Joint Development Commissioner"),
    ("raw_documents/Sri K. Pavan Raj vs The Municipal Corporation Of Hyderabad on 3 December, 2007 .pdf",
     "Sri K Pavan Raj vs Municipal Corporation of Hyderabad"),
    ("raw_documents/T. Rameshwar vs Commissioner, Municipal Corporation on 18 March, 2006.pdf",
     "T Rameshwar vs Commissioner, Municipal Corporation"),
    ("raw_documents/Smt_Hombalamma_vs_The_Chief_Executive_Officer_And_The_on_13_March_2026.PDF",
     "Smt Hombalamma vs Chief Executive Officer"),
    ("raw_documents/Hansraj Jain & Sons & Others vs Santanu Chaudhuri & Others on 8 December, 2009.pdf",
     "Hansraj Jain & Sons & Others vs Santanu Chaudhuri & Others"),
    ("raw_documents/Maniac Rao vs The State Of Telangana on 22 September, 2026.pdf",
     "Maniac Rao vs State of Telangana"),
]

family_docs = [
    ("raw_documents/Bharti Garg vs Ashish Garg And Anr on 28 July, 2026.pdf",
     "Bharti Garg vs Ashish Garg & Anr"),
    ("raw_documents/Dhananjay Rathi vs Ruchika Rathi on 13 April, 2026.pdf",
     "Dhananjay Rathi vs Ruchika Rathi"),
    ("raw_documents/Dr. Sangeeta Vishwakarma @ Sangeeta vs Rahul Ranjan on 7 September, 2026.pdf",
     "Dr Sangeeta Vishwakarma vs Rahul Ranjan"),
    ("raw_documents/Harpinder Kaur vs Shamsher Singh on 18 August, 2026.pdf",
     "Harpinder Kaur vs Shamsher Singh"),
    ("raw_documents/Kavita Kumari vs Rakesh Kumar on 22 July, 2026.pdf",
     "Kavita Kumari vs Rakesh Kumar"),
    ("raw_documents/Manu Chopra vs Parul Chopra on 17 August, 2026.pdf",
     "Manu Chopra vs Parul Chopra"),
    ("raw_documents/Sandip Shankarlal Kedia vs Pooja Kedia on 29 April, 2013.pdf",
     "Sandip Shankarlal Kedia vs Pooja Kedia"),
    ("raw_documents/Shri B R Vijay Kumar vs Smt. Nagarathna on 3 September, 2026.pdf",
     "Shri B R Vijay Kumar vs Smt Nagarathna"),
    ("raw_documents/Sou. Ramkuvar Madanlal Atale vs Madanlal Surajkaran Atale on 21 August, 1974.pdf",
     "Sou Ramkuvar Madanlal Atale vs Madanlal Surajkaran Atale"),
    ("raw_documents/Tillana Shripal Shah W_O Shripal vs State Of Gujarat on 18 March, 2026.pdf",
     "Tillana Shripal Shah (W/O Shripal) vs State of Gujarat"),
    ("raw_documents/Xxxxxxxxxxxx vs Xxxxxxxxxxxx on 21 April, 2026.pdf",
     "Redacted Parties Case (21 April 2026)"),
    ("raw_documents/Xxxxxxxxxxxx vs Xxxxxxxxxxxx on 24 March, 2026.pdf",
     "Redacted Parties Case (24 March 2026)"),
    ("raw_documents/Arti Mehta vs The State Of Madhya Pradesh on 25 May, 2026.pdf",
     "Arti Mehta vs State of Madhya Pradesh"),  # confirmed: 498A quashing petition, substance is matrimonial discord & divorce proceedings
    ("raw_documents/Jyotsna Goel vs State Of Haryana And Others on 24 March, 2026.pdf",
     "Jyotsna Goel vs State of Haryana & Others"),  # confirmed: child custody dispute between estranged spouses
    ("raw_documents/Snehasis Rashbihari Batabyal vs Dolan Acharya on 1 September, 2026.pdf",
     "Snehasis Rashbihari Batabyal vs Dolan Acharya"),  # confirmed: divorce + custody modification, joint parenting order
    ("raw_documents/Shrawan S_O Rajeramji Banasure vs Ankita W_O Shrawan Banasure on 24 September, 2025.pdf",
     "Shrawan S/O Rajeramji Banasure vs Ankita W/O Shrawan Banasure"),  # confirmed: PWDVA interim maintenance + restitution of conjugal rights
    ("raw_documents/Hirawati Bhaskar Ate vs Bhaskar Patruji Ate And Ors on 15 October, 2025.pdf",
     "Hirawati Bhaskar Ate vs Bhaskar Patruji Ate & Ors"),  # confirmed: PWDVA maintenance appeal
    ("raw_documents/Bimbisar Baburao Gaikwad vs Diksha Wo Bimbisar Gaikwad @ Diksha D_O ... on 15 January, 2025.pdf",
     "Bimbisar Baburao Gaikwad vs Diksha W/O Bimbisar Gaikwad"),  # confirmed: PWDVA Section 12 application, domestic violence + interim maintenance
    ("raw_documents/Jyoti Abhijeet Kandage vs Abhijeet Narayan Kandage on 15 September, 2025.pdf",
     "Jyoti Abhijeet Kandage vs Abhijeet Narayan Kandage"),  # confirmed: PWDVA proceedings transfer to Family Court
    ("raw_documents/Mirza Khalid Baig S_O Mirza Sujat Baig ... vs Fouziya Mariyam W_O Mirza Khalid Baig on 19 December, 2025.pdf",
     "Mirza Khalid Baig vs Fouziya Mariyam Baig"),  # confirmed: PWDVA + 498-A quashing petition, matrimonial dispute
]

def process_domain(docs, domain_label, output_prefix):
    print(f"\n📄 Processing {domain_label} Documents...")
    total = 0
    for pdf_path, doc_name in docs:
        if os.path.exists(pdf_path):
            count = extract_clauses_from_pdf(
                pdf_path, output_prefix, doc_name
            )
            total += count
        else:
            print(f"  ⚠️  File not found: {pdf_path} — skipping")
    return total


def main():
    print("=" * 60)
    print("FinGuard AI — Clause Extraction Tool")
    print("=" * 60)

    total_financial = process_domain(financial_docs, "Financial", "FINANCIAL")
    total_consumer = process_domain(consumer_docs, "Consumer", "CONSUMER")
    total_property = process_domain(property_docs, "Property", "PROPERTY")
    total_family = process_domain(family_docs, "Family Law", "FAMILY")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"Financial clauses:  {total_financial}")
    print(f"Consumer clauses:   {total_consumer}")
    print(f"Property clauses:   {total_property}")
    print(f"Family Law clauses: {total_family}")
    print(f"Total:              {total_financial + total_consumer + total_property + total_family}")
    print(f"\nFiles saved in: datasets/")
    print("  → financial_clauses.csv")
    print("  → consumer_clauses.csv")
    print("  → property_clauses.csv")
    print("  → family_clauses.csv")
    print("\nNext: Upload CSVs to Google Sheets and start labeling!")
    print("=" * 60)


if __name__ == "__main__":
    main()