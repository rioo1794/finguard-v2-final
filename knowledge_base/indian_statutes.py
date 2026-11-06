# Indian Legal Knowledge Base (final version, 24 Oct 2026)
# Paraphrased summaries of key provisions, NOT official text. Verify against India Code
# (indiacode.nic.in) or the RBI website before quoting in the paper.
# Format rules used by modules/rag_pipeline.py: first line of each block = act name (skipped),
# then one provision per line written as "Label: text". Labels must not contain a colon.
# Comments marked "NOT RE-CHECKED" mark lines not confirmed online in this session.

INDIAN_CONTRACT_ACT = """
Indian Contract Act 1872:
Section 16 Undue influence: A contract is induced by undue influence where one party is in a position to dominate the will of the other and uses that position to obtain an unfair advantage. Where a transaction appears unconscionable, the burden of proving it was not induced by undue influence lies on the party in the dominant position.
Section 23 Unlawful object or consideration: The consideration or object of an agreement is unlawful if it is forbidden by law, would defeat the provisions of any law, is fraudulent, involves injury to a person or property, or is regarded by the court as immoral or opposed to public policy. Such an agreement is void.
Section 27 Agreement in restraint of trade: Every agreement that restrains anyone from exercising a lawful profession, trade or business is void to that extent, subject to the exception for sale of goodwill.
Section 28 Agreements in restraint of legal proceedings: An agreement that absolutely restricts a party from enforcing his rights through the usual legal proceedings, or that limits the time within which he may enforce them, or that extinguishes his rights or discharges the other party from liability after a specified period so as to restrict enforcement, is void to that extent. Agreements to refer future disputes to arbitration are saved by an exception.
Section 73 Compensation for breach: A party who suffers by a breach of contract is entitled to compensation for loss or damage that naturally arose in the usual course of things from the breach or that the parties knew, when making the contract, was likely to result. Compensation is not given for remote and indirect loss.
Section 74 Penalty stipulated for breach: When a contract names a sum to be paid on breach, or contains a penalty clause, the aggrieved party is entitled to reasonable compensation not exceeding the amount named, whether or not actual damage or loss is proved.
"""
# Headings of sections 23, 27, 28, 73 and 74 confirmed online.  # NOT RE-CHECKED: section 16 and the wording of each section.

CONSUMER_PROTECTION_ACT = """
Consumer Protection Act 2019:
Section 2(7) Consumer: A person who buys goods or hires or avails of a service for a consideration. It does not include a person who obtains goods for resale or goods or services for a commercial purpose.
Section 2(9) Consumer rights: The six rights are protection against hazardous goods and services, information about quality, quantity, potency, purity, standard and price, access to a variety of goods and services at competitive prices, the right to be heard, the right to seek redressal against unfair or restrictive trade practices or unscrupulous exploitation, and the right to consumer awareness.
Section 2(11) Deficiency: Any fault, imperfection, shortcoming or inadequacy in the quality, nature and manner of performance required by law or undertaken under a contract. It includes negligence or omission that causes loss or injury to the consumer and deliberate withholding of relevant information.
Section 2(46) Unfair contract: A contract between a manufacturer, trader or service provider and a consumer with terms that cause a significant change in the consumer's rights. Examples include excessive security deposits, penalties for breach that are disproportionate to the loss, refusing to accept early repayment of debts on payment of the applicable penalty, unilateral termination without reasonable cause, assignment of the contract to the consumer's detriment without consent, and unreasonable charges, obligations or conditions that put the consumer at a disadvantage.
Section 2(47) Unfair trade practice: A trade practice that adopts an unfair method or deceptive practice to promote the sale, use or supply of goods or services, including false or misleading representations and advertisements, not issuing a bill or cash memo or receipt, refusing to take back defective goods or withdraw deficient services and refund the price within the stated period or 30 days, and disclosing personal information given in confidence by the consumer unless the law requires it.
Section 49(2) State Commission power over unfair contracts: The State Consumer Disputes Redressal Commission can declare any term of a contract that is unfair to a consumer to be null and void.
Section 59(2) National Commission power over unfair contracts: The National Consumer Disputes Redressal Commission can declare any term of a contract that is unfair to a consumer to be null and void.
Sections 84 to 86 Product liability: A product manufacturer, product service provider and product seller are liable to compensate a consumer for harm caused by a defective product or deficient service, as set out in these sections.
Section 100 Remedy in addition to other laws: The Act is in addition to, and not in derogation of, any other law in force. Consumer remedies exist alongside other remedies.
Case Emaar MGF Land Ltd v Aftab Singh (Supreme Court 2018): An arbitration clause in a builder-buyer agreement cannot circumscribe the jurisdiction of a consumer forum, so a consumer can still complain to the consumer commission.
Dark patterns Scope: The Guidelines for Prevention and Regulation of Dark Patterns, 2023 were issued by the Central Consumer Protection Authority on 30 November 2023 under section 18 of the Consumer Protection Act 2019. No person, including any platform, may engage in a dark pattern. They apply to all platforms systematically offering goods or services in India, to advertisers and to sellers, and they are in addition to other laws. A dark pattern is a deceptive design practice that misleads or tricks users into doing something they did not intend, by impairing consumer autonomy or choice, and can amount to a misleading advertisement, an unfair trade practice or a violation of consumer rights. The illustrations in the Annexure are guidance only.
Dark pattern False urgency: Falsely stating or implying urgency or scarcity to push an immediate purchase, such as showing false popularity or claiming stock is more limited than it is.
Dark pattern Basket sneaking: Adding items, services or payments such as charity or donations at checkout without the user's consent, so the total exceeds the price of what the user chose. Free samples, complimentary services and necessary fees disclosed at purchase, such as delivery charges, gift wrapping and government taxes, are excluded. Examples are pre-ticked paid extras and auto-added travel insurance.
Dark pattern Confirm shaming: Using a phrase, video, audio or other means to create fear, shame, ridicule or guilt so the user buys a product or continues a subscription, for commercial gain, for example "I will stay unsecured" when a user declines insurance.
Dark pattern Forced action: Forcing a user to buy additional goods, subscribe or sign up for an unrelated service, or share personal information, in order to get the product originally intended. Examples are requiring an upgrade to keep using a service already paid for, forcing a newsletter subscription, demanding Aadhaar or credit card details not needed for the purchase, demanding contacts or social network details, and making privacy settings hard to change.
Dark pattern Subscription trap: Making cancellation of a paid subscription impossible or complex and lengthy, hiding the cancellation option, forcing payment details or auto-debit authorisation to get a free subscription, or making cancellation instructions ambiguous or confusing.
Dark pattern Interface interference: Designing the interface to highlight some information while obscuring other relevant information, so the user is misdirected, for example a faint "No" option, a tiny cancellation symbol, or an X icon that opens another advertisement.
Dark pattern Bait and switch: Advertising one outcome but deceptively delivering another, such as a product shown as available that turns out to be out of stock and is replaced by a costlier one.
Dark pattern Drip pricing: Not revealing elements of the price upfront or revealing them surreptitiously, charging more than the amount shown at checkout, advertising something as free without disclosing that continued use needs in-app purchases, or preventing use of a service already paid for unless something extra is bought. A marketplace is not liable for price changes made by third party sellers or beyond its control.
Dark pattern Disguised advertisement: Presenting advertisements as other content such as user-generated content or news articles so users click on them. The seller or advertiser must disclose that the content is an advertisement.
Dark pattern Nagging: Repeated and persistent requests or interruptions that annoy the user to push a transaction for commercial gain, unless the user permitted them, such as constant requests to accept notifications or cookies with no option to say no.
Dark pattern Trick question: Deliberately confusing or vague wording, such as double negatives, to misguide the user into a particular response.
Dark pattern SaaS billing: Collecting recurring payments as surreptitiously as possible, for example no notice when a free trial turns into a paid plan, silent recurring debits or auto-renewal without telling users, charging for features not used, or shady credit card authorisation practices.
Dark pattern Rogue malware: Using ransomware or scareware to make users believe their device has a virus so they pay for a fake removal tool.
"""
# Definitions 2(7), 2(9), 2(11), 2(46), 2(47), sections 49(2), 59(2), 84-86, the Emaar case and all dark pattern entries were confirmed online / against the Gazette notification.
# NOT RE-CHECKED: the list of examples under 2(46) and section 100.

RBI_DIGITAL_LENDING = """
RBI Digital Lending Directions 2025:
Scope: The Reserve Bank of India (Digital Lending) Directions, 2025, dated 8 May 2025, apply to digital lending by regulated entities such as banks, NBFCs and housing finance companies. They consolidate the 2022 Guidelines on Digital Lending and the 2023 Default Loss Guarantee Guidelines.
Key Fact Statement: A regulated entity must give the borrower a Key Fact Statement in a standardised format before the loan contract is executed.
Annual Percentage Rate: The APR, the all-inclusive cost of the digital loan to the borrower, must be disclosed upfront and be part of the Key Fact Statement.
Fees not in the Key Fact Statement: Under the 2022 digital lending guidelines (para 5.2.3), fees and charges not mentioned in the Key Fact Statement cannot be charged to the borrower at any stage of the loan.
Cooling-off period: The borrower must be given an explicit option to exit the digital loan during a cooling-off period by repaying the principal and the proportionate APR without penalty. The board of the lender sets the period, which must be at least one day. The lender may charge a reasonable one-time processing fee on exit.
"""
# Confirmed online: scope and date, Key Fact Statement before execution, APR upfront, para 5.2.3 of the 2022 guidelines, cooling-off of at least one day with processing fee.

RBI_PREPAYMENT = """
RBI Pre-payment Charges Directions 2025:
Individuals non-business floating rate: Under the Reserve Bank of India (Pre-payment Charges on Loans) Directions, 2025, regulated entities cannot levy pre-payment or foreclosure charges on floating rate loans given to individuals for non-business purposes, even if there is a co-obligant, whatever the source of repayment funds.
Applicability date: The Directions apply to loans and advances sanctioned or renewed on or after 1 January 2026.
Hybrid rate loans: For loans with both fixed and floating components, the prohibition applies if the loan is on a floating rate at the time of pre-payment.
Business loans to individuals and small enterprises: The Directions also bar pre-payment charges on floating rate loans for business purposes to individuals and micro and small enterprises, for the categories of lenders they specify.
"""
# Confirmed online (issued 2 July 2025, effective 1 January 2026). Fixed rate personal loans are not covered by the ban.

CREDIT_INFORMATION_ACT = """
Credit Information Companies Act 2005:
Section 17 Collection and furnishing of credit information: Credit information is collected from credit institutions and furnished to specified users, as provided in this section.
Section 21 Correction of credit information: A borrower can ask the credit institution or the credit information company to correct or update credit information. Under section 21(3) read with the Credit Information Companies Rules 2006, the correction must be done within 30 days of the request.
RBI fortnightly reporting 2024: Under the RBI notification dated 8 August 2024, credit institutions must report credit information to credit information companies fortnightly, within seven calendar days of the end of the reporting fortnight.
"""
# Confirmed online (section 17 heading, section 21(3) and the 30-day limit, RBI notification RBI/2024-25/60).

TRANSFER_OF_PROPERTY_ACT = """
Transfer of Property Act 1882:
Section 54 Sale: A sale is a transfer of ownership in exchange for a price paid or promised or part-paid and part-promised. In the case of tangible immovable property worth one hundred rupees or more, or a reversion or other intangible thing, the transfer can be made only by a registered instrument. A contract for sale does not by itself create any interest in or charge on the property.
Section 55(1)(a) Seller must disclose defects: The seller must disclose to the buyer any material defect in the property or in the seller's title that the seller knows of, that the buyer does not know of, and that the buyer could not discover with ordinary care.
Section 55(1)(b) Seller must produce title documents: The seller must produce all documents of title relating to the property in his possession for examination by the buyer.
Section 58 Mortgage: A mortgage is the transfer of an interest in specific immovable property to secure the payment of money advanced or to be advanced by way of loan, an existing or future debt, or performance of an engagement that may give rise to a pecuniary liability.
Section 107 Lease: A lease of immovable property from year to year, or for any term exceeding one year, or reserving a yearly rent, can be made only by a registered instrument.
Section 123 Gift of immovable property: A gift of immovable property must be made by a registered instrument signed by or on behalf of the donor and attested by at least two witnesses.
"""
# Sections 54 and 55(1) confirmed online. NOT RE-CHECKED: wording of sections 58, 107 and 123 (their headings and registered-instrument requirement were confirmed).

REGISTRATION_ACT = """
Registration Act 1908:
Section 17 Compulsory registration: Certain documents must be registered, including sale deeds and other instruments that create or transfer rights in immovable property worth one hundred rupees or more, gifts of immovable property, and leases for more than one year.
"""
# Link between section 54 of the Transfer of Property Act and section 17 confirmed online. NOT RE-CHECKED: full list of documents under section 17.

REAL_ESTATE_ACT = """
Real Estate (Regulation and Development) Act 2016:
Section 13 Advance before agreement for sale: A promoter cannot accept more than 10 percent of the cost of the apartment, plot or building as an advance or application fee without first entering into a written agreement for sale and registering it.
Section 14 Sanctioned plans: The promoter must develop the project as per the sanctioned plans and specifications. Other additions or alterations need the previous consent of at least two-thirds of the allottees.
Section 14(3) Structural defects: If a structural defect or other defect in workmanship, quality or provision of services is brought to the promoter's notice within five years of handing over possession, the promoter must rectify it within 30 days at no cost to the allottee.
Section 18(1) Delay in possession: If the promoter fails to complete the project or give possession by the date in the agreement for sale, the allottee can withdraw and claim a refund of the amount paid with interest and compensation, or, if the allottee does not withdraw, receive interest for every month of delay until possession.
Section 19(6) Allottee payments: An allottee who enters into an agreement for sale must make the necessary payments in the manner and within the time specified in the agreement.
Section 19(7) Interest payable by allottee: An allottee is liable to pay interest at the prescribed rate for any delay in making payments.
Section 2(za) Interest symmetry: The rate of interest chargeable from the allottee by the promoter on default must be equal to the rate the promoter must pay the allottee on default. A clause with a much higher penalty rate against the buyer than against the builder conflicts with this.
"""
# Sections 13, 14 (two-thirds consent), 18(1), 19(6), 19(7) and the interest symmetry in 2(za) confirmed online. NOT RE-CHECKED: the five-year, 30-day detail under 14(3).

HINDU_MARRIAGE_ACT = """
Hindu Marriage Act and Family Law:
Hindu Marriage Act Section 13B Divorce by mutual consent: Both parties may jointly petition for divorce if they have been living separately for one year or more, have not been able to live together, and have mutually agreed that the marriage should be dissolved. The second motion can be made not earlier than six months after the petition. Courts have held this six month period is directory and can be waived in suitable cases. Consent must continue until the decree is passed.
Hindu Marriage Act Section 14 One year bar: No petition for divorce may be presented within one year of the marriage, except with the leave of the court in cases of exceptional hardship or depravity.
Hindu Marriage Act Section 24 Interim maintenance: If either spouse has no independent income sufficient for support and the necessary expenses of the proceeding, the court may order the other spouse to pay the expenses of the proceeding and a monthly sum during the proceeding.
Hindu Marriage Act Section 25 Permanent alimony and maintenance: At the time of passing a decree or later, the court may order one spouse to pay maintenance to the other, having regard to the income and property of both, the conduct of the parties and other circumstances. The order can be varied or rescinded if circumstances change.
Hindu Marriage Act Section 26 Custody of children: In any proceeding under the Act, the court may pass orders in the decree about the custody, maintenance and education of minor children, consistently with their wishes wherever possible.
Hindu Marriage Act Section 27 Disposal of property: The court may make provisions in the decree about property presented at or about the time of marriage that belongs jointly to the husband and wife.
Hindu Minority and Guardianship Act Section 13 Welfare of minor: In the appointment or declaration of any person as guardian of a Hindu minor, the welfare of the minor is the paramount consideration.
"""
# Sections 13B, 14, 24, 25 and 26 confirmed online. NOT RE-CHECKED: the wording of sections 27 and 13 (Hindu Minority and Guardianship Act).

# All statutes combined for the unified knowledge base.
# Keys must match ACT_CODES and DOMAIN_ACTS in modules/rag_pipeline.py.
ALL_STATUTES = {
    "RBI Digital Lending Directions 2025": RBI_DIGITAL_LENDING,
    "RBI Pre-payment Charges Directions 2025": RBI_PREPAYMENT,
    "Credit Information Companies Act 2005": CREDIT_INFORMATION_ACT,
    "Transfer of Property Act": TRANSFER_OF_PROPERTY_ACT,
    "Registration Act 1908": REGISTRATION_ACT,
    "Real Estate (Regulation and Development) Act 2016": REAL_ESTATE_ACT,
    "Hindu Marriage Act and Family Law": HINDU_MARRIAGE_ACT,
    "Consumer Protection Act 2019": CONSUMER_PROTECTION_ACT,
    "Indian Contract Act 1872": INDIAN_CONTRACT_ACT,
}