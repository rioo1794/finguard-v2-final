"""
app.py
FinGuard AI - Streamlit frontend. Local-only inference (Ollama + BERT). No cloud calls.
"""
import gc
import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -- page config -------------------------------------------------------------
st.set_page_config(
    page_title="FinGuard AI",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTabs [data-baseweb="tab"] { color: #aabbcc; }
    .stTabs [aria-selected="true"] { color: #4fc3f7; }
    .metric-card { background: #1a1f2e; border-radius: 8px; padding: 12px; margin: 4px 0; }
    .domain-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# -- imports -----------------------------------------------------------------
from knowledge_base.indian_statutes import ALL_STATUTES
from modules.ingestion import extract_text, scrub_pii, chunk_text
from modules.domain_classifier import classify_domain, DOMAIN_INFO, domain_key
from modules.clause_classifier import predict_clauses, has_model
from modules.rag_pipeline import (
    build_faiss_index,
    build_knowledge_base,
    prewarm_models,
    verify_ollama_setup,
)
from modules.compliance_checker import run_compliance_check
from modules.predatory_detector import detect_predatory_clauses
from modules.cibil_analyzer import analyze_cibil_impact
from modules.multiparty_analyzer import analyze_multiparty
from modules.precedent_retriever import retrieve_precedents
from modules.fraud_detector import detect_fraud_patterns
from modules.plain_language import generate_plain_language
from modules.report_generator import generate_report_docx, generate_report_text


# -- heavy resources, loaded once --------------------------------------------
@st.cache_resource(show_spinner="Loading Indian legal knowledge base...")
def load_knowledge_base():
    kb_chunks, kb_index = build_knowledge_base(ALL_STATUTES)
    return kb_chunks, kb_index


@st.cache_resource(show_spinner="Prewarming local Llama models...")
def warm_models():
    try:
        return prewarm_models()
    except Exception as e:
        return [f"prewarm failed: {e}"]


kb_chunks, kb_index = load_knowledge_base()
_ollama_ok = verify_ollama_setup()
_warm_status = warm_models() if _ollama_ok else ["Ollama not reachable - analysis tabs will fail"]


# -- sidebar -----------------------------------------------------------------
with st.sidebar:
    st.markdown("## FinGuard AI")
    st.caption("Multi-Domain Indian Legal Document Intelligence")
    st.divider()

    if "domain" in st.session_state:
        domain = st.session_state.domain
        info = DOMAIN_INFO[domain]
        st.markdown("**Detected Domain:**")
        st.markdown(
            f"<div style='background:{info['color']}22; "
            f"border-left:4px solid {info['color']}; "
            f"padding:8px; border-radius:4px;'>"
            f"{info['icon']} {info['description']}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        if "safety_score" in st.session_state:
            score = st.session_state.safety_score
            color = "#4caf50" if score >= 70 else "#ff9800" if score >= 40 else "#f44336"
            st.markdown("**Safety Score:**")
            st.markdown(
                f"<div style='font-size:36px; font-weight:800; "
                f"color:{color}; text-align:center;'>{score}/100</div>",
                unsafe_allow_html=True,
            )
            st.divider()

        if "doc_stats" in st.session_state:
            stats = st.session_state.doc_stats
            st.markdown("**Document Stats:**")
            st.markdown(f"Pages: {stats['pages']}")
            st.markdown(f"Chunks: {stats['chunks']}")
            st.markdown(f"Words: {stats['words']:,}")

    st.divider()
    st.caption("Model status:")
    for line in _warm_status:
        st.caption(line)
    st.divider()
    st.caption("SAKEC - AI & DS Department")
    st.caption("Research Project 2026")


# -- main --------------------------------------------------------------------
st.title("FinGuard AI")
st.caption("Multi-Domain Indian Legal Document Intelligence Platform")
st.info(
    "General information only. This is not legal advice. "
    "Consult a qualified Indian advocate before making decisions based on this report."
)

uploaded_file = st.file_uploader(
    "Upload a legal document (PDF)",
    type=["pdf"],
    help="Supports: Loan agreements, Property deeds, Divorce petitions, Consumer agreements",
)

if not uploaded_file:
    st.info("Upload any Indian legal document to begin analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### Financial")
        st.caption("Loan agreements, NBFC documents, Securities, Pledge documents")
    with col2:
        st.markdown("### Property")
        st.caption("Sale deeds, Lease agreements, Title documents, Mortgages")
    with col3:
        st.markdown("### Family Law")
        st.caption("Divorce petitions, Custody applications, Maintenance agreements")
    with col4:
        st.markdown("### Consumer")
        st.caption("Service agreements, Warranties, Subscription terms, Complaints")
    st.stop()


# -- document processing -----------------------------------------------------
with st.spinner("Reading and processing document..."):
    raw_text, num_pages = extract_text(uploaded_file)
    clean_text = scrub_pii(raw_text)
    chunks = chunk_text(clean_text)
    chunks_array = np.array(chunks)
    doc_index, _ = build_faiss_index(list(chunks))

    domain, confidence, domain_scores, subtype = classify_domain(clean_text)
    st.session_state.domain = domain
    st.session_state.subtype = subtype
    st.session_state.doc_stats = {
        "pages": num_pages,
        "chunks": len(chunks),
        "words": len(clean_text.split()),
    }

    # BERT pre-classification (Consumer and Financial only)
    dkey = domain_key(domain)
    try:
        bert_preds = predict_clauses(list(chunks_array), dkey) if has_model(dkey) else None
    except Exception as e:
        bert_preds = None
        st.warning(f"BERT classifier could not run: {e}")
    st.session_state.bert_preds = bert_preds
    st.session_state.bert_domain = dkey


info = DOMAIN_INFO[domain]
st.success(f"Document processed - {num_pages} pages, {len(chunks)} chunks indexed")

subtype = st.session_state.get("subtype", None)
subtype_label = f" - {subtype.replace('_', ' ').title()}" if subtype else ""

st.markdown(
    f"<div style='background:{info['color']}22; "
    f"border:1px solid {info['color']}; "
    f"border-radius:8px; padding:12px; margin:8px 0;'>"
    f"<b>Detected Domain:</b> {info['icon']} "
    f"{info['description']}{subtype_label} "
    f"<span style='color:{info['color']};'>({confidence:.0%} confidence)</span></div>",
    unsafe_allow_html=True,
)

with st.expander("Domain confidence breakdown"):
    cols = st.columns(4)
    for i, (d, score) in enumerate(domain_scores.items()):
        with cols[i]:
            di = DOMAIN_INFO[d]
            st.metric(f"{di['icon']} {d}", f"{score}%")

if bert_preds is not None:
    n_risky = sum(1 for p in bert_preds if p["label"] == "RISKY")
    st.markdown(
        f"**BERT clause classifier ({dkey}):** {n_risky} of {len(bert_preds)} "
        f"clauses flagged RISKY"
    )
else:
    st.markdown(
        f"**BERT clause classifier:** not trained for '{dkey}'. "
        f"Falling back to baseline methods for this domain."
    )

st.divider()


# -- tabs --------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Compliance", "Predatory", "CIBIL", "Multi-Party",
    "Fraud Check", "Precedents", "Plain Language", "Full Report",
])

CONTRACT_INDICATORS = ["party", "agreement", "whereas", "hereinafter", "herein",
                       "shall pay", "shall be", "obligations", "signature"]
text_lower = clean_text.lower()
contract_score = sum(1 for kw in CONTRACT_INDICATORS if kw in text_lower)
is_likely_contract = contract_score >= 4

if not is_likely_contract:
    st.warning("This document does not appear to be a legal contract "
               "(no parties, obligations, or signature clauses detected). "
               "Compliance and fraud analysis work best on actual contracts.")

# -- TAB 1: COMPLIANCE -------------------------------------------------------
with tab1:
    st.subheader(f"Compliance Check - {info['icon']} {domain}")
    if st.button("Run Compliance Check", key="compliance"):
        with st.spinner("Checking against Indian legal requirements..."):
            results = run_compliance_check(
                domain, chunks_array, doc_index, kb_chunks, kb_index,
            )
            st.session_state.compliance_results = results

        pass_c = sum(1 for r in results if r["status"] == "PASS")
        fail_c = sum(1 for r in results if r["status"] == "FAIL")
        unc_c = sum(1 for r in results if r["status"] == "UNCLEAR")
        total = len(results)
        score = int((pass_c / total) * 100) if total else 0
        st.session_state.safety_score = score

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Passed", pass_c)
        col2.metric("Failed", fail_c)
        col3.metric("Unclear", unc_c)
        col4.metric("Score", f"{score}/100")

        st.divider()
        for r in results:
            if r["status"] == "PASS":
                st.success(f"**{r['check']}**")
            elif r["status"] == "FAIL":
                st.error(f"**{r['check']}**")
                with st.expander("See details"):
                    st.write(r["result"])
            else:
                st.warning(f"**{r['check']}**")
                with st.expander("See details"):
                    st.write(r["result"])


# -- TAB 2: PREDATORY --------------------------------------------------------
with tab2:
    st.subheader("Predatory & Exploitative Clause Detection")

    bert_preds = st.session_state.get("bert_preds")
    if bert_preds:
        with st.expander("BERT clause-level risk scores", expanded=False):
            risky = [(i, p) for i, p in enumerate(bert_preds) if p["label"] == "RISKY"]
            if not risky:
                st.success("No clauses flagged RISKY by BERT.")
            else:
                for i, p in risky:
                    st.markdown(f"**Clause {i + 1}  |  p_risky = {p['p_risky']}**")
                    st.caption(str(chunks_array[i])[:400])

    if st.button("Scan for Predatory Clauses", key="predatory"):
        with st.spinner("Scanning for exploitative patterns..."):
            result = detect_predatory_clauses(
                domain, chunks_array, doc_index, kb_chunks, kb_index,
            )
            st.session_state.predatory_result = result

        if "LIKELY PREDATORY" in result:
            st.error("Likely Predatory Document Detected")
        elif "SUSPICIOUS" in result:
            st.warning("Suspicious Patterns Found")
        else:
            st.success("No Major Predatory Patterns Detected")
        st.divider()
        st.markdown(result)


# -- TAB 3: CIBIL ------------------------------------------------------------
with tab3:
    st.subheader("CIBIL Credit Score Impact Analysis")
    if domain != "FINANCIAL":
        st.info("CIBIL analysis is most relevant for financial documents. Running anyway...")
    if st.button("Analyze CIBIL Impact", key="cibil"):
        with st.spinner("Analyzing credit score impact..."):
            result = analyze_cibil_impact(
                chunks_array, doc_index, kb_chunks, kb_index,
                domain=domain,
            )
            st.session_state.cibil_result = result

        if "HIGH" in result[:200]:
            st.error("HIGH CIBIL Risk Detected")
        elif "MEDIUM" in result[:200]:
            st.warning("Medium CIBIL Risk")
        else:
            st.success("Low CIBIL Risk")
        st.divider()
        st.markdown(result)


# -- TAB 4: MULTI-PARTY ------------------------------------------------------
with tab4:
    st.subheader("Multi-Party Interest Analysis")
    st.caption("Analyzes the document from the perspective of each affected party simultaneously")
    if st.button("Run Multi-Party Analysis", key="multiparty"):
        with st.spinner("Analyzing from each party's perspective..."):
            results = analyze_multiparty(domain, chunks_array, doc_index)
            st.session_state.multiparty_results = results

        for party, analysis in results.items():
            with st.expander(f"{party}", expanded=True):
                if "HEAVILY UNFAIR" in analysis:
                    st.error("Document heavily unfair to this party")
                elif "SLIGHTLY UNFAIR" in analysis:
                    st.warning("Document slightly unfair to this party")
                else:
                    st.success("Document appears balanced")
                st.markdown(analysis)


# -- TAB 5: FRAUD ------------------------------------------------------------
with tab5:
    st.subheader("Fraud Pattern Detection")
    if st.button("Run Fraud Check", key="fraud"):
        with st.spinner("Checking document authenticity..."):
            result, missing = detect_fraud_patterns(
                domain, clean_text, chunks_array, doc_index,
            )
            st.session_state.fraud_result = result

        if "LIKELY FRAUDULENT" in result:
            st.error("Likely Fraudulent Document")
        elif "SUSPICIOUS" in result:
            st.warning("Suspicious Patterns Detected")
        else:
            st.success("Document Appears Authentic")

        if missing:
            st.warning(f"Missing mandatory sections: {', '.join(missing)}")
        st.divider()
        st.markdown(result)


# -- TAB 6: PRECEDENTS -------------------------------------------------------
with tab6:
    st.subheader("Relevant Indian Legal Precedents")
    if st.button("Retrieve Precedents", key="precedents_btn"):
        with st.spinner("Searching legal databases..."):
            precedents = retrieve_precedents(domain)
            st.session_state.precedents = precedents

        for i, prec in enumerate(precedents, 1):
            with st.expander(f"{i}. {prec['case']} ({prec['year']})"):
                st.markdown(f"**Court:** {prec['court']}")
                if prec.get("citation"):
                    st.markdown(f"**Citation:** {prec['citation']}")
                st.markdown(f"**Outcome:** {prec['outcome']}")
                st.markdown(f"**Relevance:** {prec['relevance']}")


# -- TAB 7: PLAIN LANGUAGE ---------------------------------------------------
with tab7:
    st.subheader("Plain Language Explanation")
    st.caption("What this document means in simple everyday language")
    if st.button("Generate Plain Language Summary", key="plain"):
        with st.spinner("Creating simple explanation..."):
            result = generate_plain_language(domain, chunks_array, doc_index)
            st.session_state.plain_result = result
        st.markdown(result)


# -- TAB 8: FULL REPORT ------------------------------------------------------
with tab8:
    st.subheader("Download Full Analysis Report")
    if st.button("Generate Full Report", key="report"):
        required = ["compliance_results", "predatory_result", "multiparty_results"]
        missing_analyses = [r for r in required if r not in st.session_state]

        if missing_analyses:
            st.warning(f"Please run these analyses first: {', '.join(missing_analyses)}")
        else:
            with st.spinner("Compiling full report..."):
                report_kwargs = dict(
                    domain=domain,
                    url_or_filename=uploaded_file.name,
                    safety_score=st.session_state.get("safety_score", 0),
                    compliance_results=st.session_state.get("compliance_results", []),
                    predatory_result=st.session_state.get("predatory_result", "Not analyzed"),
                    cibil_result=st.session_state.get("cibil_result", None),
                    multiparty_results=st.session_state.get("multiparty_results", {}),
                    fraud_result=st.session_state.get("fraud_result", "Not analyzed"),
                    precedents=st.session_state.get("precedents") or [],
                    plain_language=st.session_state.get("plain_result", "Not analyzed"),
                )
                docx_bytes = generate_report_docx(**report_kwargs)
                txt_str = generate_report_text(**report_kwargs)

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "Download Report (.docx)",
                    data=docx_bytes,
                    file_name=f"finguard_report_{domain.lower()}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            with col_b:
                st.download_button(
                    "Download Report (.txt)",
                    data=txt_str,
                    file_name=f"finguard_report_{domain.lower()}.txt",
                    mime="text/plain",
                )
            with st.expander("Preview Report"):
                st.text(txt_str[:2000] + "...")


# -- cleanup -----------------------------------------------------------------
if uploaded_file and "processed" not in st.session_state:
    st.session_state.processed = True
    gc.collect()