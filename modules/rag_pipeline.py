"""
modules/rag_pipeline.py
FinGuard AI - RAG Pipeline (Ollama only, fully local)
Retrieval : one entry per statute provision, all-MiniLM-L6-v2 + FAISS (cosine)
Baseline  : TF-IDF keyword retrieval (used for evaluation)
Generation: Llama 3.2 3B via Ollama, with automatic citation and anchor checks.
            Falls back to Llama 3.2 1B on failure.

Run a demo from the project root:  python modules/rag_pipeline.py
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import re
import sys
import json
import time
import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------- configuration ----------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:1b")
EMBED_MODEL = "all-MiniLM-L6-v2"
KB_DIR = os.path.join(ROOT, "knowledge_base", "index")
CTX = 2048
MAX_CLAUSE_CHARS = 1200
MAX_SOURCE_CHARS = 320
DEFAULT_NUM_PREDICT = 280
DEFAULT_TEMPERATURE = 0.15

embedder = SentenceTransformer(EMBED_MODEL)

ACT_CODES = {
    "RBI Digital Lending Directions 2025": "RBI_DL",
    "RBI Pre-payment Charges Directions 2025": "RBI_PP",
    "Credit Information Companies Act 2005": "CICRA",
    "SEBI LODR Regulations": "SEBI_LODR",
    "Transfer of Property Act": "TPA",
    "Registration Act 1908": "REG",
    "Real Estate (Regulation and Development) Act 2016": "RERA",
    "Hindu Marriage Act and Family Law": "HMA",
    "Consumer Protection Act 2019": "CPA",
    "Indian Contract Act 1872": "ICA",
}

DOMAIN_ACTS = {
    "consumer": ["Consumer Protection Act 2019", "Indian Contract Act 1872"],
    "financial": ["RBI Digital Lending Directions 2025",
                  "RBI Pre-payment Charges Directions 2025",
                  "Credit Information Companies Act 2005",
                  "SEBI LODR Regulations",
                  "Consumer Protection Act 2019",
                  "Indian Contract Act 1872"],
    "property": ["Transfer of Property Act", "Registration Act 1908",
                 "Real Estate (Regulation and Development) Act 2016",
                 "Consumer Protection Act 2019", "Indian Contract Act 1872"],
    "family": ["Hindu Marriage Act and Family Law", "Indian Contract Act 1872"],
}


# ---------------- knowledge base ----------------
def parse_statutes(statutes_dict):
    entries, used = [], set()
    for act, text in statutes_dict.items():
        code = ACT_CODES.get(act) or re.sub(r"\W+", "_", act).strip("_")[:12].upper()
        lines = [l.rstrip() for l in text.strip().splitlines() if l.strip()][1:]
        cur, notes = None, 0
        for line in lines:
            if line[0] in " \t" and cur is not None:
                cur["text"] += " " + line.strip()
                continue
            if cur is not None:
                entries.append(cur)
            label, sep, body = line.partition(":")
            if not sep or len(label) > 70:
                notes += 1
                label, body = f"Note {notes}", line
            cur = {"act": act, "label": label.strip(), "text": body.strip(), "_code": code}
        if cur is not None:
            entries.append(cur)
    for e in entries:
        base = f"{e.pop('_code')}|{e['label']}"
        eid, n = base, 2
        while eid in used:
            eid, n = f"{base} #{n}", n + 1
        used.add(eid)
        e["id"] = eid
    return entries


def _embed(texts):
    v = embedder.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(v, dtype="float32")


def _embed_text(e):
    return f"{e['act']} {e['label']}: {e['text']}"


def build_faiss_index(texts):
    emb = _embed(texts)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, emb


_KB = None


def load_kb(rebuild=False):
    global _KB
    if _KB is not None and not rebuild:
        return _KB
    from knowledge_base.indian_statutes import ALL_STATUTES
    entries = parse_statutes(ALL_STATUTES)
    os.makedirs(KB_DIR, exist_ok=True)
    idx_path = os.path.join(KB_DIR, "statutes.faiss")
    meta_path = os.path.join(KB_DIR, "statutes_meta.json")
    index = None
    if not rebuild and os.path.exists(idx_path) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            if json.load(f) == entries:
                index = faiss.read_index(idx_path)
    if index is None:
        index, _ = build_faiss_index([_embed_text(e) for e in entries])
        faiss.write_index(index, idx_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=1)
    _KB = {"entries": entries, "index": index}
    return _KB


# ---------------- retrieval ----------------
def _pick(order, scores, k, domain):
    kb = load_kb()
    allowed = set(DOMAIN_ACTS.get((domain or "").lower(), [])) or None
    out = []
    for i in order:
        e = kb["entries"][int(i)]
        if allowed and e["act"] not in allowed:
            continue
        out.append({**e, "score": float(scores[int(i)])})
        if len(out) == k:
            break
    return out


def retrieve(query, k=4, domain=None):
    kb = load_kb()
    n = len(kb["entries"])
    sc, ids = kb["index"].search(_embed([query]), n)
    scores = np.zeros(n, dtype="float32")
    scores[ids[0]] = sc[0]
    return _pick(ids[0], scores, k, domain)


_TFIDF = None


def keyword_retrieve(query, k=4, domain=None):
    global _TFIDF
    kb = load_kb()
    if _TFIDF is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        mat = vec.fit_transform([_embed_text(e) for e in kb["entries"]])
        _TFIDF = (vec, mat)
    vec, mat = _TFIDF
    scores = (mat @ vec.transform([query]).T).toarray().ravel()
    return _pick(np.argsort(-scores), scores, k, domain)


def build_knowledge_base(statutes_dict):
    entries = parse_statutes(statutes_dict)
    chunks = [f"{e['act']} - {e['label']}: {e['text']}" for e in entries]
    index, _ = build_faiss_index(chunks)
    return chunks, index


def retrieve_chunks(query, chunks, index, k=4):
    _, ids = index.search(_embed([query]), min(k, len(chunks)))
    return "\n\n".join(chunks[i] for i in ids[0] if 0 <= i < len(chunks))


# ---------------- Ollama calls ----------------
def _chat_once(model, system_prompt, user_prompt, num_predict, temperature, context_window):
    return ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "num_ctx": context_window,
            "num_predict": num_predict,
            "temperature": temperature,
        },
        keep_alive="1h",
    )


def ask_llm(system_prompt, user_prompt, model=None, max_tokens=DEFAULT_NUM_PREDICT,
            temperature=DEFAULT_TEMPERATURE, context_window=CTX):
    """Calls primary model; on any failure (timeout, OOM, crash) retries on fallback."""
    primary = model or OLLAMA_MODEL
    try:
        resp = _chat_once(primary, system_prompt, user_prompt,
                          max_tokens, temperature, context_window)
        return resp["message"]["content"]
    except Exception as e:
        if primary == OLLAMA_FALLBACK_MODEL:
            raise
        print(f"[ask_llm] primary model '{primary}' failed ({e}); retrying with '{OLLAMA_FALLBACK_MODEL}'")
        resp = _chat_once(OLLAMA_FALLBACK_MODEL, system_prompt, user_prompt,
                          max_tokens, temperature, context_window)
        return resp["message"]["content"]


def ask_llm_fast(system_prompt, user_prompt):
    return ask_llm(system_prompt, user_prompt, model=OLLAMA_FALLBACK_MODEL,
                   max_tokens=400, context_window=1536)


def prewarm_models():
    """Load both models into RAM. Returns a status list. Never raises."""
    status = []
    for name in (OLLAMA_MODEL, OLLAMA_FALLBACK_MODEL):
        try:
            t = time.time()
            _chat_once(name, "Warmup.", "Reply with: ready",
                       num_predict=4, temperature=0.0, context_window=512)
            status.append(f"{name}: {time.time() - t:.1f}s")
        except Exception as e:
            status.append(f"{name}: FAILED ({e})")
    return status

# ---------------- explanation with citation + anchor checks ----------------
SYSTEM_PROMPT = (
    "You explain Indian contract clauses to people who are not lawyers. "
    "STRICT RULES:\n"
    "1. Use ONLY the numbered sources provided below. The sources are labelled [S1], [S2], [S3] "
    "in the order given. Cite them by writing [S1], [S2], or [S3].\n"
    "2. Never write [S#] as a literal placeholder. If you mean source 1, write [S1].\n"
    "3. Every factual sentence in WHY IT MAY BE RISKY and IN PLAIN WORDS must end with "
    "either [S1]/[S2]/[S3] if the point comes from that source, or [CLAUSE] if it paraphrases "
    "the clause itself.\n"
    "4. If no source supports a point, do not make the point. You may say "
    "'No directly relevant source found' in RELEVANT LAW.\n"
    "5. Never invent section numbers or legal provisions.\n"
    "This is general information, not legal advice."
)

_SECTION_PAT = re.compile(r"(?:[Ss]ections?|[Rr]egulations?)\s+(\d+[A-Z]?)")
_ANCHOR_PAT = re.compile(r"\[(CLAUSE|S\d+)\]")


def build_prompt(clause, sources):
    clause = (clause or "").strip()
    if len(clause) > MAX_CLAUSE_CHARS:
        clause = clause[:MAX_CLAUSE_CHARS] + " ..."
    src = "\n".join(
        f"[S{i}] {s['act']} - {s['label']}: {s['text'][:MAX_SOURCE_CHARS]}"
        for i, s in enumerate(sources, 1)
    )
    return (
        f"CLAUSE:\n{clause}\n\nSOURCES:\n{src}\n\n"
        "Answer in EXACTLY this format. Every sentence in the first and third block must "
        "end with a real citation like [S1] or [S2] if the fact comes from a source, "
        "or [CLAUSE] if it paraphrases the input clause. Do not write [S#].\n\n"
        "WHY IT MAY BE RISKY: (1-2 sentences, each ending with [S1]/[S2]/[S3] or [CLAUSE])\n"
        "RELEVANT LAW: (list the source labels that truly apply, e.g. [S1], [S2]. "
        "If none apply, write 'No directly relevant source found')\n"
        "IN PLAIN WORDS: (2 sentences, each ending with [S1]/[S2]/[S3] or [CLAUSE])"
    )


def check_citations(answer, sources):
    """Three checks:
       1. every [S<n>] refers to a retrieved source;
       2. every 'Section N' the model mentions appears in the retrieved text;
       3. every substantive sentence in WHY / PLAIN blocks carries a marker."""
    cited = sorted({int(x) for x in re.findall(r"\[S(\d+)\]", answer)})
    invalid = [c for c in cited if c < 1 or c > len(sources)]

    retrieved_text = " ".join(f"{s['label']} {s['text']}" for s in sources)
    unsupported = sorted(
        set(_SECTION_PAT.findall(answer)) - set(_SECTION_PAT.findall(retrieved_text))
    )

    unanchored = []
    for block, stop_blocks in (
        ("WHY IT MAY BE RISKY:", ("RELEVANT LAW:", "IN PLAIN WORDS:")),
        ("IN PLAIN WORDS:", ()),
    ):
        if block not in answer:
            continue
        body = answer.split(block, 1)[1]
        for stop in stop_blocks:
            if stop in body:
                body = body.split(stop, 1)[0]
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith(("WHY IT", "IN PLAIN", "RELEVANT LAW")):
                continue
            if not _ANCHOR_PAT.search(line):
                unanchored.append(line[:120])

    placeholder = "[S#]" in answer
    citation_ok = (not invalid) and (not unsupported) and bool(cited) and not placeholder
    anchor_ok = not unanchored

    return {
        "cited": cited,
        "has_citation": bool(cited),
        "invalid_citations": invalid,
        "unsupported_sections": unsupported,
        "unanchored_sentences": unanchored,
        "placeholder_used": placeholder,
        "citation_ok": citation_ok,
        "anchor_ok": anchor_ok,
        "grounded_ok": citation_ok and anchor_ok,
    }


def explain_clause(clause, domain=None, k=3):
    sources = retrieve(clause, k=k, domain=domain)
    answer = ask_llm(
        SYSTEM_PROMPT,
        build_prompt(clause, sources),
        max_tokens=220,
        temperature=0.1,
        context_window=2048,
    )
    checks = check_citations(answer, sources)
    return {"answer": answer, "sources": sources, **checks}


# ---------------- setup check ----------------
def _model_names():
    resp = ollama.list()
    models = resp["models"] if isinstance(resp, dict) else getattr(resp, "models", [])
    names = []
    for m in models:
        try:
            name = m.get("model") or m.get("name")
        except AttributeError:
            name = getattr(m, "model", None) or getattr(m, "name", None)
        if name:
            names.append(name)
    return names


def verify_ollama_setup():
    try:
        names = _model_names()
    except Exception as e:
        print(f"Ollama not running: {e}\nStart it with: ollama serve")
        return False
    missing = [r for r in (OLLAMA_MODEL, OLLAMA_FALLBACK_MODEL)
               if not any(n == r or n.startswith(r) for n in names)]
    if missing:
        print(f"Missing models: {missing} | available: {names}")
        for m in missing:
            print(f"Run: ollama pull {m}")
        return False
    print(f"Ollama ready | primary: {OLLAMA_MODEL} | fallback: {OLLAMA_FALLBACK_MODEL}")
    return True


# ---------------- demo ----------------
SANITY = [
    ("consumer", "Your subscription renews automatically each month and can only be cancelled by "
                 "writing to our head office; no refund will be given for any reason."),
    ("financial", "A prepayment charge of 4 percent of the outstanding principal applies if the "
                  "Borrower closes the loan before the end of the tenure."),
    ("property", "If the Developer delays possession, the Allottee is entitled to Rs 5 per sq ft per month, "
                 "whereas the Allottee must pay interest at 24 percent per annum on any delayed instalment."),
    ("family", "The husband shall pay maintenance to the wife until the decree is passed, and the custody "
               "of the minor child will be decided by the court in the interest of the child."),
]

if __name__ == "__main__":
    kb = load_kb()
    print(f"Knowledge base: {len(kb['entries'])} entries")
    print("\n--- Retrieval test (no LLM needed) ---")
    for dom, clause in SANITY:
        print(f"\n[{dom}] {clause[:80]}...")
        print("  dense:")
        for s in retrieve(clause, k=3, domain=dom):
            print(f"    {s['score']:.3f}  {s['id']}")
        print("  tfidf:")
        for s in keyword_retrieve(clause, k=3, domain=dom):
            print(f"    {s['score']:.3f}  {s['id']}")
    if verify_ollama_setup():
        print("\nPrewarming models...")
        for line in prewarm_models():
            print("  " + line)
        t0 = time.time()
        res = explain_clause(SANITY[1][1], domain="financial")
        print("\n--- Explanation ---\n" + res["answer"])
        print({k: v for k, v in res.items() if k not in ("answer", "sources")})
        print(f"Took {time.time() - t0:.1f} s")