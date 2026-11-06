"""
make_rag_eval_sheet.py
Builds a relevance sheet: for each human-verified RISKY clause, a pooled and shuffled list of
candidate statute entries. Annotators fill the 'relevant' column (1 = relevant, 0 = not).
Run from project root:  python make_rag_eval_sheet.py
"""
import os
import random
import pandas as pd
from modules.rag_pipeline import retrieve, keyword_retrieve

N_PER_DOMAIN, TOPN, SEED = 20, 6, 42
DOMAINS = ["consumer", "financial"]
OUT_DIR = os.path.join("datasets", "rag_eval")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join("datasets", "final", "final_dataset.csv"))
df["domain"] = df["domain"].astype(str).str.strip().str.lower()
df["is_gold"] = df["is_gold"].astype(str).str.lower().eq("true")
risky = df[df["is_gold"] & (df["final_label"] == "RISKY") & df["domain"].isin(DOMAINS)]

rows = []
for dom in DOMAINS:
    d = risky[risky["domain"] == dom]
    d = d.sample(n=min(N_PER_DOMAIN, len(d)), random_state=SEED)
    for _, r in d.iterrows():
        text = str(r["clause_text"])
        pool = {}
        for lst in (retrieve(text, TOPN), retrieve(text, TOPN, dom),
                    keyword_retrieve(text, TOPN), keyword_retrieve(text, TOPN, dom)):
            for e in lst:
                pool[e["id"]] = e
        cands = list(pool.values())
        random.Random(f"{SEED}-{r['clause_id']}").shuffle(cands)
        for e in cands:
            rows.append({"clause_id": r["clause_id"], "domain": dom, "clause_text": text,
                         "candidate_id": e["id"],
                         "candidate_text": f"{e['act']} - {e['label']}: {e['text']}",
                         "relevant": ""})

sheet = pd.DataFrame(rows)
path = os.path.join(OUT_DIR, "relevance_sheet.csv")
sheet.to_csv(path, index=False, encoding="utf-8-sig")
print(f"{sheet['clause_id'].nunique()} clauses, {len(sheet)} rows -> {path}")
print("Copy it as relevance_annotator_1.csv and relevance_annotator_2.csv (one per annotator).")