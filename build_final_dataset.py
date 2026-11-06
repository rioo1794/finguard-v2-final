"""
build_final_dataset.py  (binary version)
Labels: RISKY / NOT_RISKY / UNSURE (UNSURE rows are excluded from training and scoring).
Silver (LLM) mapping: PREDATORY->RISKY, COMPLIANT/NEUTRAL->NOT_RISKY, AMBIGUOUS->UNSURE
Gold = human consensus from human_consensus_binary.csv (written by binary_kappa.py)

Run from project root:  python build_final_dataset.py
Reads : datasets/labeled/{domain}_clauses_CLEAN_AUTO_LABELED.csv
        datasets/verification/human_consensus_binary.csv
        datasets/verification/verification_key.csv
Writes: datasets/final/final_dataset.csv and datasets/final/{domain}_final.csv
Does NOT modify any existing file.
"""
import os
import re
import random
import numpy as np
import pandas as pd

DOMAINS = ["consumer", "financial", "property", "family"]
MAP = {"PREDATORY": "RISKY", "COMPLIANT": "NOT_RISKY", "NEUTRAL": "NOT_RISKY",
       "AMBIGUOUS": "UNSURE", "RISKY": "RISKY", "NOT_RISKY": "NOT_RISKY", "UNSURE": "UNSURE"}
LABELED_DIR = os.path.join("datasets", "labeled")
VERIF_DIR = os.path.join("datasets", "verification")
CONSENSUS_PATH = os.path.join(VERIF_DIR, "human_consensus_binary.csv")
KEY_PATH = os.path.join(VERIF_DIR, "verification_key.csv")
OUT_DIR = os.path.join("datasets", "final")
SEEDS = [42, 43, 44]
N_FOLDS = 5

# doc_type: guessed from the document name. CHECK the printed list and fix with overrides.
JUDGMENT_RE = re.compile(r"\bvs?\b\.?|state of|high court|supreme court|ncdrc|tribunal|judg", re.I)
DOC_TYPE_OVERRIDES = {
    "RBI Circular - PSS Act Monetary Penalty & Compounding Framework": "regulation",
    "RBI DOR Credit Recommendation 5": "regulation",
    "RBI DOR Financial Recommendation 289": "regulation",
    "RBI Master Direction 5": "regulation",
    "RBI Master Direction 6": "regulation",
    "Redacted Parties Case (21 April 2026)": "judgment",
    "Redacted Parties Case (24 March 2026)": "judgment",
}


def to_bin(x):
    return MAP.get(str(x).strip().upper()) if pd.notna(x) else None


def doc_type(name):
    if name in DOC_TYPE_OVERRIDES:
        return DOC_TYPE_OVERRIDES[name]
    return "judgment" if JUDGMENT_RE.search(str(name)) else "contract_or_regulation"


def read_any(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def assign_group_folds(df, seed, n_folds=N_FOLDS):
    """Whole documents go to one fold (no leakage). Largest docs first, each into the
    currently smallest fold. Seed only shuffles tie order."""
    rng = random.Random(seed)
    sizes = df.groupby("source_document").size().to_dict()
    docs = list(sizes.keys())
    rng.shuffle(docs)
    docs.sort(key=lambda d: -sizes[d])
    fold_size = [0] * n_folds
    doc_fold = {}
    for d in docs:
        f = min(range(n_folds), key=lambda i: fold_size[i])
        doc_fold[d] = f
        fold_size[f] += sizes[d]
    return df["source_document"].map(doc_fold)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cons = read_any(CONSENSUS_PATH)
    cons["consensus_binary"] = cons["consensus_binary"].map(to_bin)
    cons = cons.dropna(subset=["consensus_binary"]).drop_duplicates("clause_id")
    gold_map = dict(zip(cons["clause_id"], cons["consensus_binary"]))

    key = read_any(KEY_PATH).drop_duplicates("clause_id")
    type_map = dict(zip(key["clause_id"], key["sample_type"].astype(str).str.lower()))
    group_map = dict(zip(key["clause_id"], key["group"].astype(str).str.lower()))

    frames = []
    for dom in DOMAINS:
        path = os.path.join(LABELED_DIR, f"{dom}_clauses_CLEAN_AUTO_LABELED.csv")
        if not os.path.exists(path):
            print(f"SKIP {dom}: {path} not found")
            continue
        df = pd.read_csv(path)
        df["domain"] = dom                                   # FIX: always set (blank domains were dropping rows)

        df["silver_label_4"] = df["label"]                   # original 4-label LLM output, kept for reference
        df["silver_label"] = df["label"].map(to_bin)
        df["gold_label"] = df["clause_id"].map(gold_map)     # RISKY / NOT_RISKY / UNSURE / NaN
        df["has_human"] = df["gold_label"].notna()
        df["final_label"] = df["gold_label"].where(df["has_human"], df["silver_label"])
        df["final_label_source"] = np.where(df["has_human"], "human", "llm_silver")
        df["usable"] = df["final_label"].isin(["RISKY", "NOT_RISKY"])
        df["is_gold"] = df["has_human"] & df["usable"]
        df["is_risky"] = (df["final_label"] == "RISKY").astype(int)
        df["is_predatory"] = df["is_risky"]                  # kept for compatibility with older code

        df["sample_type"] = df["clause_id"].map(type_map)
        df["verif_group"] = df["clause_id"].map(group_map)
        df["is_random_sample"] = df["sample_type"].fillna("").str.contains("random")
        df["is_overlap"] = df["verif_group"].fillna("").str.contains("overlap")
        df["is_eval"] = df["is_gold"] & df["is_random_sample"]   # headline evaluation set
        df["doc_type"] = df["source_document"].map(doc_type)

        for s in SEEDS:
            df[f"fold_seed{s}"] = assign_group_folds(df, s)

        df.to_csv(os.path.join(OUT_DIR, f"{dom}_final.csv"), index=False)
        frames.append(df)

        print(f"\n=== {dom.upper()} ===")
        print(f"rows: {len(df)} | usable: {int(df['usable'].sum())} | documents: {df['source_document'].nunique()}")
        print("final labels:", df["final_label"].value_counts(dropna=False).to_dict())
        ev = df[df["is_eval"]]
        print(f"eval rows: {len(ev)} | eval RISKY: {int(ev['is_risky'].sum())} | "
              f"human UNSURE (excluded): {int((df['has_human'] & ~df['usable']).sum())}")
        docs = df.groupby("source_document").agg(rows=("clause_id", "size"), doc_type=("doc_type", "first"))
        print("documents and guessed doc_type (fix wrong ones in DOC_TYPE_OVERRIDES):")
        print(docs.to_string())
        fc = df.groupby("fold_seed42").agg(rows=("clause_id", "size"), eval=("is_eval", "sum"),
                                           eval_risky=("is_risky", lambda s: int(s[df.loc[s.index, "is_eval"]].sum())))
        print("folds (seed 42):")
        print(fc.to_string())
        if (fc["eval_risky"] == 0).any():
            print("  WARNING: some folds have 0 RISKY eval rows (fine if results are pooled across folds).")

    alldf = pd.concat(frames, ignore_index=True)
    if alldf["domain"].isna().any():
        print("WARNING: some rows still have no domain.")
    alldf.to_csv(os.path.join(OUT_DIR, "final_dataset.csv"), index=False)
    print(f"\nSaved {len(alldf)} rows -> {os.path.join(OUT_DIR, 'final_dataset.csv')}")
    print(f"Usable: {int(alldf['usable'].sum())} | Gold: {int(alldf['is_gold'].sum())} | Eval: {int(alldf['is_eval'].sum())}")


if __name__ == "__main__":
    main()