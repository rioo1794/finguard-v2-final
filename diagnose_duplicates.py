"""
FinGuard AI - Why are there so many duplicate clauses?
Run from the project root: python diagnose_duplicates.py
Read-only: it changes no files.
"""
import os

import pandas as pd

CLEAN_DIR = os.path.join("datasets", "clean")
norm = lambda t: " ".join(str(t).lower().split())

for d in ["consumer", "financial", "property", "family"]:
    path = os.path.join(CLEAN_DIR, f"{d}_clauses_CLEAN.csv")
    if not os.path.exists(path):
        print(f"\n{d}: {path} not found")
        continue

    df = pd.read_csv(path, engine="python", encoding_errors="replace")
    df["norm"] = df["clause_text"].map(norm)
    n, u = len(df), df["norm"].nunique()
    print(f"\n{'=' * 64}\n{d.upper()}: {n} rows -> {u} unique texts ({n - u} duplicates, {1 - u / n:.0%})\n{'=' * 64}")

    has_doc = "source_document" in df.columns
    agg = {"copies": ("clause_id", "size")}
    if has_doc:
        agg["docs"] = ("source_document", "nunique")
    g = df.groupby("norm").agg(**agg)
    dup = g[g["copies"] > 1]
    if has_doc:
        print(f"Duplicated texts: {len(dup)} | copies spread across different documents: "
              f"{(dup['docs'] > 1).sum()} | copies inside ONE document: {(dup['docs'] == 1).sum()}")
        per = df.groupby("source_document").agg(rows=("norm", "size"), unique=("norm", "nunique"))
        print("\nPer source document:")
        print(per.sort_values("rows", ascending=False).to_string())

    lens = df.drop_duplicates("norm")["clause_text"].str.split().str.len()
    print(f"\nUnique-clause length in words: median {lens.median():.0f}, "
          f"90th percentile {lens.quantile(.9):.0f}, over 150 words: {(lens > 150).sum()}")

    print("\nMost repeated texts:")
    for text, r in g.sort_values("copies", ascending=False).head(3).iterrows():
        print(f"  x{int(r['copies'])}: {text[:100]}...")