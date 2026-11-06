"""
binary_kappa.py
Collapses the 4-label scheme to binary and recomputes agreement.
  PREDATORY -> RISKY | COMPLIANT, NEUTRAL -> NOT_RISKY | AMBIGUOUS -> UNSURE (excluded)
Reads : datasets/verification/annotator_*_done.csv, verification_key.csv,
        human_consensus.csv (fallback for ties)
Writes: datasets/verification/human_consensus_binary.csv
Does NOT modify any existing file.
"""
import glob
import itertools
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score

VERIF = os.path.join("datasets", "verification")
MAP = {"PREDATORY": "RISKY", "COMPLIANT": "NOT_RISKY", "NEUTRAL": "NOT_RISKY",
       "AMBIGUOUS": "UNSURE", "RISKY": "RISKY", "NOT_RISKY": "NOT_RISKY", "UNSURE": "UNSURE"}


def to_bin(x):
    return MAP.get(str(x).strip().upper()) if pd.notna(x) else None


def read_any(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def find_label_col(df):
    cands = []
    for c in df.columns:
        if c.lower() in ("clause_id", "clause_text"):
            continue
        vals = df[c].dropna().astype(str).str.strip().str.upper()
        if len(vals) and vals.isin(MAP.keys()).mean() >= 0.8:
            cands.append(c)
    if not cands:
        return None
    for c in cands:  # prefer an obvious name
        if any(k in c.lower() for k in ("label", "annot", "human")):
            return c
    return cands[0]


def fleiss(rows, cats):
    """rows: list of label lists (same length). Returns Fleiss' kappa."""
    n = len(rows[0]); N = len(rows)
    counts = [[r.count(c) for c in cats] for r in rows]
    p_j = [sum(row[j] for row in counts) / (N * n) for j in range(len(cats))]
    P_i = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in counts]
    P_bar = sum(P_i) / N
    P_e = sum(p * p for p in p_j)
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else float("nan")


def main():
    # ---- load annotator sheets
    sheets = {}
    for p in sorted(glob.glob(os.path.join(VERIF, "annotator_*_done.csv"))):
        df = read_any(p)
        col = find_label_col(df)
        name = os.path.basename(p)
        if col is None:
            print(f"!! {name}: could not find a label column. Columns: {list(df.columns)}")
            continue
        s = df.dropna(subset=[col]).drop_duplicates("clause_id").set_index("clause_id")[col].map(to_bin)
        sheets[name] = s
        print(f"{name}: label column = '{col}', rows labeled = {len(s)}, "
              f"{s.value_counts(dropna=False).to_dict()}")
    if len(sheets) < 2:
        print("Need at least 2 annotator sheets. Check the messages above."); return

    key = read_any(os.path.join(VERIF, "verification_key.csv")).drop_duplicates("clause_id")
    key["domain"] = key["domain"].astype(str).str.lower()
    dom_map = dict(zip(key["clause_id"], key["domain"]))

    old = read_any(os.path.join(VERIF, "human_consensus.csv"))
    old["consensus"] = old["consensus"].map(to_bin)
    old_map = dict(zip(old["clause_id"], old["consensus"]))

    # ---- overlap = clauses labeled by every annotator
    names = list(sheets)
    overlap = set.intersection(*[set(s.index) for s in sheets.values()])
    print(f"\nOverlap rows (in all {len(names)} sheets): {len(overlap)}")

    # agreement on overlap, UNSURE excluded
    ids = sorted(overlap)
    table = pd.DataFrame({n: sheets[n].reindex(ids) for n in names})
    clean = table[(table != "UNSURE").all(axis=1) & table.notna().all(axis=1)]
    print(f"Rows with no UNSURE from anyone: {len(clean)} (dropped {len(table) - len(clean)})")
    if len(clean) > 1:
        print("\nHuman agreement, BINARY, overlap:")
        for a, b in itertools.combinations(names, 2):
            print(f"  Cohen's kappa {a[:13]} vs {b[:13]}: {cohen_kappa_score(clean[a], clean[b]):.3f}")
        if len(names) >= 3:
            rows = clean[names].values.tolist()
            print(f"  Fleiss' kappa ({len(names)} raters): {fleiss(rows, ['RISKY', 'NOT_RISKY']):.3f}")
        allagree = (clean.nunique(axis=1) == 1).mean()
        print(f"  All annotators agree on {allagree:.0%} of rows")
        print(f"  RISKY share in overlap (per annotator): "
              f"{ {n: round((clean[n] == 'RISKY').mean(), 2) for n in names} }")
        print("  NOTE: kappa is low when one class is rare, even with good agreement.")

    # ---- binary consensus for all rows
    out = []
    for cid in sorted(set().union(*[set(s.index) for s in sheets.values()])):
        labs = [s[cid] for s in sheets.values() if cid in s.index and s[cid] != "UNSURE" and pd.notna(s[cid])]
        if len(labs) == 0:
            cons = "UNSURE"
        elif len(labs) == 1:
            cons = labs[0]
        else:
            r, n_ = labs.count("RISKY"), labs.count("NOT_RISKY")
            cons = "RISKY" if r > n_ else "NOT_RISKY" if n_ > r else (old_map.get(cid) or "UNSURE")
        out.append({"clause_id": cid, "domain": dom_map.get(cid), "consensus_binary": cons})
    cons_df = pd.DataFrame(out)
    cons_df.to_csv(os.path.join(VERIF, "human_consensus_binary.csv"), index=False)
    print(f"\nSaved {len(cons_df)} rows -> {os.path.join(VERIF, 'human_consensus_binary.csv')}")
    print("consensus counts:", cons_df["consensus_binary"].value_counts().to_dict())

    # ---- LLM vs human consensus, random sample only
    kr = key[key["sample_type"].astype(str).str.lower().str.contains("random")].copy()
    kr["llm_bin"] = kr["llm_label"].map(to_bin)
    m = kr.merge(cons_df[["clause_id", "consensus_binary"]], on="clause_id")
    m = m[(m["llm_bin"] != "UNSURE") & (m["consensus_binary"] != "UNSURE")
          & m["llm_bin"].notna() & m["consensus_binary"].notna()]
    print("\nLLM vs human consensus, BINARY, random sample only (UNSURE excluded):")
    for dom, g in m.groupby("domain"):
        print(f"  {dom:10s} n={len(g):4d}  kappa={cohen_kappa_score(g['llm_bin'], g['consensus_binary']):.3f}")
    print(f"  {'OVERALL':10s} n={len(m):4d}  kappa={cohen_kappa_score(m['llm_bin'], m['consensus_binary']):.3f}")


if __name__ == "__main__":
    main()