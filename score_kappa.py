"""
FinGuard AI - Kappa scoring for 3 annotators (hybrid plan).

Inputs (datasets/verification/):
  verification_key.csv
  annotator_1_done.csv, annotator_2_done.csv, annotator_3_done.csv  (columns: clause_id, label, ...)
  adjudicated.csv  (optional; columns: clause_id, consensus) - only for 3-way splits

Consensus:
  * overlap rows -> majority of the 3 labels (2 of 3 is enough)
  * single rows  -> the one annotator's label
  * overlap rows where all 3 disagree -> disagreements.csv, team resolves them

Reports:
  * Human IAA on the overlap: Fleiss' kappa (+ pairwise Cohen's kappa)
  * LLM vs human consensus on the random sample: the headline kappa
  * LLM agreement on the low-confidence rows (report separately)

Also writes human_consensus.csv (clause_id, domain, consensus) for updating final_label.
"""
import itertools
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

VALID = ["PREDATORY", "COMPLIANT", "NEUTRAL", "AMBIGUOUS"]
H = ["h1", "h2", "h3"]
VERIFY_DIR = os.path.join("datasets", "verification")
P = lambda name: os.path.join(VERIFY_DIR, name)


def clean_label(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    return x if x in VALID else np.nan


def kappa(a, b):
    if len(a) < 2:
        return None
    return round(cohen_kappa_score(a, b, labels=VALID), 3)


def fleiss_kappa(ratings: pd.DataFrame):
    """ratings: N rows x 3 columns of labels (no missing values)."""
    n = ratings.shape[1]
    counts = np.array([(ratings == c).sum(axis=1).to_numpy() for c in VALID]).T  # N x k
    N = counts.shape[0]
    P_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    p_j = counts.sum(axis=0) / (N * n)
    P_e = (p_j ** 2).sum()
    return None if P_e == 1 else round((P_bar - P_e) / (1 - P_e), 3)


# ---------- load ----------
key = pd.read_csv(P("verification_key.csv"))
df = key.copy()

for i in (1, 2, 3):
    a = pd.read_csv(P(f"annotator_{i}_done.csv"))[["clause_id", "label"]]
    cleaned = a["label"].map(clean_label)
    n_bad = (a["label"].notna() & cleaned.isna() & (a["label"].astype(str).str.strip() != "")).sum()
    if n_bad:
        print(f"WARNING annotator {i}: {n_bad} rows have a label outside {VALID} (ignored)")
    a[f"h{i}"] = cleaned
    df = df.merge(a[["clause_id", f"h{i}"]], on="clause_id", how="left")

# ---------- split into overlap / single and drop incomplete rows ----------
overlap = df[df["group"] == "overlap"].copy()
single = df[df["group"] == "single"].copy()

miss_o = overlap[H].isna().any(axis=1)
single["own"] = single.apply(lambda r: r[f"h{int(r['assigned_to'])}"], axis=1)
miss_s = single["own"].isna()
if miss_o.sum() or miss_s.sum():
    print(f"WARNING: skipping {miss_o.sum()} incomplete overlap rows and {miss_s.sum()} unlabeled single rows")
overlap, single = overlap[~miss_o].copy(), single[~miss_s].copy()


# ---------- consensus ----------
def majority(row):
    label, cnt = Counter(row[H]).most_common(1)[0]
    return label if cnt >= 2 else np.nan


overlap["consensus"] = overlap.apply(majority, axis=1)
single["consensus"] = single["own"]
all_rows = pd.concat([overlap, single], ignore_index=True)

# 3-way splits need a human decision
split = all_rows[all_rows["consensus"].isna()]
if os.path.exists(P("adjudicated.csv")):
    adj = pd.read_csv(P("adjudicated.csv")).set_index("clause_id")["consensus"]
    adj = adj.map(clean_label)
    all_rows["consensus"] = all_rows["consensus"].fillna(all_rows["clause_id"].map(adj))
elif len(split):
    split[["clause_id", "clause_text", "h1", "h2", "h3"]].assign(consensus="").to_csv(
        P("disagreements.csv"), index=False
    )
    print(f"{len(split)} rows where all 3 annotators disagree -> {P('disagreements.csv')}")
    print("Fill the 'consensus' column, save as adjudicated.csv, then re-run.\n")

# ---------- human agreement (overlap only) ----------
print(f"Overlap rows (all 3 labeled): {len(overlap)} | single-annotator rows: {len(single)}")
if len(overlap):
    print("\nHuman inter-annotator agreement (overlap):")
    print(f"  Fleiss' kappa (3 raters): {fleiss_kappa(overlap[H])}")
    for x, y in itertools.combinations(H, 2):
        print(f"  Cohen's kappa {x} vs {y}: {kappa(overlap[x], overlap[y])}")
    full = (overlap[H].nunique(axis=1) == 1).mean()
    print(f"  All 3 agree on {full:.0%} of rows | 3-way splits: {len(split)}")

# ---------- LLM vs human consensus ----------
done = all_rows.dropna(subset=["consensus"])
rand = done[done["sample_type"] == "random"]

print("\nLLM vs human consensus (random sample only - the headline number):")
for d, g in rand.groupby("domain"):
    print(f"  {d:10s} n={len(g):4d}  kappa={kappa(g['llm_label'], g['consensus'])}")
print(f"  {'OVERALL':10s} n={len(rand):4d}  kappa={kappa(rand['llm_label'], rand['consensus'])}")

low = done[done["sample_type"] == "low_conf"]
if len(low):
    agree = (low["llm_label"] == low["consensus"]).mean()
    print(f"\nLow-confidence rows (n={len(low)}): LLM agreed with humans on {agree:.0%}")
    print("Report this separately; these are the hard cases, so don't mix into the headline kappa.")

done[["clause_id", "domain", "consensus"]].to_csv(P("human_consensus.csv"), index=False)
print(f"\nSaved {len(done)} verified labels -> {P('human_consensus.csv')}")