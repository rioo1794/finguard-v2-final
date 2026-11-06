"""
FinGuard AI - Build the human verification sample (3 annotators, hybrid plan).

Plan:
  * OVERLAP rows (default 50 per domain = 200): ALL 3 annotators label them
    -> used for human inter-annotator agreement (Fleiss' kappa) + majority consensus.
  * The remaining rows are split 3 ways (balanced by domain + sample type);
    each annotator labels only their own third.

Annotators never see the LLM label, and overlap rows are shuffled in with
their own rows so they can't tell which ones are shared.

Creates (in datasets/verification/):
  verification_key.csv     -> PRIVATE (LLM labels + who got what)
  annotator_1_sheet.csv, annotator_2_sheet.csv, annotator_3_sheet.csv
"""
import os

import pandas as pd

SEED = 42
FRAC = 0.15                # random sample = 15% per domain...
MIN_N = 200                # ...but at least 200 per domain
LOW_CONF = 0.6             # every row below this confidence is also verified
OVERLAP_PER_DOMAIN = 50    # rows all 3 annotators label (taken from the random sample)
N_ANNOT = 3
DOMAINS = ["consumer", "financial", "property", "family"]
LABELS = "PREDATORY / COMPLIANT / NEUTRAL / AMBIGUOUS"

LABELED_DIR = os.path.join("datasets", "labeled")
VERIFY_DIR = os.path.join("datasets", "verification")
os.makedirs(VERIFY_DIR, exist_ok=True)

# ---------- 1. random sample + low-confidence rows ----------
parts = []
for d in DOMAINS:
    df = pd.read_csv(
        os.path.join(LABELED_DIR, f"{d}_clauses_CLEAN_AUTO_LABELED.csv"),
        engine="python", encoding_errors="replace",
    )
    df["domain"] = d

    n = min(len(df), max(MIN_N, int(len(df) * FRAC)))
    rand = df.sample(n=n, random_state=SEED).assign(sample_type="random")

    low = df[(df["llm_confidence"] < LOW_CONF) & ~df["clause_id"].isin(rand["clause_id"])]
    parts += [rand, low.assign(sample_type="low_conf")]
    print(f"{d}: {len(rand)} random + {len(low)} low-confidence")

key = pd.concat(parts, ignore_index=True)
assert key["clause_id"].is_unique, "clause_id must be unique across all domains"

# ---------- 2. choose overlap rows (random sample only, stratified by domain) ----------
overlap_ids = []
for d, g in key[key["sample_type"] == "random"].groupby("domain"):
    overlap_ids += g.sample(n=min(OVERLAP_PER_DOMAIN, len(g)), random_state=SEED)["clause_id"].tolist()

key["group"] = key["clause_id"].isin(overlap_ids).map({True: "overlap", False: "single"})

# ---------- 3. split the rest evenly across annotators ----------
key["assigned_to"] = 0   # 0 = everyone (overlap)
single = key["group"] == "single"
shuffled = key[single].sample(frac=1, random_state=SEED)
key.loc[single, "assigned_to"] = shuffled.groupby(["domain", "sample_type"]).cumcount() % N_ANNOT + 1

# ---------- 4. save key + blind sheets ----------
key = key.sample(frac=1, random_state=SEED)
key[["clause_id", "domain", "clause_text", "llm_label", "llm_confidence",
     "sample_type", "group", "assigned_to"]].to_csv(
    os.path.join(VERIFY_DIR, "verification_key.csv"), index=False
)

print(f"\nOverlap (all {N_ANNOT} label): {(key['group'] == 'overlap').sum()}")
for i in range(1, N_ANNOT + 1):
    rows = key[(key["group"] == "overlap") | (key["assigned_to"] == i)]
    rows = rows.sample(frac=1, random_state=SEED + i)   # different order per annotator
    sheet = rows[["clause_id", "domain", "clause_text"]].copy()
    sheet["label"] = ""   # annotator fills this in
    sheet.to_csv(os.path.join(VERIFY_DIR, f"annotator_{i}_sheet.csv"), index=False)
    print(f"annotator_{i}_sheet.csv: {len(sheet)} rows")

print(f"\nSend each annotator ONLY their own sheet from {VERIFY_DIR}.")
print(f"Allowed labels: {LABELS}")
print("Tell them to save the finished file as annotator_N_done.csv (same columns).")