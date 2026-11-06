"""
FinGuard AI - Automated Clause Labeling Pipeline
Rule-based patterns + Groq LLM classification, resume-safe.

Setup: put GROQ_API_KEY=your_key in your .env file (pip install python-dotenv).

Usage (run from the project root):
  Test run (PowerShell):  $env:LIMIT="50"; python auto_label_clauses.py
  Full run:               Remove-Item Env:LIMIT; python auto_label_clauses.py
"""

import json
import os
import re
import time

import groq
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()  # reads GROQ_API_KEY from your .env file
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# llama-3.3-70b-versatile was shut down by Groq on 16 Aug 2026.
# Override without editing code: add GROQ_MODEL=qwen/qwen3.6-27b to .env
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
BATCH_SIZE = 10
DELAY_BETWEEN_CALLS = 1.0
LIMIT = int(os.getenv("LIMIT", 0))  # 0 = all rows

VALID = {"PREDATORY", "COMPLIANT", "NEUTRAL", "AMBIGUOUS"}

CLEAN_DIR = os.path.join("datasets", "clean")
LABELED_DIR = os.path.join("datasets", "labeled")
os.makedirs(LABELED_DIR, exist_ok=True)

# ============================================================
# TIER 1: RULE-BASED PRE-LABELING (fast, free)
# ============================================================

PREDATORY_PATTERNS = {
    "unilateral_change": [
        r"may (change|modify|amend|vary|update) .{0,30}(terms|conditions|policies)",
        r"reserves? the right to (change|modify|amend|alter|update)",
        r"at (its|our|their) sole discretion",
        r"without (prior )?notice",
        r"from time to time entirely at",
    ],
    "liability_limitation": [
        r"(shall not|will not|not) be liable",
        r"disclaims? all (warranties|liability|responsibility)",
        r"no liability (whatsoever|for any)",
        r"maximum extent permitted by law",
        r"aggregate liability.{0,50}limited to",
    ],
    "indemnity_burden": [
        r"you (shall|will|agree to) indemnify",
        r"indemnify and hold harmless",
        r"release and fully indemnify",
        r"waive any claims",
    ],
    "arbitration_mandate": [
        r"shall be (resolved|settled) (exclusively )?by (arbitration|arbitrator)",
        r"binding arbitration",
        r"waive.{0,30}right to (a )?(jury|trial|court)",
    ],
    "data_overreach": [
        r"perpetual.{0,30}irrevocable.{0,30}(license|right)",
        r"royalty[- ]free.{0,30}(license|right)",
        r"share (your|the) (personal )?(information|data).{0,30}third part",
        r"without (your )?consent",
    ],
    "cancellation_penalty": [
        r"cancellation (charges?|fees?).{0,40}%",
        r"forfeit.{0,30}(advance|deposit|amount)",
        r"non[- ]refundable",
    ],
    "forced_compliance": [
        r"take it (as it is|or leave it)",
        r"no (changes|modifications).{0,30}(allowed|permitted)",
        r"one[- ]sided agreement",
    ],
}

COMPLIANT_PATTERNS = {
    "consumer_rights": [
        r"you (may|can|have the right to) (cancel|terminate|withdraw)",
        r"opt[- ](out|in)",
        r"refund.{0,30}(within|entitled)",
        r"cooling[- ]off period",
    ],
    "transparency": [
        r"(disclose|inform|notify).{0,30}(clearly|prominently|in writing)",
        r"prior (written )?notice",
        r"grievance (officer|redressal)",
        r"right to be (informed|heard)",
    ],
    "fair_practice": [
        r"fair (practice|treatment)",
        r"reasonable (notice|time|opportunity)",
        r"mutual(ly)? (agree|consent)",
    ],
    "compliance_mention": [
        r"in (accordance|compliance) with.{0,50}(act|law|regulation|rules)",
        r"(RBI|SEBI|IRDAI|Consumer Protection Act|IT Act|Companies Act)",
        r"as (amended|applicable) from time to time",
    ],
}

NEUTRAL_PATTERNS = {
    "definitions": [
        r"^[A-Z][a-z]+ .{0,20}(means|shall mean|refers to)",
        r"for the purposes of (this|these)",
    ],
    "boilerplate": [
        r"this (document|agreement) (is|constitutes) an electronic record",
        r"governing law",
        r"severability",
        r"entire agreement",
        r"waiver",
        r"contact us",
        r"grievance officer",
        r"registered office",
    ],
    "administrative": [
        r"(address|email|phone|contact)",
        r"(version|last updated|effective date)",
    ],
}


def rule_based_label(text: str) -> dict:
    """Regex pre-labeling. Returns label, risk_level, confidence."""
    text_lower = text.lower()
    scores = {"PREDATORY": 0, "COMPLIANT": 0, "NEUTRAL": 0}

    for label, pattern_dict in (
        ("PREDATORY", PREDATORY_PATTERNS),
        ("COMPLIANT", COMPLIANT_PATTERNS),
        ("NEUTRAL", NEUTRAL_PATTERNS),
    ):
        for patterns in pattern_dict.values():
            if any(re.search(p, text_lower, re.IGNORECASE) for p in patterns):
                scores[label] += 1  # count each category once

    max_score = max(scores.values())
    if max_score == 0:
        return {"label": "AMBIGUOUS", "risk_level": "NONE", "confidence": 0.3, "method": "rule"}

    label = max(scores, key=scores.get)
    confidence = max_score / sum(scores.values())

    if label == "PREDATORY":
        n = scores["PREDATORY"]
        risk = "HIGH" if n >= 3 else ("MEDIUM" if n == 2 else "LOW")
    elif label == "COMPLIANT":
        risk = "LOW"
    else:
        risk = "NONE"

    return {"label": label, "risk_level": risk, "confidence": round(confidence, 2), "method": "rule"}


# ============================================================
# TIER 2: LLM LABELING (accurate, slower)
# ============================================================

SYSTEM_PROMPT = """You are a legal clause classifier for Indian legal documents.
Classify each clause into ONE label:
- PREDATORY: unfair, one-sided or exploitative toward the consumer/user
- COMPLIANT: fair, balanced, protects user rights
- NEUTRAL: boilerplate, administrative or definitional, no risk
- AMBIGUOUS: unclear or needs legal interpretation

Also give risk_level (HIGH, MEDIUM, LOW, NONE), confidence (an INTEGER from 0 to 100, how sure you are; never use decimals)
and a reason (max 15 words).

Return a JSON object: {"results": [{"id": "...", "label": "...", "risk_level": "...",
"confidence": 85, "reason": "..."}]}
Include every ID given. No markdown, no extra text."""


def build_user_prompt(batch: list) -> str:
    items = [f'ID: {c["clause_id"]}\nTEXT: {str(c["clause_text"])[:1500]}' for c in batch]
    return "Classify these clauses:\n\n" + "\n\n---\n\n".join(items)


class DailyLimitReached(Exception):
    """Groq's tokens-per-day quota is used up; retrying today is pointless."""


def parse_wait(msg: str, default: float = 20.0) -> float:
    """Read 'try again in 4m24.38s' / '592ms' from a Groq rate-limit message."""
    m = re.search(r"try again in ([\dms.]+)", msg)
    if not m:
        return default
    units = {"ms": 0.001, "m": 60, "s": 1}
    total = sum(float(v) * units[u] for v, u in re.findall(r"([\d.]+)(ms|m|s)", m.group(1)))
    return total or default


def llm_label_batch(batch: list, retries: int = 3) -> list:
    attempt, rate_waits = 0, 0
    while attempt < retries:
        try:
            kwargs = dict(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(batch)},
                ],
                temperature=0.1,
                max_tokens=4000,  # reasoning models spend part of this on thinking
                response_format={"type": "json_object"},
            )
            if "gpt-oss" in MODEL:
                kwargs["reasoning_effort"] = "low"  # classification doesn't need deep reasoning
            resp = client.chat.completions.create(**kwargs)
            parsed = json.loads(resp.choices[0].message.content)
            if isinstance(parsed, dict):
                parsed = parsed.get(
                    "results", next((v for v in parsed.values() if isinstance(v, list)), [])
                )
            return parsed if isinstance(parsed, list) else []
        except (groq.NotFoundError, groq.AuthenticationError, groq.PermissionDeniedError) as e:
            # Retrying can't fix a dead model name or a bad key: stop immediately
            raise SystemExit(
                f"\nFatal API error: {e}\n"
                f"Model in use: {MODEL}. Check GROQ_API_KEY in .env, and the current model list "
                f"at console.groq.com/docs/models (set GROQ_MODEL in .env to change it)."
            )
        except groq.RateLimitError as e:
            msg = str(e)
            if "per day" in msg or "(TPD)" in msg:
                raise DailyLimitReached(msg)  # nothing more to do today
            rate_waits += 1  # per-minute limit: wait as long as Groq says, doesn't use up a retry
            if rate_waits > 8:
                return []
            wait = parse_wait(msg) + 1
            print(f"  Per-minute rate limit; waiting {wait:.0f}s")
            time.sleep(wait)
        except Exception as e:
            attempt += 1
            wait = 5 * attempt
            print(f"  LLM error ({str(e)[:160]}); retry in {wait}s")
            time.sleep(wait)
    return []


# ============================================================
# TIER 3: HYBRID PIPELINE
# ============================================================

def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)  # safe even if interrupted mid-write


def label_dataframe(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    print(f"\n{'=' * 60}\nLabeling {len(df)} clauses | domain: {domain}\n{'=' * 60}")

    # ---- Tier 1: rules ----
    rules = [rule_based_label(str(t)) for t in tqdm(df["clause_text"], desc="Rules")]
    df["rule_label"] = [r["label"] for r in rules]
    df["rule_risk"] = [r["risk_level"] for r in rules]
    df["rule_confidence"] = [r["confidence"] for r in rules]

    # ---- Tier 2: LLM (resume-safe cache, saved after every batch) ----
    cache_path = os.path.join(LABELED_DIR, f"{domain}_llm_cache.json")
    cache = load_cache(cache_path)
    todo = [c for c in df.to_dict("records") if c["clause_id"] not in cache]
    print(f"Model: {MODEL} | Cached: {len(cache)} | To label: {len(todo)}")

    for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="LLM"):
        try:
            results = llm_label_batch(todo[i:i + BATCH_SIZE])
        except DailyLimitReached:
            save_cache(cache, cache_path)
            raise SystemExit(
                f"\nDaily token limit reached. Progress is saved ({len(cache)} {domain} clauses labeled). "
                f"Run the script again later (the quota refills on a rolling 24h window) and it resumes "
                f"where it stopped. No output file was written for this domain yet."
            )
        for r in results:
            if isinstance(r, dict) and r.get("id") and r.get("label") in VALID:
                cache[r["id"]] = r
        save_cache(cache, cache_path)
        time.sleep(DELAY_BETWEEN_CALLS)

    # ---- Tier 3: merge ----
    labels, risks, confs, reasons, sources = [], [], [], [], []
    for _, row in df.iterrows():
        r = cache.get(row["clause_id"])
        if r:
            labels.append(r["label"])
            risks.append(r.get("risk_level", "NONE"))
            try:
                c = float(r.get("confidence", 50))
                confs.append(round(c / 100 if c > 1 else c, 3))  # 85 -> 0.85; old 0.85 stays
            except (TypeError, ValueError):
                confs.append(0.5)
            reasons.append(r.get("reason", ""))
            sources.append("llm")
        else:  # fallback: confidence capped low so it always goes to human review
            labels.append(row["rule_label"])
            risks.append(row["rule_risk"])
            confs.append(min(float(row["rule_confidence"]), 0.5))
            reasons.append("Rule-based fallback")
            sources.append("rule")

    df["llm_label"] = labels
    df["label"] = labels
    df["risk_level"] = risks
    df["llm_confidence"] = confs
    df["reason"] = reasons
    df["label_source"] = sources
    df["is_predatory"] = df["label"].map(lambda x: "YES" if x == "PREDATORY" else "NO")
    df["final_label"] = df["label"]  # overwritten after human verification
    df["notes"] = df["reason"]
    # annotator_1-4 columns are left untouched for the humans

    llm_share = (df["label_source"] == "llm").mean()
    if llm_share < 0.9:
        print(f"\n!! WARNING: only {llm_share:.0%} of rows were labeled by the LLM. "
              f"The rest are rule-based fallbacks. Re-run to retry the missing ones.")

    print("\nLabel distribution:\n" + df["label"].value_counts().to_string())
    print(f"Low-confidence (<0.6): {(df['llm_confidence'] < 0.6).sum()}")
    print(f"Rule-fallback rows: {(df['label_source'] == 'rule').sum()}")
    return df


def normalize(t) -> str:
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def main():
    for domain in ["consumer", "financial", "property", "family"]:
        path = os.path.join(CLEAN_DIR, f"{domain}_clauses_CLEAN.csv")
        if not os.path.exists(path):
            print(f"Skipping {path} - not found (run clean_csvs.py first)")
            continue

        df = pd.read_csv(path, engine="python", encoding_errors="replace", on_bad_lines="warn")
        before = len(df)

        df = df.dropna(subset=["clause_id", "clause_text"])
        df = df.drop_duplicates(subset="clause_id")
        df = df[~df["clause_text"].map(normalize).duplicated()]  # exact-text duplicates
        print(f"\n{path}: {before} rows -> {len(df)} after dedup")

        if LIMIT:
            df = df.head(LIMIT)

        out = label_dataframe(df.copy(), domain)
        out_path = os.path.join(LABELED_DIR, f"{domain}_clauses_CLEAN_AUTO_LABELED.csv")
        out.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()