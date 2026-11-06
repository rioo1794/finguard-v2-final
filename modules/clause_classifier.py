"""
modules/clause_classifier.py
Loads the saved BERT models (one per domain) and predicts RISKY / NOT_RISKY for clauses.
Domains without a trained model (property, family) return None so the caller can use its old method.
"""
import os
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
TRAINED_DOMAINS = {"consumer", "financial"}
MAX_LEN = 256
THRESHOLD = 0.5          # probability above which a clause is called RISKY

_cache = {}
_lock = threading.Lock()


def _folder(domain):
    return os.path.join(MODEL_DIR, f"bert_{domain}")


def has_model(domain):
    domain = (domain or "").lower()
    return domain in TRAINED_DOMAINS and os.path.isdir(_folder(domain))


def _load(domain):
    with _lock:
        if domain not in _cache:
            tok = AutoTokenizer.from_pretrained(_folder(domain))
            model = AutoModelForSequenceClassification.from_pretrained(_folder(domain)).eval()
            _cache[domain] = (tok, model)
        return _cache[domain]


def predict_clauses(texts, domain, batch_size=16):
    """Returns a list of {label, p_risky} dicts, or None if there is no trained model for the domain."""
    domain = (domain or "").lower()
    if not has_model(domain):
        return None
    tok, model = _load(domain)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc = tok(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist()
        out += [{"label": "RISKY" if x >= THRESHOLD else "NOT_RISKY", "p_risky": round(x, 3)} for x in p]
    return out


def predict_clause(text, domain):
    r = predict_clauses([text], domain)
    return r[0] if r else None