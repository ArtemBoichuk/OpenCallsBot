# utils.py
# ----------
# • Sentence-transformers model (multilingual MiniLM)
# • embed_text() returns a normalised vector for cosine search
# • load_deadlines() returns a list[dict]
# • load_missing_calls() for the fallback list we already had
# =================================================================================================================

import json, numpy as np, datetime
from sentence_transformers import SentenceTransformer

_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def embed_text(txt: str) -> np.ndarray:
    vec = _MODEL.encode([txt])[0]
    vec = vec / np.linalg.norm(vec)                # unit length  → cosine
    return vec.astype("float32").reshape(1, -1)

def load_deadlines(path="data/merged_deadlines.json"):
    """Loads raw JSON and always returns list of {code,deadline,status}."""
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        return []

    out = []
    today = datetime.date.today()
    for d in raw:
        code = d.get("code") or d.get("programme")
        dl   = d.get("deadline")
        if not code or not dl:
            continue

        status = d.get("status")
        if status not in ("OPEN", "CLOSED"):
            try:
                dt_obj = datetime.datetime.strptime(dl, "%d %b %Y").date()
                status = "OPEN" if dt_obj >= today else "CLOSED"
            except ValueError:
                status = "OPEN"

        out.append({"code":code, "deadline":dl, "status":status})
    return out

def load_missing_calls(path="missing_calls.json"):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {d["programme"].lower(): d for d in data}
    except FileNotFoundError:
        return {}
