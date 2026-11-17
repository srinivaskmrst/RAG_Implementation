"""Small helpers shared by several modules."""

import hashlib
import json
import re

STOPWORDS = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were",
             "what", "which", "how", "many", "much", "does", "do", "did", "can", "i", "my", "me",
             "it", "this", "that", "by", "with", "at", "as", "from", "per", "get", "got"}


def sha256(data):
    """SHA-256 hex digest of bytes or text (document hash / chunk hash)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def tokenize(text):
    """Lower-case words without stop words (for BM25)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def years_in(text):
    """Years named in a question, e.g. {2019}. Also finds the year in 'HR-LV-2022'."""
    return {int(y) for y in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text)}


def parse_json(text, default):
    """Parse an LLM's JSON answer, tolerating stray prose or code fences around it."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"[\[{].*[\]}]", text or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return default


def matches_filter(meta, where):
    """Does a chunk / metadata dict satisfy a Pinecone-style filter ({"key": {"$eq"|"$in": ...}})?
    Used to apply the same access filter to the in-memory BM25 search as to the Pinecone query."""
    for key, cond in (where or {}).items():
        value = meta.get(key)
        if isinstance(cond, dict):
            if "$eq" in cond and value != cond["$eq"]:
                return False
            if "$in" in cond and value not in cond["$in"]:
                return False
            if "$ne" in cond and value == cond["$ne"]:
                return False
        elif value != cond:
            return False
    return True
