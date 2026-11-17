"""
PII masking (Data Privacy SOP §2).

Used at ingestion (before chunking / embedding: raw PII never reaches Pinecone or the LLM),
on retrieved chunks (defence in depth), on the question and on the LLM's answer.
"""

import re


def _mask_account(m):
    digits = m.group(m.lastindex)
    return (m.group(1) if m.lastindex > 1 else "") + "X" * (len(digits) - 4) + digits[-4:]


# (name, pattern, replacement) - applied in this order (cards before accounts, etc.)
PII_RULES = [
    ("card", re.compile(r"\b(\d{4})[ -]?(\d{2})\d{2}[ -]?\d{4}[ -]?(\d{4})\b"),
     lambda m: f"{m.group(1)} {m.group(2)}XX XXXX {m.group(3)}"),
    ("aadhaar", re.compile(r"\b\d{4}[ -]\d{4}[ -](\d{4})\b"),
     lambda m: f"XXXX XXXX {m.group(1)}"),
    ("aadhaar", re.compile(r"(?i)(aadha+r(?:\s+(?:number|no\.?))?\s*[:\-]?\s*)\d{8}(\d{4})\b"),
     lambda m: f"{m.group(1)}XXXX XXXX {m.group(2)}"),
    ("pan", re.compile(r"\b[A-Z]{5}(\d{4})[A-Z]\b"),
     lambda m: f"XXXXX{m.group(1)}X"),
    ("account", re.compile(r"(?i)((?:account\s*(?:number|no\.?)|a/c)\s*[:\-]?\s*)(\d{9,18})\b"), _mask_account),
    ("account", re.compile(r"\b(\d{13,18})\b"), _mask_account),
    ("mobile", re.compile(r"(\+91[ -]?)\d{6}(\d{4})\b"),
     lambda m: f"{m.group(1)}XXXXXX{m.group(2)}"),
    ("mobile", re.compile(r"(?i)(mobile(?:\s*(?:number|no\.?))?[^\d\n]{0,12})[6-9]\d{5}(\d{4})\b"),
     lambda m: f"{m.group(1)}XXXXXX{m.group(2)}"),
    ("email", re.compile(r"\b([A-Za-z0-9])[A-Za-z0-9._%+-]*@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)\b"),
     lambda m: f"{m.group(1)}***@{m.group(2)}"),
    ("dob", re.compile(r"(?i)(\b(?:DOB|date of birth)\b[^\d]{0,25})\d{4}-\d{2}-\d{2}"),
     lambda m: f"{m.group(1)}XXXX-XX-XX"),
]


def mask_pii(text):
    """Mask personal data. Returns (masked_text, {pii_type: count})."""
    counts = {}
    for name, pattern, repl in PII_RULES:
        def replace(m, name=name, repl=repl):
            new = repl(m)
            if new != m.group(0):          # already-masked values (e.g. XXXXX1234X) are not counted
                counts[name] = counts.get(name, 0) + 1
            return new
        text = pattern.sub(replace, text or "")
    return text, counts


def masked_total(counts):
    return sum(counts.values())
