"""
Threat detection: patterns for questions (and document chunks) that must be blocked.

    find_injection(text)   prompt injection / jailbreak ("ignore previous instructions", ...)
    SECRET_RE              asks for OTP / PIN / CVV / password
    UNMASK_RE              asks for unmasked / full personal data
    BULK_RE                asks for many customers' personal data at once
    names_in(text, names)  which customers are named in a text (customer data isolation)
"""

import re

INJECTION_PATTERNS = [
    r"\bignore\b[^.\n]{0,30}\b(instructions?|rules|prompts?|guardrails?|polic(y|ies))\b",
    r"\bdisregard\b[^.\n]{0,30}\b(instructions?|rules|guardrails?|polic(y|ies)|above)\b",
    r"\bforget\b[^.\n]{0,30}\b(instructions?|rules|guardrails?)\b",
    r"\b(reveal|show|print|repeat|output)\b[^.\n]{0,20}\b(system|hidden|initial)\s+(prompt|instructions?|message)\b",
    r"\byou are now\b", r"\bfrom now on,? you\b", r"\bpretend (to be|you are|that you)\b",
    r"\bact as\b[^.\n]{0,20}\b(admin|administrator|developer|root|unrestricted|unfiltered|dan)\b",
    r"\b(jailbreak|developer mode|do anything now|god mode)\b",
    r"\b(override|bypass|disable|turn off)\b[^.\n]{0,20}\b(guardrails?|safety|security|filters?|masking|restrictions?)\b",
    r"<\s*/?\s*(system|instructions?)\s*>", r"\bnew (system )?instructions?\s*:",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)

SECRET_RE = re.compile(r"\b(cvv|cvc|otp|one[- ]time password|upi pin|atm pin|m-?pin|pin number|"
                       r"passwords?|passcodes?|net ?banking (login|credentials|password)|login credentials)\b", re.I)

_PII_WORDS = (r"(pan|aadhaa?r|account\s*(numbers?|nos?\.?)|a/c|card\s*numbers?|mobile|phone|"
              r"e-?mails?|dob|date of birth|address(es)?)")
UNMASK_RE = re.compile(r"\b(unmask(ed)?|unredact(ed)?|full|complete|entire|raw|actual|real|original|"
                       r"without (masking|redaction|x))\b[^.?!\n]{0,40}\b" + _PII_WORDS + r"\b"
                       r"|\b" + _PII_WORDS + r"\b[^.?!\n]{0,30}\b(unmasked|unredacted|in full|without masking)\b", re.I)
BULK_RE = re.compile(r"\b(all|every|each|list|dump|export|download)\b[^.?!\n]{0,40}"
                     r"\b(customers?|clients?|account holders?|applicants?)\b[^.?!\n]{0,40}\b" + _PII_WORDS + r"\b"
                     r"|\b(all|every|list|dump|export)\b[^.?!\n]{0,15}\b" + _PII_WORDS + r"\b[^.?!\n]{0,40}"
                     r"\b(customers?|clients?|account holders?|applicants?)\b", re.I)


def find_injection(text):
    """The matched prompt-injection text, or None."""
    m = INJECTION_RE.search(text or "")
    return m.group(0) if m else None


def name_parts(name):
    return [name] + [p for p in re.split(r"\s+", name) if len(p) >= 3]


def names_in(text, customers):
    """Customers whose full, first or last name appears in the text as a whole word."""
    return {name for name in customers
            if any(re.search(rf"\b{re.escape(p)}\b", text, re.I) for p in name_parts(name))}
