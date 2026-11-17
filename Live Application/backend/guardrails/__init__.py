"""
Guardrails for the BANKING RAG (Bank_Documents -> Pinecone namespace "bank-docs").

The rules follow the bank's own "Data Privacy & Customer Information Handling SOP" (DOC-026):
  §2 Masking   PAN XXXXX1234X, Aadhaar XXXX XXXX 1234, account numbers last 4 digits only,
               card numbers first 6 + last 4 (PCI-DSS)
  §3 Access    least privilege: relationship managers see statements, underwriters see loan files
  §4 AI usage  never return raw PII, refuse requests for another customer's data,
               log every retrieval with user, role and document ids

Files in this package:
  pii_masking.py       mask PAN / Aadhaar / account / card / mobile / e-mail / date of birth
  threat_detection.py  prompt injection, secrets (OTP/PIN/CVV/password), unmask & bulk requests,
                       customer names in a text
  access_control.py    roles -> metadata filter used inside the Pinecone query and BM25
  audit_log.py         one JSON line per question in logs/bank_audit.jsonl
  bank_guard.py        BankGuard: runs input -> retrieval -> output checks for one question

INGESTION guardrails (used by ingestion/ingest.py): PII masking before embedding, injection scan
(chunks quarantined), classification label (missing -> CONFIDENTIAL), access metadata.
"""

from guardrails.pii_masking import mask_pii, masked_total
from guardrails.threat_detection import find_injection
from guardrails.access_control import ROLES, DEFAULT_ROLE, roles_for_ui
from guardrails.bank_guard import BankGuard, REFUSALS

GUARD_VERSION = 1    # stored on bank vectors; bump when the masking rules change -> re-ingest

__all__ = ["mask_pii", "masked_total", "find_injection", "ROLES", "DEFAULT_ROLE", "roles_for_ui",
           "BankGuard", "REFUSALS", "GUARD_VERSION"]
