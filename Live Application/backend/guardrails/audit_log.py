"""
Audit log (Data Privacy SOP §4): one JSON line per bank question in Live Application/logs/bank_audit.jsonl
(git-ignored). Stores time, role, customer, the MASKED question, the decision and document ids.
"""

import json
import threading
import uuid
from pathlib import Path

# backend/guardrails/audit_log.py -> parents: guardrails, backend, Live Application
AUDIT_LOG = Path(__file__).resolve().parents[2] / "logs" / "bank_audit.jsonl"
_lock = threading.Lock()


def write_audit(entry):
    """Append one entry; returns its new id."""
    entry = {"id": uuid.uuid4().hex[:12], **entry}
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock, AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry["id"]
