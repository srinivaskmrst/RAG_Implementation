"""
BankGuard: runs all retrieval-time guardrails for ONE bank question.

    check_input()      role / identity, prompt injection, secrets, unmask, bulk export,
                       another customer's data -> BLOCKED; PII in the question masked
    where              role-based metadata filter for Pinecone + BM25 (access_control.py)
    screen(chunks)     retrieved chunks: access re-check, injection scan, PII re-mask
    check_output(ans)  PII in the answer masked; a customer's answer naming another customer withheld
    audit(...)         one line in logs/bank_audit.jsonl
"""

import time

from core.utils import matches_filter
from guardrails.pii_masking import mask_pii, masked_total
from guardrails.threat_detection import find_injection, names_in, SECRET_RE, UNMASK_RE, BULK_RE
from guardrails import access_control
from guardrails.audit_log import write_audit, AUDIT_LOG

REFUSALS = {
    "injection": "🛡️ Request blocked by the guardrails: it contains instructions that try to change or "
                 "bypass the assistant's rules (prompt injection). Please ask a normal question about the "
                 "bank documents.",
    "secret": "🛡️ Request blocked: OTPs, PINs, CVVs and passwords are never stored or shared. "
              "Meridian Trust Bank will never ask for them – do not share them with anyone.",
    "unmask": "🛡️ Request blocked: personal data (PAN, Aadhaar, account / card numbers, mobile, e-mail, "
              "date of birth) is only shown masked, as required by the bank's Data Privacy SOP §2.",
    "bulk": "🛡️ Request blocked: bulk export of customers' personal data is not allowed "
            "(Data Privacy SOP §3 – least privilege).",
    "other_customer": "🛡️ Request blocked: you can only ask about your own accounts and loans. "
                      "Requests for another customer's data are refused (Data Privacy SOP §4).",
    "role": "🛡️ Request blocked: unknown role or customer – access cannot be determined.",
    "withheld": "🛡️ The answer was withheld by the output guardrail because it referred to another "
                "customer's data. You can only see information about your own accounts.",
}


class BankGuard:
    def __init__(self, question, role, customer, customers):
        self.raw_question = question
        self.question = question
        self.role = role if role in access_control.ROLES else None
        self.customer = (customer or "").strip()
        self.customers = list(customers)
        self.checks = []                 # [{"stage", "name", "status", "detail"}]
        self.blocked = None              # reason key in REFUSALS
        self.stats = {"chunks_checked": 0, "access_dropped": 0, "injection_dropped": 0,
                      "chunks_remasked": 0, "output_masked": 0, "tool_denied": 0}
        self.tool_calls = []             # Agentic RAG: every tool call (for the audit log)
        self.audit_id = None

    # ---- helpers
    def _check(self, stage, name, status, detail=""):
        self.checks.append({"stage": stage, "name": name, "status": status, "detail": detail})
        return status != "blocked"

    def _block(self, reason, name, detail):
        self._check("input", name, "blocked", detail)
        self.blocked = reason
        return False

    @property
    def role_label(self):
        return access_control.role_label(self.role, self.customer)

    @property
    def where(self):
        return access_control.access_filter(self.role, self.customer)

    @property
    def access_text(self):
        return access_control.access_text(self.role, self.customer)

    # ---- 1. input guardrails
    def check_input(self):
        q = self.raw_question
        if not self.role or (self.role == "customer" and self.customer not in self.customers):
            return self._block("role", "Role / identity", f"role={self.role}, customer={self.customer or '-'}")
        self._check("input", "Role / identity", "pass", self.role_label)

        hit = find_injection(q)
        if hit:
            return self._block("injection", "Prompt injection / jailbreak", f"matched “{hit}”")
        self._check("input", "Prompt injection / jailbreak", "pass", "no injection pattern")

        hit = SECRET_RE.search(q)
        if hit:
            return self._block("secret", "Credentials (OTP / PIN / CVV / password)", f"asked for “{hit.group(0)}”")
        self._check("input", "Credentials (OTP / PIN / CVV / password)", "pass", "not requested")

        hit = UNMASK_RE.search(q)
        if hit:
            return self._block("unmask", "Unmasked PII request", f"“{hit.group(0)}”")
        self._check("input", "Unmasked PII request", "pass", "not requested")

        hit = BULK_RE.search(q)
        if hit:
            return self._block("bulk", "Bulk PII export", f"“{hit.group(0)}”")
        self._check("input", "Bulk PII export", "pass", "not requested")

        if self.role == "customer":
            others = names_in(q, [c for c in self.customers if c != self.customer])
            if others:
                return self._block("other_customer", "Customer data isolation",
                                   f"question names another customer: {', '.join(sorted(others))}")
            self._check("input", "Customer data isolation", "pass", f"only {self.customer}")
        else:
            self._check("input", "Customer data isolation", "pass", "staff role – limited by access filter")

        masked, counts = mask_pii(q)
        if counts:
            self.question = masked
            self._check("input", "PII in the question", "masked",
                        f"{masked_total(counts)} value(s) masked before search / LLM / log")
        else:
            self._check("input", "PII in the question", "pass", "none found")
        return True

    # ---- 2. retrieval guardrails (called for every search)
    def screen(self, chunks):
        """Defence in depth on retrieved chunks: access re-check, injection scan, PII re-mask."""
        kept = []
        for c in chunks:
            self.stats["chunks_checked"] += 1
            if not matches_filter(c, self.where):
                self.stats["access_dropped"] += 1
                continue
            if find_injection(c["text"]):
                self.stats["injection_dropped"] += 1
                continue
            text, counts = mask_pii(c["text"])
            if counts:
                self.stats["chunks_remasked"] += 1
                c = {**c, "text": text}
            kept.append(c)
        return kept

    def log_tool(self, tool, args, result):
        """Agentic RAG: record one tool call (denied calls are counted)."""
        self.tool_calls.append({"tool": tool, "args": mask_pii(str(args))[0], "result": str(result)[:120]})
        if result == "denied":
            self.stats["tool_denied"] += 1

    def retrieval_summary(self):
        s = self.stats
        text = (f"{s['chunks_checked']} chunks checked · {s['access_dropped']} dropped (access) · "
                f"{s['injection_dropped']} dropped (injection) · {s['chunks_remasked']} re-masked")
        if self.tool_calls:
            text += f" · {len(self.tool_calls)} agent tool calls, {s['tool_denied']} denied"
        return text

    # ---- 3. output guardrails
    def check_output(self, answer):
        masked, counts = mask_pii(answer or "")
        self.stats["output_masked"] = masked_total(counts)
        self._check("output", "PII in the answer", "masked" if counts else "pass",
                    f"{masked_total(counts)} value(s) masked" if counts else "no unmasked PII")
        if self.role == "customer":
            others = names_in(masked, [c for c in self.customers if c != self.customer])
            if others:
                self._check("output", "Customer data isolation", "blocked",
                            f"answer mentioned {', '.join(sorted(others))} – withheld")
                return REFUSALS["withheld"]
            self._check("output", "Customer data isolation", "pass", "answer only about this customer")
        return masked

    # ---- 4. audit log (SOP §4)
    def audit(self, mode, documents, llm_label):
        self.audit_id = write_audit({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "pipeline": mode,
            "role": self.role, "customer": self.customer or None,
            "question": mask_pii(self.question)[0],
            "decision": f"blocked:{self.blocked}" if self.blocked else "answered",
            "documents": sorted(set(documents)), "llm": llm_label,
            "checks": [f"{c['name']}={c['status']}" for c in self.checks if c["status"] != "pass"],
            "tool_calls": self.tool_calls, **self.stats})
        return self.audit_id

    def report(self):
        return {"role": self.role_label, "access": self.access_text, "blocked": self.blocked,
                "checks": self.checks, "stats": self.stats, "tool_calls": self.tool_calls,
                "audit_id": self.audit_id,
                "audit_log": f"logs/{AUDIT_LOG.name}"}
