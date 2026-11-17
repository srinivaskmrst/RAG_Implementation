"""
Document loading: list the files of a dataset folder, read their pages and describe each document.

    list_files(folder)                         supported files (.pdf / .txt / .md), sorted
    read_pages(path)                           [(page_number, text), ...]
    load_manifest(folder)                      optional manifest.json {file name: info}
    describe_leave(path, text, manifest)       -> year, doc_id, info (CL / SL / EL, effective dates)
    describe_bank(path, text)                  -> year, doc_id, info (customer, type, category,
                                                  classification), has_classification_label
"""

import json
import re

from pypdf import PdfReader

from core.config import SUPPORTED_EXTENSIONS

LEAVE_FIELDS = ("effective_from", "effective_to", "cl", "sl", "el")   # headline info from manifest.json


def list_files(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder '{folder.name}' not found next to 'Live Application'")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def read_pages(path):
    """List of (page_number, text) for one file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [(n, page.extract_text() or "") for n, page in enumerate(reader.pages, start=1)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def load_manifest(folder):
    path = folder / "manifest.json"
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return {d["file"]: d for d in entries}


def describe_leave(path, text, manifest):
    """Year, document id and headline info: from manifest.json, else detected from name / text."""
    entry = manifest.get(path.name)
    if entry:
        return int(entry["year"]), entry["doc_id"], {k: str(entry[k]) for k in LEAVE_FIELDS if entry.get(k)}
    in_name = re.search(r"(?:19|20)\d{2}", path.stem)                     # Leave_Policy_2019.pdf
    in_text = re.search(r"Policy year\s*((?:19|20)\d{2})", text)          # "Policy year 2019"
    year = int(in_name.group(0)) if in_name else int(in_text.group(1)) if in_text else 0
    doc_id = re.search(r"HR-LV-\d{4}", text)
    return year, doc_id.group(0) if doc_id else path.stem, {}


def describe_bank(path, text):
    """DOC-011_HomeLoan_Arjun_Mehta.pdf -> doc DOC-011, type "Home Loan", customer "Arjun Mehta",
    category loan, classification from the document header (fail-safe CONFIDENTIAL)."""
    m = re.match(r"(DOC-\d+)_([A-Za-z]+)(?:_(.+))?$", path.stem)
    doc_id = m.group(1) if m else path.stem
    doc_type = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m.group(2)) if m else "Document"
    customer = m.group(3).replace("_", " ").strip() if m and m.group(3) else ""
    category = "statement" if "Statement" in doc_type else "loan" if "Loan" in doc_type else "policy"
    label = re.search(r"Classification:\s*([A-Z_]+)", text)
    info = {"doc_type": doc_type, "category": category,
            "classification": label.group(1) if label else "CONFIDENTIAL"}
    if customer:
        info["customer"] = customer
    cust_id = re.search(r"\bCUST-\d+\b", text)
    if cust_id:
        info["customer_id"] = cust_id.group(0)
    return 0, doc_id, info, bool(label)
