"""
Role-based access control (Data Privacy SOP §3 - least privilege).

Each role becomes a Pinecone metadata filter that is applied INSIDE the vector query and to
the in-memory BM25 search, so chunks outside the role's access are never retrieved.
In this demo the role is picked in the UI; a real application would take it from the login.
"""

# role -> (label, what it may see, metadata filter). "customer" is built per customer.
ROLES = {
    "relationship_manager": ("Relationship manager", "bank & credit-card statements + bank policies",
                             {"category": {"$in": ["statement", "policy"]}}),
    "underwriter": ("Underwriter / credit officer", "loan application files + bank policies",
                    {"category": {"$in": ["loan", "policy"]}}),
    "compliance": ("Compliance officer", "all bank documents (audit)", None),
    "customer": ("Customer", "only their own documents", None),
}
DEFAULT_ROLE = "relationship_manager"
NOBODY = {"customer": {"$eq": "__nobody__"}}       # unknown role -> matches nothing


def roles_for_ui():
    return [{"value": k, "label": v[0], "access": v[1]} for k, v in ROLES.items()]


def access_filter(role, customer=None):
    """Metadata filter for the role (None = no restriction)."""
    if role == "customer":
        return {"customer": {"$eq": customer}}
    return ROLES[role][2] if role in ROLES else NOBODY


def role_label(role, customer=None):
    if role not in ROLES:
        return "unknown"
    return f"{ROLES[role][0]}: {customer}" if role == "customer" else ROLES[role][0]


def access_text(role, customer=None):
    if role == "customer":
        return f"only documents of {customer}"
    return ROLES[role][1] if role in ROLES else "nothing"
