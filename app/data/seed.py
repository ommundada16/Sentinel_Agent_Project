"""Deterministic fixtures (users, roles, entitlements, requests) + embedded docs.
WHY seeded: reproducible demos and evals. Run: python -m app.data.seed
"""
from sqlalchemy.orm import Session

from app.agent.embeddings import EmbeddingProvider, get_embedder
from app.data import db
from app.data.ingest import ingest_corpus
from app.data.models import (
    AccessRequest, Entitlement, Role, RoleEntitlement, User, UserRole,
)

USERS = [
    (1, "Priya Sharma", "priya.sharma@example.com", "Finance", "active"),
    (2, "Rahul Verma", "rahul.verma@example.com", "Engineering", "active"),
    (3, "Anita Desai", "anita.desai@example.com", "HR", "active"),
    (4, "Vikram Singh", "vikram.singh@example.com", "IT", "active"),
    (5, "Meera Nair", "meera.nair@example.com", "Finance", "active"),
    (6, "Arjun Mehta", "arjun.mehta@example.com", "Sales", "active"),
    (7, "Sara Khan", "sara.khan@example.com", "Engineering", "active"),
    (8, "Dev Patel", "dev.patel@example.com", "Security", "active"),
    (9, "Neha Gupta", "neha.gupta@example.com", "HR", "active"),
    (10, "Karan Joshi", "karan.joshi@example.com", "Sales", "inactive"),
    (11, "Isha Reddy", "isha.reddy@example.com", "Engineering", "active"),
    (12, "Tom Becker", "tom.becker@example.com", "Finance", "active"),
    (13, "Lina Park", "lina.park@example.com", "IT", "active"),
    (14, "Omar Haddad", "omar.haddad@example.com", "Sales", "active"),
    (15, "Priya Nambiar", "priya.nambiar@example.com", "Engineering", "active"),
]
ROLES = [
    (1, "Finance Analyst", "Read/write in the finance app"),
    (2, "Finance Admin", "Administers the finance app"),
    (3, "HR Specialist", "Reads HR portal"),
    (4, "Engineer", "Source and cloud read access"),
    (5, "DevOps", "Deploys and administers cloud"),
    (6, "Sales Rep", "Reads CRM"),
]
ENTITLEMENTS = [
    (1, "FinanceApp", "read", "low"), (2, "FinanceApp", "write", "medium"),
    (3, "FinanceApp", "admin", "high"), (4, "HRPortal", "read", "low"),
    (5, "HRPortal", "admin", "high"), (6, "GitHub", "read", "low"),
    (7, "GitHub", "admin", "high"), (8, "AWS", "console", "medium"),
    (9, "AWS", "admin", "high"), (10, "Salesforce", "read", "low"),
]
ROLE_ENTS = {1: [1, 2], 2: [1, 2, 3], 3: [4], 4: [6, 8], 5: [6, 7, 8, 9], 6: [10]}
USER_ROLES = {1: [2], 2: [4], 3: [3], 4: [5], 5: [1], 6: [6], 7: [4], 8: [5], 9: [3, 6],
              10: [6], 11: [4], 12: [1], 13: [5], 14: [6], 15: [4]}


def seed(s: Session, embedder: EmbeddingProvider | None = None) -> None:
    s.add_all(User(id=i, name=n, email=e, department=d, status=st) for i, n, e, d, st in USERS)
    s.add_all(Role(id=i, name=n, description=d) for i, n, d in ROLES)
    s.add_all(Entitlement(id=i, app_name=a, permission=p, sensitivity=sv) for i, a, p, sv in ENTITLEMENTS)
    s.flush()
    s.add_all(RoleEntitlement(role_id=r, entitlement_id=e) for r, es in ROLE_ENTS.items() for e in es)
    s.add_all(UserRole(user_id=u, role_id=r) for u, rs in USER_ROLES.items() for r in rs)
    s.add_all([
        AccessRequest(user_id=6, entitlement_id=2, status="pending", justification="Quarter-end reporting"),
        AccessRequest(user_id=2, entitlement_id=8, status="approved", decided_by="manager"),
        AccessRequest(user_id=12, entitlement_id=3, status="denied", decided_by="security",
                      justification="Not eligible"),
    ])
    s.flush()
    ingest_corpus(s, embedder or get_embedder())


if __name__ == "__main__":
    db.init_schema()
    with db.session_scope() as session:
        seed(session)
    print("seeded")
