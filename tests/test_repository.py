from app.data import db
from app.data import repository as repo
from app.data.ingest import chunk_text


def test_find_users_by_name_and_email(seeded):
    with db.session_scope() as s:
        users, more = repo.find_users(s, "priya")
        assert {u["name"] for u in users} == {"Priya Sharma", "Priya Nambiar"} and not more
        assert repo.find_users(s, "priya.sharma@example.com")[0][0]["id"] == 1


def test_find_users_is_bounded(seeded):
    with db.session_scope() as s:
        users, more = repo.find_users(s, "a", limit=3)  # matches many users
        assert len(users) == 3 and more is True


def test_entitlements_via_role_and_pagination(seeded):
    with db.session_scope() as s:
        items, more = repo.list_entitlements_for_user(s, 1)  # Finance Admin
        assert {e["id"] for e in items} == {1, 2, 3} and not more
        items, more = repo.list_entitlements_for_user(s, 1, limit=2)
        assert len(items) == 2 and more


def test_revoke_removes_entitlement(seeded):
    with db.session_scope() as s:
        repo.revoke_entitlement(s, 1, 3, "no longer needed")
    with db.session_scope() as s:
        assert not repo.user_has_entitlement(s, 1, 3)
        assert repo.user_has_entitlement(s, 1, 1)


def test_sql_injection_string_is_just_data(seeded):
    with db.session_scope() as s:
        users, _ = repo.find_users(s, "'; DROP TABLE users; --")
        assert users == []
        assert repo.get_user(s, 1) is not None


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(text, size=150, overlap=25)
    assert all(len(c.split()) <= 150 for c in chunks)
    assert chunks[0].split()[-25:] == chunks[1].split()[:25]
    assert " ".join(chunks).count("w399") >= 1
