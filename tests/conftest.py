"""Test DB fixtures.

WHY: tests use TEST_DATABASE_URL (Postgres+pgvector in CI) and fall back to in-memory SQLite locally.
Each test starts from freshly seeded tables (rows deleted + re-seeded) so tests are isolated.
WHY not rollback-per-test: the tools open their own sessions and commit, as in production.
"""
import os

import pytest

from app.agent.embeddings import HashEmbeddingProvider
from app.data import db
from app.data.models import Base
from app.data.seed import seed


@pytest.fixture(scope="session")
def engine():
    eng = db.configure(os.environ.get("TEST_DATABASE_URL", "sqlite://"))
    db.init_schema(eng)
    return eng


@pytest.fixture()
def seeded(engine):
    with engine.begin() as c:
        for table in reversed(Base.metadata.sorted_tables):
            c.execute(table.delete())
    with db.session_scope() as s:
        seed(s, HashEmbeddingProvider())
    return engine


@pytest.fixture(autouse=True)
def _offline_embeddings(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    from app.config import get_settings

    get_settings.cache_clear()
