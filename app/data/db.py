"""Engine/session handling + schema creation."""
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.data.models import Base

_engine: Engine | None = None
_factory: sessionmaker | None = None


def configure(url: str | None = None) -> Engine:
    """(Re)point the app at a database. Tests call this with an in-memory SQLite URL."""
    global _engine, _factory
    url = url or get_settings().database_url
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url or url == "sqlite://":
            kwargs["poolclass"] = StaticPool  # WHY: one shared connection so the in-memory DB persists
    _engine = create_engine(url, **kwargs)
    _factory = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_engine() -> Engine:
    return _engine or configure()


@contextmanager
def session_scope():
    get_engine()
    session: Session = _factory()  # type: ignore[misc]
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_schema(engine: Engine | None = None) -> None:
    """Create extension, tables and the ANN index. Shared by Alembic and tests."""
    engine = engine or get_engine()
    if engine.dialect.name == "postgresql":
        with engine.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as c:
            # WHY HNSW: approximate-nearest-neighbour index so search stays fast at 8,000 docs or 8M.
            c.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_doc_chunks_embedding_hnsw "
                "ON doc_chunks USING hnsw (embedding vector_cosine_ops)"
            ))
