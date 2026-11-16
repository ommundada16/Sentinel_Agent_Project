"""initial schema: tables, pgvector extension, HNSW index

Revision ID: 0001
"""
from alembic import op

from app.data import db

revision = "0001"
down_revision = None


def upgrade() -> None:
    # WHY delegate: one definition of the schema (models + HNSW index) shared with tests.
    # Trade-off: this first migration is a snapshot of today's models; later changes get real diffs.
    db.init_schema(op.get_bind().engine)


def downgrade() -> None:
    from app.data.models import Base

    Base.metadata.drop_all(op.get_bind())
