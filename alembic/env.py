from alembic import context
from app.config import get_settings
from app.data import db
from app.data.models import Base

target_metadata = Base.metadata


def run_migrations_online() -> None:
    engine = db.configure(get_settings().database_url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
