from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, text
from alembic import context

# --- import SQLModel metadata ---
from munshi_machine.models.private import PrivateSchemaBase
from dotenv import load_dotenv
import os

# register_models = private_models  # to avoid linter unused import warning

load_dotenv()  # loads .env from project root
DATABASE_URL = os.getenv("DATABASE_URL")


# this is the Alembic Config object
config = context.config

# Override sqlalchemy.url in alembic.ini
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# interpret logging config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# target metadata for 'autogenerate':
target_metadata = PrivateSchemaBase.metadata


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        connection.execute(text('CREATE SCHEMA IF NOT EXISTS "private"'))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            compare_type=True,  # detect column type changes
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
