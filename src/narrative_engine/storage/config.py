"""Database configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _as_async_postgres_url(database_url: str) -> str:
    """Normalize supported synchronous PostgreSQL URLs for asyncpg."""
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgresql+psycopg2://"):
        return database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    return database_url


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection configuration."""

    database_url: str
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

    @classmethod
    def from_env(cls, prefix: str = "NE_") -> DatabaseConfig:
        """Create config from environment variables."""
        # Primary: DATABASE_URL or NE_DATABASE_URL
        database_url = os.getenv(f"{prefix}DATABASE_URL") or os.getenv("DATABASE_URL")

        if not database_url:
            # Match the checked-in Docker Compose development database. An
            # authority-free URL silently uses the host OS account and makes
            # a fresh local checkout fail authentication by default.
            database_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/narrative_engine"

        database_url = _as_async_postgres_url(database_url)

        return cls(
            database_url=database_url,
            echo=os.getenv(f"{prefix}DB_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv(f"{prefix}DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv(f"{prefix}DB_MAX_OVERFLOW", "10")),
        )

    def with_test_db(self) -> DatabaseConfig:
        """Return config pointing to test database."""
        test_url = os.getenv("NE_TEST_DATABASE_URL") or os.getenv("TEST_DATABASE_URL")
        if not test_url:
            test_url = self.database_url.rsplit("/", 1)[0] + "/narrative_engine_test"
        return DatabaseConfig(
            database_url=_as_async_postgres_url(test_url),
            echo=self.echo,
            pool_size=0,  # Use NullPool for tests
            max_overflow=0,
        )
