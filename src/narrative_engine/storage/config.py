"""Database configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


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
            # Default local development database
            database_url = "postgresql+asyncpg://localhost:5432/narrative_engine"

        # Convert sync driver to async if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgresql+psycopg2://"):
            database_url = database_url.replace(
                "postgresql+psycopg2://", "postgresql+asyncpg://", 1
            )

        return cls(
            database_url=database_url,
            echo=os.getenv(f"{prefix}DB_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv(f"{prefix}DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv(f"{prefix}DB_MAX_OVERFLOW", "10")),
        )

    def with_test_db(self) -> DatabaseConfig:
        """Return config pointing to test database."""
        test_url = self.database_url.rsplit("/", 1)[0] + "/narrative_engine_test"
        return DatabaseConfig(
            database_url=test_url,
            echo=self.echo,
            pool_size=0,  # Use NullPool for tests
            max_overflow=0,
        )
