"""Pytest configuration and fixtures."""

import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from narrative_engine.storage.database import Base

# Use test database URL from environment or default
# CI PostgreSQL service: postgres/postgres@localhost:5432
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/narrative_engine_test",
)


@pytest_asyncio.fixture(scope="function")
async def engine():
    """Create database engine for each test function."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        poolclass=NullPool,  # NullPool for tests - no connection pooling
    )

    # Create pgvector extension AND tables in same connection block
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop all tables after test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for a test."""
    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    async with async_session() as session:
        yield session
        # Rollback after each test
        await session.rollback()


@pytest.fixture
def sample_episode_data():
    """Sample episode data for testing."""
    return {
        "title": "1929 Stock Market Crash",
        "summary": "The collapse of the US stock market in October 1929",
        "location": "United States",
        "initiating_conditions": [
            "Speculative excess in stock market",
            "Margin trading widespread",
        ],
        "escalation_mechanics": ["Panic selling cascades", "Margin calls force liquidation"],
        "consequences": ["Great Depression begins", "Banking crisis follows"],
    }


@pytest.fixture
def sample_cycle_data():
    """Sample cycle data for testing."""
    return {
        "name": "Fourth Turning",
        "scale": "generational",
        "description": "Strauss-Howe generational crisis",
        "framework_source": "Strauss-Howe-1997",
    }


@pytest.fixture
def sample_thesis_data():
    """Sample thesis data for testing."""
    return {
        "query": "Will 2024 see a market crash?",
        "dominant_continuation": "Soft landing likely based on historical analogs",
        "alternative_continuations": [("Hard landing", 0.3), ("Continued expansion", 0.2)],
        "watch_for_indicators": ["Credit spreads widening", "Yield curve inversion persistence"],
        "model_version": "gpt-4",
        "taxonomy_version": "v0.1.0",
    }


# pytest-asyncio configuration
# Using function-scoped event loop to prevent asyncpg connection conflicts
