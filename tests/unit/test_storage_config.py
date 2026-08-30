"""Database configuration tests for a zero-setup local checkout."""

from narrative_engine.storage.config import DatabaseConfig


def test_local_default_matches_docker_compose_database(monkeypatch):
    monkeypatch.delenv("NE_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    config = DatabaseConfig.from_env()

    assert config.database_url == (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/narrative_engine"
    )


def test_test_database_override_is_honored_and_normalized(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://tester:secret@db.test/custom_test")
    config = DatabaseConfig(database_url="postgresql+asyncpg://unused/base")

    test_config = config.with_test_db()

    assert test_config.database_url == "postgresql+asyncpg://tester:secret@db.test/custom_test"
    assert test_config.pool_size == 0
