"""Record source availability for leakage-safe historical backtests.

Revision ID: 20260829_110000
Revises: 20260829_100000
Create Date: 2026-08-29
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "20260829_110000"
down_revision: Union[str, None] = "20260829_100000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "episodes",
        sa.Column("source_published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_episodes_source_published_at",
        "episodes",
        ["source_published_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_episodes_source_published_at", table_name="episodes")
    op.drop_column("episodes", "source_published_at")
