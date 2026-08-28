"""Add scale-neutral situation ontology and focal-scope claims.

Revision ID: 20260828_100000
Revises: 20260712_030000
Create Date: 2026-08-28
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "20260828_100000"
down_revision: Union[str, None] = "20260712_030000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("episodes", sa.Column("scope_name", sa.String(length=500), nullable=True))
    op.add_column("episodes", sa.Column("scope_kind", sa.String(length=30), nullable=True))
    op.add_column("episodes", sa.Column("parent_scope_name", sa.String(length=500), nullable=True))
    op.add_column("episodes", sa.Column("scope_confidence", sa.Float(), nullable=True))
    op.add_column("episodes", sa.Column("scope_evidence", sa.Text(), nullable=True))
    op.add_column("episodes", sa.Column("scope_notes", sa.Text(), nullable=True))

    op.add_column("episodes", sa.Column("change_pattern", sa.String(length=60), nullable=True))
    op.add_column(
        "episodes",
        sa.Column("pattern_confidence", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column("episodes", sa.Column("pattern_rationale", sa.Text(), nullable=True))
    op.add_column("episodes", sa.Column("situation_scale", sa.String(length=30), nullable=True))
    op.add_column("episodes", sa.Column("domains", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("episodes", sa.Column("configuration", sa.JSON(), nullable=False, server_default="{}"))
    op.add_column(
        "episodes",
        sa.Column("mechanism_families", sa.JSON(), nullable=False, server_default="[]"),
    )

    op.create_index("ix_episodes_scope_name", "episodes", ["scope_name"])
    op.create_index("ix_episodes_change_pattern", "episodes", ["change_pattern"])
    op.create_index("ix_episodes_situation_scale", "episodes", ["situation_scale"])


def downgrade() -> None:
    op.drop_index("ix_episodes_situation_scale", table_name="episodes")
    op.drop_index("ix_episodes_change_pattern", table_name="episodes")
    op.drop_index("ix_episodes_scope_name", table_name="episodes")

    for column in (
        "mechanism_families",
        "configuration",
        "domains",
        "situation_scale",
        "pattern_rationale",
        "pattern_confidence",
        "change_pattern",
        "scope_notes",
        "scope_evidence",
        "scope_confidence",
        "parent_scope_name",
        "scope_kind",
        "scope_name",
    ):
        op.drop_column("episodes", column)
