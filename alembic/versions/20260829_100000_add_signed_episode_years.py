"""Add signed episode years so BCE chronology is not discarded.

Revision ID: 20260829_100000
Revises: 20260828_100000
Create Date: 2026-08-29
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "20260829_100000"
down_revision: Union[str, None] = "20260828_100000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("episodes", sa.Column("start_year", sa.Integer(), nullable=True))
    op.add_column("episodes", sa.Column("end_year", sa.Integer(), nullable=True))
    op.execute("UPDATE episodes SET start_year = EXTRACT(YEAR FROM start_date)::integer WHERE start_date IS NOT NULL")
    op.execute("UPDATE episodes SET end_year = EXTRACT(YEAR FROM end_date)::integer WHERE end_date IS NOT NULL")
    op.create_index("ix_episodes_start_year", "episodes", ["start_year"])


def downgrade() -> None:
    op.drop_index("ix_episodes_start_year", table_name="episodes")
    op.drop_column("episodes", "end_year")
    op.drop_column("episodes", "start_year")
