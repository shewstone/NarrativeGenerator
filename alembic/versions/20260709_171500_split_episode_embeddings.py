"""Split episode embedding into surface_embedding and structural_embedding

Design doc Sec 3.3a: identity resolution (SAME_EVENT_AS, arc composition)
must use the raw-text "surface" embedding; analog retrieval and discovery
clustering must use the role-substituted "structural" embedding. A single
shared column made that invariant unenforceable -- composition code fell
back to whatever the one column held.

Also corrects the vector dimension from 768 to 384: the pinned embedding
model (sentence-transformers/all-MiniLM-L6-v2) emits 384-dim vectors: the
original 768 was never actually usable for inserts from this model.

Revision ID: 20260709_171500
Revises: 20260709_170000
Create Date: 2026-07-09 17:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = '20260709_171500'
down_revision: Union[str, None] = '20260709_170000'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'episodes',
        sa.Column('surface_embedding', pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=True),
    )
    op.add_column(
        'episodes',
        sa.Column('structural_embedding', pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=True),
    )

    # The old single `embedding` column was populated (where present) via
    # generate_for_episode(), which rendered the role-substituted narrative
    # template -- i.e. it was already structurally shaped. Its dimension
    # (768) doesn't match the new 384-dim columns, so any existing rows
    # cannot be copied as-is; those episodes need re-embedding via the
    # batch job (design doc Sec 6.3), not a same-dimension column copy.
    # surface_embedding starts NULL for existing rows for the same reason.

    op.drop_index('ix_episodes_embedding', table_name='episodes', postgresql_using='ivfflat')
    op.drop_column('episodes', 'embedding')

    op.create_index(
        'ix_episodes_structural_embedding',
        'episodes',
        ['structural_embedding'],
        unique=False,
        postgresql_using='ivfflat',
    )


def downgrade() -> None:
    op.drop_index('ix_episodes_structural_embedding', table_name='episodes', postgresql_using='ivfflat')

    op.add_column(
        'episodes',
        sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=768), nullable=True),
    )
    op.create_index('ix_episodes_embedding', 'episodes', ['embedding'], unique=False, postgresql_using='ivfflat')

    op.drop_column('episodes', 'structural_embedding')
    op.drop_column('episodes', 'surface_embedding')
