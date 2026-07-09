"""Add episodes.classification_state and theses.mode

tau_class / no-fit path (design doc Sec 6.2 stage 4, Sec 6.5.8):
episodes that fail the classification confidence floor are marked
"unclassified" instead of being force-fitted; theses generated from an
unclassified query situation are marked mode="arc_less".

Revision ID: 20260710_100000
Revises: 20260709_181500
Create Date: 2026-07-10 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '20260710_100000'
down_revision: Union[str, None] = '20260709_181500'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'episodes',
        sa.Column(
            'classification_state',
            sa.String(length=20),
            nullable=False,
            server_default='classified',
        ),
    )
    op.create_index(
        'ix_episodes_classification_state', 'episodes', ['classification_state']
    )
    op.add_column(
        'theses',
        sa.Column(
            'mode',
            sa.String(length=20),
            nullable=False,
            server_default='arc_based',
        ),
    )
    op.create_index('ix_theses_mode', 'theses', ['mode'])


def downgrade() -> None:
    op.drop_index('ix_theses_mode', table_name='theses')
    op.drop_column('theses', 'mode')
    op.drop_index('ix_episodes_classification_state', table_name='episodes')
    op.drop_column('episodes', 'classification_state')
