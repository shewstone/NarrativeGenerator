"""Add arc_instances and episode_links tables

Revision ID: 20260709_162900
Revises: c732fb5ac39c
Create Date: 2026-07-09 16:29:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20260709_162900'
down_revision: Union[str, None] = 'c732fb5ac39c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create arc_instances table
    op.create_table(
        'arc_instances',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('arc_type', sa.String(length=100), nullable=False),
        sa.Column('canonical_name', sa.String(length=500), nullable=False),
        sa.Column('start_date', sa.DateTime(), nullable=True),
        sa.Column('end_date', sa.DateTime(), nullable=True),
        sa.Column('phases', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('source_coverage', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('coverage_gaps', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('framework_id', sa.UUID(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_arc_instances_arc_type', 'arc_instances', ['arc_type'])
    op.create_index('ix_arc_instances_status', 'arc_instances', ['status'])
    op.create_index('ix_arc_instances_canonical_name', 'arc_instances', ['canonical_name'])

    # Create episode_links table
    op.create_table(
        'episode_links',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('source_episode_id', sa.UUID(), nullable=False),
        sa.Column('target_episode_id', sa.UUID(), nullable=False),
        sa.Column('edge_kind', sa.String(length=50), nullable=False),  # CAUSES, PRECEDES, COMPOSES, MEMBER_OF, SAME_EVENT_AS
        sa.Column('link_status', sa.String(length=50), nullable=False, server_default='attested'),  # attested | inferred
        sa.Column('distance', sa.Float(), nullable=True),  # semantic distance for inferred links
        sa.Column('evidence', sa.Text(), nullable=True),  # quote for attested, similarity score for inferred
        sa.Column('review_status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('reviewed_by', sa.String(length=255), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['source_episode_id'], ['episodes.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_episode_id'], ['episodes.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_episode_id', 'target_episode_id', 'edge_kind', name='uq_episode_link'),
        # Invariant: CAUSES edges must be attested (no inferred causal claims)
        sa.CheckConstraint("NOT (edge_kind = 'CAUSES' AND link_status = 'inferred')", name='chk_causal_must_be_attested')
    )
    op.create_index('ix_episode_links_source', 'episode_links', ['source_episode_id'])
    op.create_index('ix_episode_links_target', 'episode_links', ['target_episode_id'])
    op.create_index('ix_episode_links_edge_kind', 'episode_links', ['edge_kind'])
    op.create_index('ix_episode_links_link_status', 'episode_links', ['link_status'])
    op.create_index('ix_episode_links_review_status', 'episode_links', ['review_status'])

    # Create arc_instance_episodes association table
    op.create_table(
        'arc_instance_episodes',
        sa.Column('arc_instance_id', sa.UUID(), nullable=False),
        sa.Column('episode_id', sa.UUID(), nullable=False),
        sa.Column('phase', sa.String(length=50), nullable=False),
        sa.ForeignKeyConstraint(['arc_instance_id'], ['arc_instances.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['episode_id'], ['episodes.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('arc_instance_id', 'episode_id', 'phase')
    )
    op.create_index('ix_arc_instance_episodes_arc', 'arc_instance_episodes', ['arc_instance_id'])
    op.create_index('ix_arc_instance_episodes_episode', 'arc_instance_episodes', ['episode_id'])

    # Create episode_mechanisms association table
    op.create_table(
        'episode_mechanisms',
        sa.Column('episode_id', sa.UUID(), nullable=False),
        sa.Column('mechanism_tag', sa.String(length=100), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.5'),
        sa.ForeignKeyConstraint(['episode_id'], ['episodes.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('episode_id', 'mechanism_tag')
    )


def downgrade() -> None:
    op.drop_table('episode_mechanisms')
    op.drop_table('arc_instance_episodes')
    op.drop_index('ix_episode_links_review_status', table_name='episode_links')
    op.drop_index('ix_episode_links_type', table_name='episode_links')
    op.drop_index('ix_episode_links_target', table_name='episode_links')
    op.drop_index('ix_episode_links_source', table_name='episode_links')
    op.drop_table('episode_links')
    op.drop_index('ix_arc_instances_canonical_name', table_name='arc_instances')
    op.drop_index('ix_arc_instances_status', table_name='arc_instances')
    op.drop_index('ix_arc_instances_arc_type', table_name='arc_instances')
    op.drop_table('arc_instances')
