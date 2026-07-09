"""Add hybrid taxonomy system

Revision ID: 56b589b3f8a1
Revises:
Create Date: 2026-07-09 02:46:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "56b589b3f8a1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create tables for hybrid taxonomy system.

    This migration adds support for:
    - Versioned taxonomies (canonical/discovered/hybrid)
    - Arcs stored in database (replacing hardcoded ArcType enum)
    - Soft membership between episodes and arcs
    - Arc comparisons for evaluation
    """

    # Create taxonomies table
    op.create_table(
        "taxonomies",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "taxonomy_type",
            sa.Enum("canonical", "discovered", "hybrid", name="taxonomy_type"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("draft", "active", "deprecated", "archived", name="taxonomy_status"),
            default="draft",
        ),
        sa.Column("version", sa.String(50), default="1.0.0"),
        sa.Column(
            "parent_taxonomy_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("taxonomies.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("discovery_params", sa.JSON(), default={}),
    )

    # Create arcs table (unified for canonical and discovered)
    op.create_table(
        "arcs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "arc_type",
            sa.String(20),
            nullable=False,
            comment="Discriminator: 'canonical' or 'discovered'",
        ),
        sa.Column("slug", sa.String(255), nullable=True, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("phases", sa.JSON(), default=list),
        sa.Column("theoretical_sources", sa.JSON(), default=list),
        sa.Column("keywords", sa.JSON(), default=list),
        sa.Column("example_episodes", sa.JSON(), default=list),
        sa.Column("cluster_id", sa.Integer(), nullable=True),
        sa.Column("member_count", sa.Integer(), default=0),
        sa.Column("silhouette_score", sa.Float(), nullable=True),
        sa.Column("representative_episode_ids", sa.JSON(), default=list),
        sa.Column("boundary_episode_ids", sa.JSON(), default=list),
        sa.Column("common_features", sa.JSON(), default=dict),
        sa.Column("canonical_arc_mappings", sa.JSON(), default=dict),
        sa.Column(
            "embedding_centroid",
            sa.ARRAY(sa.Float()),  # pgvector will be added in separate step
            nullable=True,
        ),
        sa.Column("embedding_model", sa.String(255), nullable=True),
        sa.Column("discovered_at", sa.DateTime(), nullable=True),
        sa.Column("discovery_algorithm", sa.String(255), nullable=True),
        sa.Column("discovery_params", sa.JSON(), default=dict),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column(
            "taxonomy_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("taxonomies.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )

    # Create episode_arc_membership table (replaces hardcoded arc_type on episodes)
    op.create_table(
        "episode_arc_membership",
        sa.Column(
            "episode_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("episodes.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "arc_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("arcs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("arc_type", sa.String(20), nullable=False),
        sa.Column("membership_score", sa.Float(), default=1.0),
        sa.Column("confidence", sa.Float(), default=1.0),
        sa.Column("phase", sa.String(50), nullable=True),
        sa.Column("phase_confidence", sa.Float(), default=0.0),
        sa.Column("assignment_method", sa.String(50), default="manual"),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("distance_to_centroid", sa.Float(), nullable=True),
        sa.Column(
            "taxonomy_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("taxonomies.id"),
            nullable=False,
        ),
        sa.Column("assigned_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("assigned_by", sa.String(255), nullable=True),
    )

    # Create arc_comparisons table
    op.create_table(
        "arc_comparisons",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "baseline_taxonomy_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("taxonomies.id"),
            nullable=False,
        ),
        sa.Column(
            "comparison_taxonomy_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("taxonomies.id"),
            nullable=False,
        ),
        sa.Column("episode_agreement_score", sa.Float(), nullable=True),
        sa.Column("silhouette_comparison", sa.Float(), nullable=True),
        sa.Column("arc_mappings", sa.JSON(), default=dict),
        sa.Column("retrieval_precision", sa.Float(), nullable=True),
        sa.Column("retrieval_recall", sa.Float(), nullable=True),
        sa.Column("thesis_accuracy", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("evaluated_by", sa.String(255), nullable=True),
    )

    # Add pgvector extension for embedding_centroid
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Add vector column with proper dimension
    op.add_column(
        "arcs",
        sa.Column(
            "embedding_centroid_vector",
            sa.dialects.postgresql.VECTOR(384),
            nullable=True,
        ),
    )

    # Create indexes for performance
    op.create_index("ix_arcs_slug", "arcs", ["slug"])
    op.create_index("ix_arcs_arc_type", "arcs", ["arc_type"])
    op.create_index("ix_arcs_taxonomy_id", "arcs", ["taxonomy_id"])
    op.create_index("ix_taxonomies_name", "taxonomies", ["name"])
    op.create_index("ix_taxonomies_status", "taxonomies", ["status"])
    op.create_index(
        "ix_episode_arc_membership_episode_id",
        "episode_arc_membership",
        ["episode_id"],
    )
    op.create_index(
        "ix_episode_arc_membership_arc_id",
        "episode_arc_membership",
        ["arc_id"],
    )

    # Create HNSW index for vector similarity search on arc centroids
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_arcs_embedding_centroid_vector
        ON arcs USING hnsw (embedding_centroid_vector vector_cosine_ops)
        """
    )


def downgrade() -> None:
    """Remove hybrid taxonomy system tables."""

    # Drop indexes
    op.drop_index("ix_arcs_embedding_centroid_vector", table_name="arcs")
    op.drop_index("ix_episode_arc_membership_arc_id", table_name="episode_arc_membership")
    op.drop_index("ix_episode_arc_membership_episode_id", table_name="episode_arc_membership")
    op.drop_index("ix_taxonomies_status", table_name="taxonomies")
    op.drop_index("ix_taxonomies_name", table_name="taxonomies")
    op.drop_index("ix_arcs_taxonomy_id", table_name="arcs")
    op.drop_index("ix_arcs_arc_type", table_name="arcs")
    op.drop_index("ix_arcs_slug", table_name="arcs")

    # Drop columns
    op.drop_column("arcs", "embedding_centroid_vector")

    # Drop tables
    op.drop_table("arc_comparisons")
    op.drop_table("episode_arc_membership")
    op.drop_table("arcs")
    op.drop_table("taxonomies")

    # Drop enums
    op.execute("DROP TYPE IF EXISTS taxonomy_type")
    op.execute("DROP TYPE IF EXISTS taxonomy_status")
