"""Repository for taxonomy operations."""

from __future__ import annotations

from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from narrative_engine.taxonomy.models import (
    ArcMembership,
    ArcTaxonomy,
    CanonicalArc,
    DiscoveredArc,
    TaxonomyStatus,
)
from narrative_engine.taxonomy.orm_models import (
    ArcORM,
    ArcTaxonomyORM,
)


class ArcRepository:
    """Repository for arc and taxonomy operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ===== Taxonomy Operations =====

    async def create_taxonomy(self, taxonomy: ArcTaxonomy) -> ArcTaxonomy:
        """Create a new taxonomy."""
        orm = ArcTaxonomyORM(
            id=taxonomy.id,
            name=taxonomy.name,
            description=taxonomy.description,
            taxonomy_type=taxonomy.taxonomy_type,
            status=taxonomy.status,
            version=taxonomy.version,
            parent_taxonomy_id=taxonomy.parent_taxonomy_id,
            created_by=taxonomy.created_by,
            discovery_params=taxonomy.discovery_params,
        )
        self.session.add(orm)
        await self.session.flush()
        return taxonomy

    async def get_taxonomy(self, taxonomy_id: UUID) -> Optional[ArcTaxonomy]:
        """Get taxonomy by ID."""
        result = await self.session.execute(
            select(ArcTaxonomyORM).where(ArcTaxonomyORM.id == taxonomy_id)
        )
        orm = result.scalar_one_or_none()
        return self._taxonomy_from_orm(orm) if orm else None

    async def get_taxonomy_by_name(self, name: str) -> Optional[ArcTaxonomy]:
        """Get taxonomy by name."""
        result = await self.session.execute(
            select(ArcTaxonomyORM).where(ArcTaxonomyORM.name == name)
        )
        orm = result.scalar_one_or_none()
        return self._taxonomy_from_orm(orm) if orm else None

    async def get_active_taxonomy(self) -> Optional[ArcTaxonomy]:
        """Get the currently active taxonomy."""
        result = await self.session.execute(
            select(ArcTaxonomyORM)
            .where(ArcTaxonomyORM.status == TaxonomyStatus.ACTIVE)
            .order_by(ArcTaxonomyORM.created_at.desc())
        )
        orm = result.scalar_one_or_none()
        return self._taxonomy_from_orm(orm) if orm else None

    async def list_taxonomies(
        self,
        status: Optional[TaxonomyStatus] = None,
        limit: int = 100,
    ) -> Sequence[ArcTaxonomy]:
        """List taxonomies, optionally filtered by status."""
        query = select(ArcTaxonomyORM)
        if status:
            query = query.where(ArcTaxonomyORM.status == status)
        query = query.order_by(ArcTaxonomyORM.created_at.desc()).limit(limit)

        result = await self.session.execute(query)
        return [self._taxonomy_from_orm(orm) for orm in result.scalars().all()]

    async def update_taxonomy_status(
        self, taxonomy_id: UUID, status: TaxonomyStatus
    ) -> None:
        """Update taxonomy status."""
        result = await self.session.execute(
            select(ArcTaxonomyORM).where(ArcTaxonomyORM.id == taxonomy_id)
        )
        orm = result.scalar_one()
        orm.status = status
        await self.session.flush()

    # ===== Canonical Arc Operations =====

    async def create_canonical_arc(
        self, arc: CanonicalArc, taxonomy_id: UUID
    ) -> CanonicalArc:
        """Create a canonical arc."""
        orm = ArcORM(
            id=arc.id,
            arc_type="canonical",
            slug=arc.slug,
            name=arc.name,
            description=arc.description,
            phases=arc.phases,
            theoretical_sources=arc.theoretical_sources,
            keywords=arc.keywords,
            example_episodes=arc.example_episodes,
            taxonomy_id=taxonomy_id,
        )
        self.session.add(orm)
        await self.session.flush()
        return arc

    async def get_canonical_arc(self, arc_id: UUID) -> Optional[CanonicalArc]:
        """Get canonical arc by ID."""
        result = await self.session.execute(
            select(ArcORM)
            .where(ArcORM.id == arc_id)
            .where(ArcORM.arc_type == "canonical")
        )
        orm = result.scalar_one_or_none()
        return self._canonical_arc_from_orm(orm) if orm else None

    async def get_canonical_arc_by_slug(self, slug: str) -> Optional[CanonicalArc]:
        """Get canonical arc by slug."""
        result = await self.session.execute(
            select(ArcORM)
            .where(ArcORM.slug == slug)
            .where(ArcORM.arc_type == "canonical")
        )
        orm = result.scalar_one_or_none()
        return self._canonical_arc_from_orm(orm) if orm else None

    async def list_canonical_arcs(
        self, taxonomy_id: Optional[UUID] = None
    ) -> Sequence[CanonicalArc]:
        """List canonical arcs, optionally filtered by taxonomy."""
        query = select(ArcORM).where(ArcORM.arc_type == "canonical")
        if taxonomy_id:
            query = query.where(ArcORM.taxonomy_id == taxonomy_id)

        result = await self.session.execute(query)
        return [self._canonical_arc_from_orm(orm) for orm in result.scalars().all()]

    # ===== Discovered Arc Operations =====

    async def create_discovered_arc(
        self, arc: DiscoveredArc, taxonomy_id: UUID
    ) -> DiscoveredArc:
        """Create a discovered arc."""
        orm = ArcORM(
            id=arc.id,
            arc_type="discovered",
            name=arc.name or f"Cluster_{arc.cluster_id}",
            description=arc.description,
            cluster_id=arc.cluster_id,
            member_count=arc.member_count,
            silhouette_score=arc.silhouette_score,
            representative_episode_ids=arc.representative_episode_ids,
            boundary_episode_ids=arc.boundary_episode_ids,
            common_features=arc.common_features,
            canonical_arc_mappings=arc.canonical_arc_mappings,
            discovered_at=arc.discovered_at,
            discovery_algorithm=arc.discovery_algorithm,
            discovery_params=arc.discovery_params,
            taxonomy_id=taxonomy_id,
        )
        if arc.embedding_centroid:
            orm.embedding_centroid = arc.embedding_centroid
        if arc.embedding_model:
            orm.embedding_model = arc.embedding_model

        self.session.add(orm)
        await self.session.flush()
        return arc

    async def get_discovered_arc(self, arc_id: UUID) -> Optional[DiscoveredArc]:
        """Get discovered arc by ID."""
        result = await self.session.execute(
            select(ArcORM)
            .where(ArcORM.id == arc_id)
            .where(ArcORM.arc_type == "discovered")
        )
        orm = result.scalar_one_or_none()
        return self._discovered_arc_from_orm(orm) if orm else None

    async def list_discovered_arcs(
        self, taxonomy_id: Optional[UUID] = None
    ) -> Sequence[DiscoveredArc]:
        """List discovered arcs, optionally filtered by taxonomy."""
        query = select(ArcORM).where(ArcORM.arc_type == "discovered")
        if taxonomy_id:
            query = query.where(ArcORM.taxonomy_id == taxonomy_id)

        result = await self.session.execute(query)
        return [self._discovered_arc_from_orm(orm) for orm in result.scalars().all()]

    # ===== Arc Membership Operations =====

    async def assign_arc_membership(self, membership: ArcMembership) -> None:
        """Assign an episode to an arc with membership score."""
        from narrative_engine.taxonomy.orm_models import episode_arc_membership

        await self.session.execute(
            episode_arc_membership.insert().values(
                episode_id=membership.episode_id,
                arc_id=membership.arc_id,
                arc_type=membership.arc_type,
                membership_score=membership.membership_score,
                confidence=membership.confidence,
                phase=membership.phase,
                phase_confidence=membership.phase_confidence,
                assignment_method=membership.assignment_method,
                rationale=membership.rationale,
                distance_to_centroid=membership.distance_to_centroid,
                taxonomy_id=membership.taxonomy_id,
                assigned_by=membership.assigned_by,
            )
        )
        await self.session.flush()

    async def get_episode_arcs(
        self, episode_id: UUID, taxonomy_id: Optional[UUID] = None
    ) -> Sequence[ArcMembership]:
        """Get all arc memberships for an episode."""
        from narrative_engine.taxonomy.orm_models import episode_arc_membership

        query = select(episode_arc_membership).where(
            episode_arc_membership.c.episode_id == episode_id
        )
        if taxonomy_id:
            query = query.where(
                episode_arc_membership.c.taxonomy_id == taxonomy_id
            )

        result = await self.session.execute(query)
        return [
            ArcMembership(
                episode_id=row.episode_id,
                arc_id=row.arc_id,
                arc_type=row.arc_type,
                membership_score=row.membership_score,
                confidence=row.confidence,
                phase=row.phase,
                phase_confidence=row.phase_confidence,
                assignment_method=row.assignment_method,
                rationale=row.rationale,
                distance_to_centroid=row.distance_to_centroid,
                taxonomy_id=row.taxonomy_id,
                assigned_at=row.assigned_at,
                assigned_by=row.assigned_by,
            )
            for row in result.all()
        ]

    # ===== Conversion Helpers =====

    def _taxonomy_from_orm(self, orm: ArcTaxonomyORM) -> ArcTaxonomy:
        """Convert ORM to Pydantic model."""
        return ArcTaxonomy(
            id=orm.id,
            name=orm.name,
            description=orm.description,
            taxonomy_type=orm.taxonomy_type,
            status=orm.status,
            version=orm.version,
            parent_taxonomy_id=orm.parent_taxonomy_id,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
            created_by=orm.created_by,
            discovery_params=orm.discovery_params,
        )

    def _canonical_arc_from_orm(self, orm: ArcORM) -> CanonicalArc:
        """Convert ORM to CanonicalArc Pydantic model."""
        return CanonicalArc(
            id=orm.id,
            slug=orm.slug or "",
            name=orm.name,
            description=orm.description or "",
            phases=orm.phases,
            theoretical_sources=orm.theoretical_sources,
            keywords=orm.keywords,
            example_episodes=orm.example_episodes,
            taxonomy_ids=[orm.taxonomy_id] if orm.taxonomy_id else [],
            created_at=orm.created_at,
            updated_at=orm.updated_at,
        )

    def _discovered_arc_from_orm(self, orm: ArcORM) -> DiscoveredArc:
        """Convert ORM to DiscoveredArc Pydantic model."""
        return DiscoveredArc(
            id=orm.id,
            cluster_id=orm.cluster_id or 0,
            name=orm.name if orm.name and not orm.name.startswith("Cluster_") else None,
            description=orm.description,
            embedding_centroid=orm.embedding_centroid,
            embedding_model=orm.embedding_model,
            member_count=orm.member_count,
            silhouette_score=orm.silhouette_score,
            representative_episode_ids=orm.representative_episode_ids,
            boundary_episode_ids=orm.boundary_episode_ids,
            common_features=orm.common_features,
            canonical_arc_mappings=orm.canonical_arc_mappings,
            taxonomy_ids=[orm.taxonomy_id] if orm.taxonomy_id else [],
            discovered_at=orm.discovered_at,
            discovery_algorithm=orm.discovery_algorithm,
            discovery_params=orm.discovery_params,
        )
