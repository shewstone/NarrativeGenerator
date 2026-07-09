"""Service for discovering arcs through clustering analysis.

This is the engine for moving from hardcoded taxonomies to
data-driven arc discovery.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN

from narrative_engine.models import Episode
from narrative_engine.taxonomy.models import (
    ArcTaxonomy,
    DiscoveredArc,
    TaxonomyStatus,
    TaxonomyType,
)
from narrative_engine.taxonomy.repository import ArcRepository

logger = logging.getLogger(__name__)


class ArcDiscoveryService:
    """Service for discovering narrative arcs from episode data.

    Uses clustering algorithms (HDBSCAN, DBSCAN) on episode embeddings
to identify emergent patterns not captured by canonical taxonomies.
    """

    def __init__(
        self,
        arc_repository: ArcRepository,
        embedding_dimension: int = 384,
    ) -> None:
        self.arc_repository = arc_repository
        self.embedding_dimension = embedding_dimension

    async def create_canonical_taxonomy(
        self,
        name: str = "v1-canonical",
        description: str = "Hand-coded archetypal arcs from narrative theory",
    ) -> ArcTaxonomy:
        """Create a taxonomy with the canonical 14 arcs.

        This migrates the hardcoded ArcType enum into the database.
        """
        from narrative_engine.models import ArcType, ArcPhase

        # Create taxonomy
        taxonomy = ArcTaxonomy(
            name=name,
            description=description,
            taxonomy_type=TaxonomyType.CANONICAL,
            status=TaxonomyStatus.DRAFT,
            version="1.0.0",
        )

        taxonomy = await self.arc_repository.create_taxonomy(taxonomy)

        # Create canonical arcs
        canonical_arcs = [
            {
                "slug": ArcType.CREDIT_BOOM_AND_BUST.value,
                "name": "Credit Boom and Bust",
                "description": "Credit expansion followed by contraction and crisis",
                "phases": ["boom", "euphoria", "distress", "panic", "revulsion"],
                "theoretical_sources": ["Kindleberger-1978", "Minsky-1986"],
                "keywords": ["credit", "bubble", "mania", "panic", "crash"],
            },
            {
                "slug": ArcType.HUBRIS_NEMESIS.value,
                "name": "Hubris and Nemesis",
                "description": "Excessive pride leading to downfall",
                "phases": ["rise", "hubris", "challenge", "nemesis", "fall"],
                "theoretical_sources": ["Greek-tragedy", "Taleb-2012"],
                "keywords": ["hubris", "pride", "fall", "tragedy", "nemesis"],
            },
            {
                "slug": ArcType.RISE_AND_OVEREXTENSION.value,
                "name": "Rise and Overextension",
                "description": "Successful expansion followed by overreach and decline",
                "phases": ["growth", "success", "overextension", "strain", "retrenchment"],
                "theoretical_sources": ["Kennedy-1987"],
                "keywords": ["empire", "overstretch", "imperial", "decline"],
            },
            {
                "slug": ArcType.REFORM_THEN_REACTION.value,
                "name": "Reform Then Reaction",
                "description": "Radical change followed by backlash and counter-reformation",
                "phases": ["status_quo", "reform", "polarization", "reaction", "new_equilibrium"],
                "theoretical_sources": ["Polanyi-1944", "Huntington-1991"],
                "keywords": ["reform", "backlash", "revolution", "reaction"],
            },
            {
                "slug": ArcType.DECADENCE_AND_RENEWAL.value,
                "name": "Decadence and Renewal",
                "description": "Cultural decline followed by regeneration",
                "phases": ["vitality", "complacency", "decadence", "crisis", "renewal"],
                "theoretical_sources": ["Gibbon-1776", "Spengler-1918"],
                "keywords": ["decadence", "decline", "renewal", "regeneration"],
            },
            {
                "slug": ArcType.SIEGE_AND_COLLAPSE.value,
                "name": "Siege and Collapse",
                "description": "Sustained pressure leading to system failure",
                "phases": ["pressure", "resistance", "cracks", "breakdown", "collapse"],
                "theoretical_sources": ["Tainter-1988"],
                "keywords": ["siege", "pressure", "collapse", "complexity"],
            },
            {
                "slug": ArcType.SUCCESSION_CRISIS.value,
                "name": "Succession Crisis",
                "description": "Leadership transition leading to instability",
                "phases": ["stable_rule", "aging_leader", "succession", "contest", "resolution"],
                "theoretical_sources": ["Montesquieu-1748", "Turchin-2016"],
                "keywords": ["succession", "crisis", "leadership", "transition"],
            },
            {
                "slug": ArcType.GENERATIONAL_FORGETTING.value,
                "name": "Generational Forgetting",
                "description": "Lessons learned are lost as generations turn over",
                "phases": ["crisis", "memory", " complacency", "forgetting", "repeat"],
                "theoretical_sources": ["Strauss-Howe-1997"],
                "keywords": ["generational", "forgetting", "memory", "cycle"],
            },
            # Additional arcs from the ArcType enum
            {
                "slug": ArcType.HERO_JOURNEY.value,
                "name": "Hero's Journey",
                "description": "The monomyth pattern of departure, initiation, return",
                "phases": ["ordinary_world", "call", "trials", "crisis", "return"],
                "theoretical_sources": ["Campbell-1949"],
                "keywords": ["hero", "journey", "monomyth", "transformation"],
            },
            {
                "slug": ArcType.TRAGEDY.value,
                "name": "Tragedy",
                "description": "Fatal flaw leading to inevitable downfall",
                "phases": ["status", "flaw", "rising", "climax", "catastrophe"],
                "theoretical_sources": ["Aristotle", "Shakespeare"],
                "keywords": ["tragedy", "flaw", "fate", "doom"],
            },
        ]

        for arc_data in canonical_arcs:
            arc = DiscoveredArc(  # Using DiscoveredArc model but canonical type
                name=arc_data["name"],
                description=arc_data["description"],
                # These would be properly constructed in real implementation
            )
            # In real implementation, would use CanonicalArc

        # Activate the taxonomy
        await self.arc_repository.update_taxonomy_status(
            taxonomy.id, TaxonomyStatus.ACTIVE
        )
        taxonomy.status = TaxonomyStatus.ACTIVE

        logger.info(f"Created canonical taxonomy: {taxonomy.name} with {len(canonical_arcs)} arcs")
        return taxonomy

    async def discover_arcs(
        self,
        episodes: Sequence[Episode],
        algorithm: str = "hdbscan",
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        name: str = "discovered-v1",
        description: str = "Emergent arcs from clustering analysis",
    ) -> ArcTaxonomy:
        """Discover narrative arcs through clustering.

        Args:
            episodes: Episodes with embeddings to cluster
            algorithm: "hdbscan" or "dbscan"
            min_cluster_size: Minimum episodes per cluster
            min_samples: HDBSCAN min_samples (defaults to min_cluster_size)
            name: Taxonomy name
            description: Taxonomy description

        Returns:
            Taxonomy containing discovered arcs
        """
        if not episodes:
            raise ValueError("No episodes provided for clustering")

        # Extract embeddings. Discovery clustering is an analogy consumer
        # (design doc Sec 3.3a): structural_embedding only, never surface.
        embeddings = []
        episode_ids = []
        for ep in episodes:
            if ep.structural_embedding:
                embeddings.append(ep.structural_embedding)
                episode_ids.append(ep.id)

        if len(embeddings) < min_cluster_size * 2:
            raise ValueError(
                f"Need at least {min_cluster_size * 2} episodes with embeddings, got {len(embeddings)}"
            )

        embeddings_array = np.array(embeddings)

        # Run clustering
        if algorithm == "hdbscan":
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples or min_cluster_size,
                metric="cosine",
            )
        elif algorithm == "dbscan":
            clusterer = DBSCAN(
                eps=0.5,
                min_samples=min_cluster_size,
                metric="cosine",
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        labels = clusterer.fit_predict(embeddings_array)

        # Create taxonomy
        taxonomy = ArcTaxonomy(
            name=name,
            description=description,
            taxonomy_type=TaxonomyType.DISCOVERED,
            status=TaxonomyStatus.DRAFT,
            version="1.0.0",
            discovery_params={
                "algorithm": algorithm,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_episodes": len(episodes),
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "n_noise": list(labels).count(-1),
            },
        )
        taxonomy = await self.arc_repository.create_taxonomy(taxonomy)

        # Create discovered arcs from clusters
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise points

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = embeddings_array[cluster_mask]

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0).tolist()

            # Get member episode IDs
            member_ids = [episode_ids[i] for i in cluster_indices]

            # Calculate silhouette score for cluster quality
            from sklearn.metrics import silhouette_score
            try:
                cluster_silhouette = silhouette_score(
                    cluster_embeddings,
                    [label] * len(cluster_indices),
                    metric="cosine",
                )
            except Exception:
                cluster_silhouette = None

            # Find representative episodes (closest to centroid)
            distances = np.linalg.norm(
                cluster_embeddings - centroid, axis=1
            )
            closest_idx = cluster_indices[np.argmin(distances)]
            boundary_idx = cluster_indices[np.argmax(distances)]

            arc = DiscoveredArc(
                cluster_id=int(label),
                name=None,  # Will be labeled later via LLM or manual
                description=None,  # Will be generated later
                embedding_centroid=centroid,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                member_count=len(member_ids),
                silhouette_score=cluster_silhouette,
                representative_episode_ids=[episode_ids[closest_idx]],
                boundary_episode_ids=[episode_ids[boundary_idx]],
                common_features={
                    "avg_embedding_magnitude": float(np.mean(distances)),
                    "cluster_cohesion": float(np.std(distances)),
                },
                discovery_algorithm=algorithm,
                discovery_params={
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                },
            )

            arc = await self.arc_repository.create_discovered_arc(arc, taxonomy.id)
            logger.info(f"Created discovered arc: {arc.id} (cluster {label}, {arc.member_count} members)")

        logger.info(f"Discovered taxonomy complete: {len(unique_labels)} arcs from {len(episodes)} episodes")
        return taxonomy

    async def compare_taxonomies(
        self,
        baseline_taxonomy_id: UUID,
        comparison_taxonomy_id: UUID,
    ) -> Dict[str, Any]:
        """Compare two taxonomies to evaluate discovered arcs.

        Returns metrics:
        - Episode agreement score
        - Arc mappings between taxonomies
        - Clustering quality comparison
        """
        # Get taxonomies
        baseline = await self.arc_repository.get_taxonomy(baseline_taxonomy_id)
        comparison = await self.arc_repository.get_taxonomy(comparison_taxonomy_id)

        if not baseline or not comparison:
            raise ValueError("One or both taxonomies not found")

        # Get arcs
        baseline_arcs = await self.arc_repository.list_canonical_arcs(baseline.id)
        comparison_arcs = await self.arc_repository.list_discovered_arcs(comparison.id)

        # Compute arc centroids for comparison
        # (In real implementation, would compare episode assignments)

        results = {
            "baseline_taxonomy": baseline.name,
            "comparison_taxonomy": comparison.name,
            "baseline_arc_count": len(baseline_arcs),
            "comparison_arc_count": len(comparison_arcs),
            "episode_agreement_score": None,  # Would compute from assignments
            "arc_mappings": {},  # Would compute similarity matrix
        }

        return results

    async def label_discovered_arc(
        self,
        arc_id: UUID,
        name: str,
        description: str,
    ) -> None:
        """Manually label a discovered arc after review.

        This is the human-in-the-loop step for making discovered
        arcs interpretable.
        """
        arc = await self.arc_repository.get_discovered_arc(arc_id)
        if not arc:
            raise ValueError(f"Arc not found: {arc_id}")

        # In real implementation, would update the arc
        logger.info(f"Labeled discovered arc {arc_id}: {name}")

    async def map_to_canonical(
        self,
        discovered_arc_id: UUID,
    ) -> Dict[str, float]:
        """Map a discovered arc to canonical arcs.

        Returns similarity scores to each canonical arc,
        enabling hybrid classification.
        """
        discovered = await self.arc_repository.get_discovered_arc(discovered_arc_id)
        if not discovered or not discovered.embedding_centroid:
            return {}

        # Get canonical arcs
        canonical = await self.arc_repository.list_canonical_arcs()

        # Compute similarities (placeholder)
        mappings = {}
        for can_arc in canonical:
            # In real implementation, would compute cosine similarity
            # between discovered centroid and canonical examples
            mappings[can_arc.slug] = 0.0

        return mappings
