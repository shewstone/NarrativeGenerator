"""Composition pipeline for building Arc Instances from episodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from narrative_engine.composition.arc_instance import (
    ArcInstance,
)
from narrative_engine.composition.identity import (
    ArcIdentityResolver,
    _episode_date_key,
    _episode_end_year,
    _episode_start_year,
)
from narrative_engine.models import (
    ArcPhase,
    ArcType,
    Cycle,
    CycleMembership,
    CycleScale,
    Episode,
    LinkStatus,
    ReviewStatus,
)
from narrative_engine.storage.orm_models import (
    CycleMembershipORM,
    CycleORM,
    EpisodeORM,
)
from narrative_engine.storage.repositories import CycleMembershipRepository, CycleRepository


@dataclass
class CompositionConfig:
    """Configuration for the composition pipeline."""

    # Cycle scale this composition pass targets. Arc instances are, by
    # design doc convention (Sec 2), implemented as episodic-scale cycles --
    # this selects which per-scale temporal gap threshold applies
    # (Sec 6.2 stage 6 item 4). Overridable per ArcDefinition in principle;
    # here it's a pipeline-level default.
    scale: CycleScale = CycleScale.EPISODIC
    # NOT used to drop episodes during clustering (every episode ends up in
    # some instance -- see _cluster_within_scope). Reserved for the
    # persistence layer's review_status size gate (Sec 6.2 stage 6
    # guardrails: "instances above a size threshold get review_status
    # pending").
    min_episodes_per_cluster: int = 2

    # Phase matching parameters
    phase_confidence_threshold: float = 0.5
    allow_phase_gaps: bool = True

    # Coverage thresholds
    min_coverage_per_phase: float = 0.3
    complete_coverage_threshold: float = 0.8


def _infer_expected_phases(arc_type: ArcType) -> List[ArcPhase]:
    """Infer expected phases from arc type.

    This would ideally come from ArcDefinition, but for now
    we use sensible defaults.
    """
    # Financial/economic arcs
    if arc_type == ArcType.CREDIT_BOOM_AND_BUST:
        return [
            ArcPhase.BOOM,
            ArcPhase.EUPHORIA,
            ArcPhase.DISTRESS,
            ArcPhase.PANIC,
            ArcPhase.REVULSION,
        ]
    # Every non-financial legacy interpretation uses the standard episode
    # phase vocabulary. Returning every enum value here previously inserted
    # boom/panic/revulsion gaps into political and personal arcs.
    return [
        ArcPhase.SETUP,
        ArcPhase.RISING_ACTION,
        ArcPhase.CLIMAX,
        ArcPhase.FALLING_ACTION,
        ArcPhase.RESOLUTION,
    ]


def _build_instance_from_cluster(
    arc_type: ArcType,
    cluster: List[Episode],
    expected_phases: Optional[List[ArcPhase]],
    scale: CycleScale,
) -> Optional[ArcInstance]:
    """Build an Arc Instance from a cluster of (already-matched) episodes."""
    if not cluster:
        return None

    first_ep = cluster[0]
    last_ep = cluster[-1]

    location = first_ep.location or "Unknown Location"
    start_year = _episode_start_year(first_ep)
    end_year = _episode_end_year(last_ep)

    def year_label(year: Optional[int]) -> str:
        if year is None:
            return "?"
        return f"{abs(year)} BCE" if year < 0 else str(year)

    from narrative_engine.scopes import resolve_scope

    raw_scope = first_ep.scope_id or first_ep.scope_name
    # Partitioning already resolves registered aliases; persist that same
    # canonical ID so the dashboard can group and traverse the hierarchy.
    # Unregistered labels remain explicit instead of collapsing to None.
    focal_scope = resolve_scope(raw_scope) or raw_scope
    scope_label = first_ep.scope_name or location
    canonical_name = f"{arc_type.value}, {scope_label}, {year_label(start_year)}–{year_label(end_year)}"

    instance = ArcInstance(
        arc_type=arc_type,
        canonical_name=canonical_name,
        scope_id=focal_scope,
        scale=scale,
        start_date=first_ep.start_date,
        end_date=last_ep.end_date,
    )

    for episode in cluster:
        if episode.arc_phase:
            source_ids = episode.extracted_from if episode.extracted_from else ["unknown"]
            source_id = source_ids[0] if source_ids else "unknown"

            # add_episode_to_phase maintains instance.source_coverage
            instance.add_episode_to_phase(
                phase=episode.arc_phase,
                episode_id=episode.id,
                source_id=source_id,
                confidence=episode.phase_confidence or 0.5,
                date=episode.start_date,
            )
        else:
            instance.unphased_episode_ids.append(episode.id)

    if expected_phases:
        instance.identify_gaps(expected_phases)
    else:
        instance.identify_gaps(_infer_expected_phases(arc_type))

    instance.update_status()

    return instance


def _infer_composition_scale(
    episodes: Sequence[Episode],
    fallback: CycleScale,
) -> CycleScale:
    """Choose identity timing from extracted scale, subject kind, and span."""
    situation_mapping = {
        "personal": CycleScale.EPISODIC,
        "interpersonal": CycleScale.EPISODIC,
        "group": CycleScale.INSTITUTIONAL,
        "organization": CycleScale.INSTITUTIONAL,
        "institution": CycleScale.INSTITUTIONAL,
        "polity": CycleScale.GENERATIONAL,
        "civilization": CycleScale.CIVILIZATIONAL,
        "system": CycleScale.CIVILIZATIONAL,
    }
    kind_mapping = {
        "person": CycleScale.EPISODIC,
        "dyad": CycleScale.EPISODIC,
        "group": CycleScale.INSTITUTIONAL,
        "faction": CycleScale.INSTITUTIONAL,
        "organization": CycleScale.INSTITUTIONAL,
        "institution": CycleScale.INSTITUTIONAL,
        "party": CycleScale.GENERATIONAL,
        "movement": CycleScale.GENERATIONAL,
        "polity": CycleScale.GENERATIONAL,
        "civilization": CycleScale.CIVILIZATIONAL,
        "region": CycleScale.CIVILIZATIONAL,
        "system": CycleScale.CIVILIZATIONAL,
    }
    votes: Dict[CycleScale, int] = {}
    scale_rank = {
        CycleScale.EPISODIC: 0,
        CycleScale.INSTITUTIONAL: 1,
        CycleScale.GENERATIONAL: 2,
        CycleScale.CIVILIZATIONAL: 3,
    }
    for episode in episodes:
        situation = (
            episode.situation_scale.value
            if hasattr(episode.situation_scale, "value")
            else episode.situation_scale
        )
        scope_kind = episode.scope_kind.value if hasattr(episode.scope_kind, "value") else episode.scope_kind
        candidates = [
            candidate
            for candidate in (
                situation_mapping.get(situation),
                kind_mapping.get(scope_kind),
            )
            if candidate is not None
        ]
        inferred = max(candidates, key=scale_rank.get) if candidates else None
        if inferred:
            votes[inferred] = votes.get(inferred, 0) + 1

    scale = max(votes, key=votes.get) if votes else fallback
    years = [
        year
        for episode in episodes
        for year in (_episode_start_year(episode), _episode_end_year(episode))
        if year is not None
    ]
    if years:
        span = max(years) - min(years)
        if span >= 80:
            scale = CycleScale.CIVILIZATIONAL
        elif span >= 20 and scale in {CycleScale.EPISODIC, CycleScale.INSTITUTIONAL}:
            scale = CycleScale.GENERATIONAL
        elif span >= 5 and scale == CycleScale.EPISODIC:
            scale = CycleScale.INSTITUTIONAL
    return scale


def _cluster_within_scope(
    episodes: List[Episode],
    resolver: ArcIdentityResolver,
    scale: CycleScale,
) -> List[List[Episode]]:
    """Sequentially merge temporally-sorted episodes within a single scope.

    Adjacent episodes merge into the current cluster only if they pass the
    resolver's full staged gate cascade (actor overlap, per-scale temporal
    threshold, arc_type agreement, phase-sequence continuity -- design doc
    Sec 6.2 stage 6, items 2, 3-4, 5). Any failed gate closes the current
    cluster and starts a new one, which is what keeps near-miss decoys
    (same scope, same arc_type, temporally close, but a different
    unfolding) from merging.

    Every episode ends up in exactly one cluster, including size-1 clusters
    for episodes that don't match anything adjacent: silently dropping
    episodes that fail to compose would hide them from the arc-instance
    layer entirely, defeating the point of tracking phase_coverage gaps
    (Sec 6.2 stage 6: "making documentation gaps in an instance visible").
    """
    sorted_eps = sorted(episodes, key=_episode_date_key)

    clusters: List[List[Episode]] = []
    current: List[Episode] = []

    for episode in sorted_eps:
        if not current:
            current = [episode]
            continue

        previous = current[-1]
        score = resolver.calculate_identity_score(previous, episode, scale=scale.value)

        if score.is_match:
            current.append(episode)
        else:
            clusters.append(current)
            current = [episode]

    if current:
        clusters.append(current)

    return clusters


def compose_arc_instances_from_episodes(
    episodes: Sequence[Episode],
    arc_type: ArcType,
    resolver: Optional[ArcIdentityResolver] = None,
    scale: CycleScale = CycleScale.EPISODIC,
    expected_phases: Optional[List[ArcPhase]] = None,
) -> List[ArcInstance]:
    """Pure, DB-free implementation of the design doc's Sec 6.2 stage 6
    identity-resolution pipeline, applied to arc-instance composition:

        1. Hard filter: partition by scope_id.
        2. Hard-ish filter: actor-entity overlap above threshold.
        3. Soft signal: surface-embedding similarity.
        4. Per-scale temporal gap threshold + arc_type agreement.
        5. Phase-sequence continuity check.

    Every input episode ends up in exactly one output instance (including
    size-1 instances for episodes that don't compose with anything else):
    see _cluster_within_scope for why nothing is silently dropped here.
    Filtering out trivial/low-value instances, if wanted, belongs at the
    persistence layer (e.g. by review_status gating), not in the matching
    algorithm itself.

    Operating on plain Episode objects (no DB session) is what makes this
    testable directly against the composition fixture (Sec 6.6) and usable
    as a thin-wrapper target from CompositionPipeline.compose_arc_instances.
    """
    resolver = resolver or ArcIdentityResolver()

    relevant = [
        e
        for e in episodes
        if e.arc_type == arc_type and (e.start_date is not None or e.start_year is not None)
    ]

    # Stage 1: hard filter -- partition by scope_id. Episodes in different
    # scopes are never compared, by construction. Episodes with NO scope are
    # excluded from merging entirely (each becomes its own singleton
    # partition): pooling them together would put, say, an unscoped Japanese
    # and an unscoped US credit boom in the same candidate pool -- exactly
    # the false-merge class the scope partition exists to eliminate. They
    # still surface as size-1 instances rather than being silently dropped.
    # Raw scope labels are normalized through the versioned alias registry
    # (T5) so "US" and "United States" land in ONE partition instead of two
    # (false-split protection). Unresolved labels keep their raw string as
    # the partition key: distinct unknown labels never merge with each
    # other, which preserves the false-merge protection this filter is for.
    from narrative_engine.scopes import DEFAULT_SCOPE_CONFIDENCE_FLOOR, scope_partition_key

    by_scope: Dict[str, List[Episode]] = {}
    unscoped: List[Episode] = []
    for episode in relevant:
        # A low-confidence scope claim is retained for audit but may not
        # participate in a hard identity partition.
        if episode.scope_confidence is not None and episode.scope_confidence < DEFAULT_SCOPE_CONFIDENCE_FLOOR:
            unscoped.append(episode)
        else:
            partition_key = scope_partition_key(episode.scope_id, episode.scope_name)
            if partition_key is None:
                unscoped.append(episode)
            else:
                by_scope.setdefault(partition_key, []).append(episode)

    instances: List[ArcInstance] = []
    for scope_episodes in by_scope.values():
        inferred_scale = _infer_composition_scale(scope_episodes, scale)
        clusters = _cluster_within_scope(scope_episodes, resolver, inferred_scale)
        for cluster in clusters:
            instance = _build_instance_from_cluster(
                arc_type,
                cluster,
                expected_phases,
                inferred_scale,
            )
            if instance:
                instances.append(instance)

    for episode in unscoped:
        inferred_scale = _infer_composition_scale([episode], scale)
        instance = _build_instance_from_cluster(
            arc_type,
            [episode],
            expected_phases,
            inferred_scale,
        )
        if instance:
            instances.append(instance)

    return instances


class CompositionPipeline:
    """Pipeline for composing Arc Instances from extracted episodes.

    This is the key innovation: stitching phase-adjacent episodes
    across sources to build concrete arc unfoldings.

    Example:
        Book 1 covers phases 1-2 of HUBRIS_NEMESIS
        Book 2 covers phases 3-5 of HUBRIS_NEMESIS
        CompositionPipeline creates unified Arc Instance

    The actual matching/clustering algorithm lives in the module-level pure
    function compose_arc_instances_from_episodes; this class is a thin
    DB-fetching wrapper around it (fetch EpisodeORM rows -> convert ->
    delegate -> return).
    """

    def __init__(
        self,
        session: AsyncSession,
        config: Optional[CompositionConfig] = None,
        resolver: Optional[ArcIdentityResolver] = None,
    ):
        self.session = session
        self.config = config or CompositionConfig()
        self.resolver = resolver or ArcIdentityResolver()

    async def compose_arc_instances(
        self,
        arc_type: ArcType,
        expected_phases: Optional[List[ArcPhase]] = None,
    ) -> List[ArcInstance]:
        """Build Arc Instances for a given arc type.

        Args:
            arc_type: The arc type to compose instances for
            expected_phases: Expected phases in order (if known)

        Returns:
            List of composed Arc Instances
        """
        episode_orms = await self._get_episodes_by_arc(arc_type)

        if not episode_orms:
            return []

        episodes = [self.resolver.episode_from_orm(orm) for orm in episode_orms]

        return compose_arc_instances_from_episodes(
            episodes=episodes,
            arc_type=arc_type,
            resolver=self.resolver,
            scale=self.config.scale,
            expected_phases=expected_phases,
        )

    async def persist_instance(self, instance: ArcInstance, arc_type: ArcType) -> Cycle:
        """Persist a composed ArcInstance as an is_arc_instance Cycle plus
        per-episode CycleMembership rows (design doc Sec 6.2 stage 6).

        Composition has no textual evidence, so every membership is
        link_status=INFERRED; review_status follows the size-threshold
        guardrail ("instances above a size threshold get review_status
        pending"). This path never touches EpisodeLinkORM, so it
        structurally cannot emit a CAUSES edge -- the invariant (Sec 4)
        holds by construction, not just by validation.
        """
        phase_order = _infer_expected_phases(arc_type)

        cycle = Cycle(
            name=instance.canonical_name,
            scale=instance.scale,
            scope_id=instance.scope_id,
            start_date=instance.start_date,
            end_date=instance.end_date,
            dominant_arc_types=[arc_type],
            is_arc_instance=True,
        )

        cycle_repo = CycleRepository(self.session)
        membership_repo = CycleMembershipRepository(self.session)

        await cycle_repo.create(cycle)

        all_episode_ids = {
            episode_id for coverage in instance.phases.values() for episode_id in coverage.episode_ids
        } | set(instance.unphased_episode_ids)
        review_status = (
            ReviewStatus.PENDING if len(all_episode_ids) > self.config.min_episodes_per_cluster else ReviewStatus.AUTO
        )

        for phase, coverage in instance.phases.items():
            try:
                phase_index = phase_order.index(phase)
            except ValueError:
                phase_index = -1

            for episode_id in coverage.episode_ids:
                await membership_repo.create(
                    CycleMembership(
                        episode_id=episode_id,
                        cycle_id=cycle.id,
                        phase_coverage=[phase_index] if phase_index >= 0 else [],
                        link_status=LinkStatus.INFERRED,
                        review_status=review_status,
                    )
                )

        for episode_id in instance.unphased_episode_ids:
            await membership_repo.create(
                CycleMembership(
                    episode_id=episode_id,
                    cycle_id=cycle.id,
                    phase_coverage=[],
                    link_status=LinkStatus.INFERRED,
                    review_status=review_status,
                )
            )

        return cycle

    async def reconcile_instances(
        self,
        instances: Sequence[ArcInstance],
        arc_type: ArcType,
    ) -> List[Cycle]:
        """Persist the current composition without accumulating snapshots.

        Exact existing instances are reused, preserving their review state.
        Instances whose episode membership changed are replaced in the same
        transaction, and duplicate/stale snapshots from earlier runs are
        removed. This bounds database growth as the watcher recomposes an arc
        after each newly ingested document.
        """
        result = await self.session.execute(
            select(CycleORM).where(CycleORM.is_arc_instance.is_(True)).options(selectinload(CycleORM.episodes))
        )
        existing = [
            cycle
            for cycle in result.scalars().unique().all()
            if arc_type.value
            in {value.value if isinstance(value, ArcType) else value for value in (cycle.dominant_arc_types or [])}
        ]

        memberships_by_cycle: Dict = {}
        if existing:
            membership_result = await self.session.execute(
                select(CycleMembershipORM).where(CycleMembershipORM.cycle_id.in_([cycle.id for cycle in existing]))
            )
            for membership in membership_result.scalars().all():
                memberships_by_cycle.setdefault(membership.cycle_id, []).append(membership)

        by_fingerprint: Dict = {}
        for cycle in existing:
            fingerprint = (
                cycle.scope_id,
                frozenset(episode.id for episode in cycle.episodes),
            )
            by_fingerprint.setdefault(fingerprint, []).append(cycle)

        reused_ids = set()
        persisted: List[Cycle] = []
        cycle_repo = CycleRepository(self.session)
        phase_order = _infer_expected_phases(arc_type)

        for instance in instances:
            episode_ids = {
                episode_id for coverage in instance.phases.values() for episode_id in coverage.episode_ids
            } | set(instance.unphased_episode_ids)
            fingerprint = (instance.scope_id, frozenset(episode_ids))
            candidates = by_fingerprint.get(fingerprint, [])
            existing_cycle = next(
                (cycle for cycle in candidates if cycle.id not in reused_ids),
                None,
            )

            if existing_cycle is None:
                persisted.append(await self.persist_instance(instance, arc_type))
                continue

            reused_ids.add(existing_cycle.id)
            existing_cycle.name = instance.canonical_name
            existing_cycle.scale = instance.scale
            existing_cycle.scope_id = instance.scope_id
            existing_cycle.start_date = instance.start_date
            existing_cycle.end_date = instance.end_date
            existing_cycle.dominant_arc_types = [arc_type.value]

            phase_by_episode = {
                episode_id: [phase_order.index(phase)] if phase in phase_order else []
                for phase, coverage in instance.phases.items()
                for episode_id in coverage.episode_ids
            }
            for episode_id in instance.unphased_episode_ids:
                phase_by_episode[episode_id] = []
            for membership in memberships_by_cycle.get(existing_cycle.id, []):
                membership.phase_coverage = phase_by_episode.get(membership.episode_id, [])

            persisted.append(cycle_repo._from_orm(existing_cycle))

        for cycle in existing:
            if cycle.id not in reused_ids:
                await self.session.delete(cycle)

        await self.session.flush()
        return persisted

    async def _get_episodes_by_arc(self, arc_type: ArcType) -> Sequence[EpisodeORM]:
        """Fetch all episodes of a given arc type."""
        result = await self.session.execute(
            select(EpisodeORM)
            .where(EpisodeORM.arc_type == arc_type)
            .options(selectinload(EpisodeORM.source_passages), selectinload(EpisodeORM.actors))
            .order_by(EpisodeORM.start_date)
        )
        return result.scalars().unique().all()

    async def stitch_across_sources(
        self,
        instance: ArcInstance,
        gap_phases: List[ArcPhase],
    ) -> ArcInstance:
        """Attempt to fill gaps by searching other sources.

        This is the key "composition" step - if Book 1 has phases 1-2
        and Book 2 has phases 3-5, stitch them together.
        """
        for phase in gap_phases:
            # Query for episodes of this phase in same time period
            candidates = await self._find_gap_fillers(instance, phase)

            for candidate in candidates:
                source_ids = [sp.work_id for sp in candidate.source_passages if hasattr(sp, "work_id")]
                source_id = source_ids[0] if source_ids else "unknown"

                instance.add_episode_to_phase(
                    phase=phase,
                    episode_id=candidate.id,
                    source_id=source_id,
                    confidence=candidate.phase_confidence or 0.5,
                    date=candidate.start_date,
                )

        # Re-evaluate gaps
        instance.identify_gaps(list(instance.phases.keys()))
        instance.update_status()

        return instance

    async def _find_gap_fillers(self, instance: ArcInstance, phase: ArcPhase) -> Sequence[EpisodeORM]:
        """Find episodes that could fill a gap in an Arc Instance."""
        if not instance.start_date or not instance.end_date:
            return []

        # Scope is the stage-1 hard filter (Sec 6.2 stage 6 item 1) and gap
        # filling is still an identity decision, so it applies here too: an
        # instance without a scope cannot safely pull episodes from anywhere,
        # and a scoped instance may only pull from its own scope.
        if instance.scope_id is None:
            return []

        # Look for episodes of the same arc type, in the time window
        buffer = timedelta(days=365 * 2)  # 2 year buffer

        result = await self.session.execute(
            select(EpisodeORM)
            .where(EpisodeORM.scope_id == instance.scope_id)
            .where(EpisodeORM.arc_type == instance.arc_type)
            .where(EpisodeORM.arc_phase == phase)
            .where(EpisodeORM.start_date >= instance.start_date - buffer)
            .where(EpisodeORM.end_date <= instance.end_date + buffer)
            .order_by(EpisodeORM.phase_confidence.desc())
            .limit(5)
        )

        return result.scalars().all()
