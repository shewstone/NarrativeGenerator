"""Model <-> ORM round-trip property tests (T3).

Every persisted aggregate must survive create -> get_by_id with all fields
intact. The maximal builders explicitly set EVERY field not listed in the
repository's exclusion allowlist; the coverage check turns "someone added a
model field and forgot the converter" into a named unit-test failure instead
of a silent data-loss bug.

See docs/tickets/T3-model-orm-roundtrip-tests.md.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from narrative_engine.models import (
    Actor,
    ArcAssignment,
    ArcPhase,
    ArcType,
    ChangePattern,
    ClassificationState,
    Continuation,
    Cycle,
    CycleMembership,
    CycleScale,
    EdgeKind,
    Episode,
    EpisodeLink,
    LinkStatus,
    MechanismFamily,
    MechanismTag,
    ReviewStatus,
    Scope,
    ScopeKind,
    SituationConfiguration,
    SituationDomain,
    SituationScale,
    SourcePassage,
    Thesis,
    ThesisConfidence,
    ThesisMode,
)
from narrative_engine.storage.repositories import (
    CYCLE_FIELDS_EXCLUDED,
    CYCLE_MEMBERSHIP_FIELDS_EXCLUDED,
    EPISODE_FIELDS_EXCLUDED,
    EPISODE_LINK_FIELDS_EXCLUDED,
    SCOPE_FIELDS_EXCLUDED,
    THESIS_FIELDS_EXCLUDED,
    CycleMembershipRepository,
    CycleRepository,
    EpisodeLinkRepository,
    EpisodeRepository,
    ScopeRepository,
    ThesisRepository,
)

UTC = timezone.utc


def _dt(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def assert_field_coverage(model_cls, instance, excluded: set) -> None:
    """Every non-excluded field must have been explicitly set on the maximal
    instance. exclude_unset tracks explicitly-passed constructor arguments."""
    explicitly_set = set(instance.model_dump(exclude_unset=True).keys())
    missing = set(model_cls.model_fields) - explicitly_set - excluded
    assert not missing, (
        f"{model_cls.__name__} maximal builder does not set {sorted(missing)}; "
        "either set them (and make the converter persist them) or add them to "
        "the exclusion allowlist in repositories.py with a justification"
    )


def maximal_episode() -> Episode:
    return Episode(
        id=uuid4(),
        title="1929 Stock Market Crash",
        summary="Speculative boom collapses into panic and depression",
        start_date=_dt(1929, 10, 24),
        end_date=_dt(1932, 7, 8),
        start_year=1929,
        end_year=1932,
        date_precision="day",
        location="United States",
        setting_description="Late-1920s New York financial markets",
        scope_id="us",
        scope_name="United States",
        scope_kind=ScopeKind.POLITY,
        parent_scope_name="Western Civilization",
        scope_confidence=0.97,
        scope_evidence="Across the United States, markets fell.",
        scope_notes="National market treated as the focal scope.",
        actors=[
            Actor(
                id=uuid4(),
                name="Herbert Hoover",
                role="President",
                canonical_role="central_authority",
                role_fit_confidence=0.9,
                attributes={"party": "R"},
            ),
            Actor(id=uuid4(), name="J.P. Morgan Jr.", role="Banker", attributes={}),
        ],
        initiating_conditions=["Speculative excess", "Margin buying"],
        escalation_mechanics=["Panic selling", "Margin calls"],
        tension="Leverage vs confidence",
        resolution="Market bottomed 89% below peak",
        consequences=["Great Depression", "New Deal"],
        mechanism_tags=[MechanismTag.CREDIT_EXPANSION, MechanismTag.ASSET_BUBBLE],
        change_pattern=ChangePattern.FRAGMENTATION_AND_RELEASE,
        pattern_confidence=0.91,
        pattern_rationale="Confidence and coordination break rapidly.",
        situation_scale=SituationScale.POLITY,
        domains=[SituationDomain.ECONOMIC, SituationDomain.POLITICAL],
        configuration=SituationConfiguration(
            capacity=-0.6,
            cohesion=-0.5,
            pressure=0.9,
            legitimacy=-0.3,
            adaptability=-0.2,
            agency=0.4,
        ),
        mechanism_families=[
            MechanismFamily.AMPLIFICATION_FEEDBACK,
            MechanismFamily.CONTAGION_DIFFUSION,
        ],
        arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        arc_phase=ArcPhase.PANIC,
        phase_confidence=0.95,
        arc_rationale="Classic Kindleberger sequence",
        classification_state=ClassificationState.CLASSIFIED,
        secondary_arcs=[(ArcType.HUBRIS_NEMESIS, ArcPhase.CLIMAX, 0.4)],
        source_passages=[
            SourcePassage(
                work_id="galbraith-1955",
                passage_id="ch5-p3",
                text="The crash came in October...",
                chapter="5",
                section="The Twilight of Illusion",
                page=99,
                historiographic_school="Keynesian",
            )
        ],
        extracted_from=["chunk-001", "chunk-002"],
        source_published_at=_dt(1955, 1, 1),
        created_at=_dt(2026, 7, 10, 12, 0, 0),
        updated_at=_dt(2026, 7, 10, 12, 0, 0),
        version=1,
        surface_embedding=[0.1] * 384,
        structural_embedding=[0.2] * 384,
        surface_embedding_epoch="all-MiniLM-L6-v2",
        structural_embedding_epoch="render-v0.7.0+all-MiniLM-L6-v2",
    )


def minimal_episode() -> Episode:
    return Episode(title="Minimal", summary="Only required fields")


def maximal_cycle(parent_id) -> Cycle:
    return Cycle(
        id=uuid4(),
        name="Fourth Turning",
        scale=CycleScale.GENERATIONAL,
        description="Strauss-Howe crisis turning",
        scope_id="us",
        start_date=_dt(1946, 1, 1),
        end_date=_dt(1964, 12, 31),
        parent_cycle_id=parent_id,
        dominant_arc_types=[ArcType.CREDIT_BOOM_AND_BUST, ArcType.REBIRTH],
        phase_estimate=ArcPhase.DISTRESS,
        framework_source="strauss_howe",
        is_arc_instance=True,
        created_at=_dt(2026, 7, 10, 12, 0, 0),
        updated_at=_dt(2026, 7, 10, 12, 0, 0),
    )


def maximal_thesis() -> Thesis:
    ep_id = uuid4()
    return Thesis(
        id=uuid4(),
        query="Will 2026 see a market crash?",
        query_date=_dt(2026, 1, 1),
        analog_episode_ids=[uuid4(), uuid4()],
        analog_similarity_scores=[0.92, 0.87],
        dominant_continuation=Continuation(description="Soft landing", probability=0.65, supporting_analogs=8),
        alternative_continuations=[("Hard landing", 0.25), ("Melt-up", 0.10)],
        confidence=ThesisConfidence.MEDIUM,
        watch_for_indicators=["credit_contraction", "leverage_buildup"],
        key_uncertainties=["Central bank reaction function"],
        confidence_interval=(0.5, 0.8),
        narrative_synthesis="Analogs suggest late-cycle distress without panic.",
        estimated_duration="6-18 months",
        resolution_criteria=["Drawdown >20% within 18 months"],
        cited_episodes={
            ep_id: [
                SourcePassage(
                    work_id="kindleberger-1978",
                    passage_id="ch2-p1",
                    text="Manias typically accelerate...",
                    chapter="2",
                    section=None,
                    page=41,
                    historiographic_school=None,
                )
            ]
        },
        resolved=True,
        resolution_date=_dt(2026, 6, 30),
        resolution_outcome="partial",
        brier_score=0.18,
        mode=ThesisMode.ARC_BASED,
        scope_registry_version="scope-v0.1.0",
        created_at=_dt(2026, 7, 10, 12, 0, 0),
        model_version="thesis-v1.0",
        taxonomy_version="arc-v0.1.0",
    )


class TestEpisodeRoundTrip:
    @pytest.mark.asyncio
    async def test_maximal(self, db_session):
        repo = EpisodeRepository(db_session)
        episode = maximal_episode()
        assert_field_coverage(Episode, episode, EPISODE_FIELDS_EXCLUDED)

        await repo.create(episode)
        fetched = await repo.get_by_id(episode.id)

        assert fetched is not None

        # Actors come back set-ordered via the association table; compare
        # full dumps order-insensitively, then compare the rest field-by-field.
        def actor_dumps(episode_model):
            return sorted(
                (a.model_dump() for a in episode_model.actors),
                key=lambda d: str(d["id"]),
            )

        assert actor_dumps(fetched) == actor_dumps(episode)
        dump_a = episode.model_dump(exclude=EPISODE_FIELDS_EXCLUDED | {"actors"})
        dump_b = fetched.model_dump(exclude=EPISODE_FIELDS_EXCLUDED | {"actors"})
        assert dump_a == dump_b

    @pytest.mark.asyncio
    async def test_minimal(self, db_session):
        repo = EpisodeRepository(db_session)
        episode = minimal_episode()

        await repo.create(episode)
        fetched = await repo.get_by_id(episode.id)

        assert fetched is not None
        assert fetched.model_dump(exclude=EPISODE_FIELDS_EXCLUDED) == episode.model_dump(
            exclude=EPISODE_FIELDS_EXCLUDED
        )


class TestCycleRoundTrip:
    @pytest.mark.asyncio
    async def test_maximal(self, db_session):
        repo = CycleRepository(db_session)
        parent = Cycle(name="Parent", scale=CycleScale.CIVILIZATIONAL)
        await repo.create(parent)

        cycle = maximal_cycle(parent.id)
        assert_field_coverage(Cycle, cycle, CYCLE_FIELDS_EXCLUDED)

        await repo.create(cycle)
        fetched = await repo.get_by_id(cycle.id)

        assert fetched is not None
        assert fetched.model_dump(exclude=CYCLE_FIELDS_EXCLUDED) == cycle.model_dump(exclude=CYCLE_FIELDS_EXCLUDED)

    @pytest.mark.asyncio
    async def test_minimal(self, db_session):
        repo = CycleRepository(db_session)
        cycle = Cycle(name="Minimal", scale=CycleScale.EPISODIC)

        await repo.create(cycle)
        fetched = await repo.get_by_id(cycle.id)

        assert fetched is not None
        assert fetched.model_dump(exclude=CYCLE_FIELDS_EXCLUDED) == cycle.model_dump(exclude=CYCLE_FIELDS_EXCLUDED)


class TestThesisRoundTrip:
    @pytest.mark.asyncio
    async def test_maximal(self, db_session):
        repo = ThesisRepository(db_session)
        thesis = maximal_thesis()
        assert_field_coverage(Thesis, thesis, THESIS_FIELDS_EXCLUDED)

        await repo.create(thesis)
        fetched = await repo.get_by_id(thesis.id)

        assert fetched is not None
        assert fetched.model_dump(exclude=THESIS_FIELDS_EXCLUDED) == thesis.model_dump(exclude=THESIS_FIELDS_EXCLUDED)

    @pytest.mark.asyncio
    async def test_minimal(self, db_session):
        repo = ThesisRepository(db_session)
        thesis = Thesis(
            query="Minimal?",
            query_date=_dt(2026, 1, 1),
            model_version="m",
            taxonomy_version="t",
        )

        await repo.create(thesis)
        fetched = await repo.get_by_id(thesis.id)

        assert fetched is not None
        assert fetched.model_dump() == thesis.model_dump()


class TestCycleMembershipRoundTrip:
    @pytest.mark.asyncio
    async def test_maximal(self, db_session):
        episode = minimal_episode()
        cycle = Cycle(name="Instance", scale=CycleScale.EPISODIC, is_arc_instance=True)
        await EpisodeRepository(db_session).create(episode)
        await CycleRepository(db_session).create(cycle)

        repo = CycleMembershipRepository(db_session)
        membership = CycleMembership(
            id=uuid4(),
            episode_id=episode.id,
            cycle_id=cycle.id,
            reading=ArcAssignment(
                arc_type=ArcType.CREDIT_BOOM_AND_BUST,
                phase_index=3,
                confidence=0.8,
                rationale="Panic beat of the instance",
            ),
            salience=0.9,
            phase_coverage=[3, 4],
            link_status=LinkStatus.INFERRED,
            review_status=ReviewStatus.PENDING,
            created_at=_dt(2026, 7, 10, 12, 0, 0),
        )
        assert_field_coverage(CycleMembership, membership, CYCLE_MEMBERSHIP_FIELDS_EXCLUDED)

        await repo.create(membership)
        fetched = await repo.get_by_cycle(cycle.id)

        assert len(fetched) == 1
        assert fetched[0].model_dump() == membership.model_dump()


class TestScopeRoundTrip:
    @pytest.mark.asyncio
    async def test_maximal(self, db_session):
        repo = ScopeRepository(db_session)
        parent = Scope(id="western", kind="civilization", name="Western Civilization")
        await repo.create(parent)

        scope = Scope(
            id="us",
            kind="polity",
            name="United States",
            parent_scope_id="western",
            aliases=["USA", "America"],
            notes="Test entry",
        )
        assert_field_coverage(Scope, scope, SCOPE_FIELDS_EXCLUDED)

        await repo.create(scope)
        fetched = await repo.get_by_id("us")

        assert fetched is not None
        assert fetched.model_dump() == scope.model_dump()


class TestEpisodeLinkRoundTrip:
    @pytest.mark.asyncio
    async def test_maximal(self, db_session):
        source = minimal_episode()
        target = minimal_episode()
        repo_e = EpisodeRepository(db_session)
        await repo_e.create(source)
        await repo_e.create(target)

        repo = EpisodeLinkRepository(db_session)
        link = EpisodeLink(
            id=uuid4(),
            source_episode_id=source.id,
            target_episode_id=target.id,
            edge_kind=EdgeKind.CAUSES,
            link_status=LinkStatus.ATTESTED,
            distance=0.12,
            evidence="'the panic spread to...' (p. 44)",
            review_status=ReviewStatus.APPROVED,
            reviewed_by="analyst-1",
            reviewed_at=_dt(2026, 7, 10, 12, 0, 0),
            created_at=_dt(2026, 7, 10, 12, 0, 0),
        )
        assert_field_coverage(EpisodeLink, link, EPISODE_LINK_FIELDS_EXCLUDED)

        await repo.create(link)
        fetched = await repo.get_by_id(link.id)

        assert fetched is not None
        assert fetched.model_dump() == link.model_dump()
