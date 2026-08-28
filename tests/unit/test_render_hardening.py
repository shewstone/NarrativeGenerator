"""Structural render hardening tests (T2, docs/tickets/T2-structural-render-hardening.md).

The render is the perceptual bottleneck of the discovery apparatus
(Sec 10): free-text roles and proper nouns leaking into the structural
embedding quietly break place-blind matching. These tests pin the
guarantee: no identity marker from the source record survives the render.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

from narrative_engine.extraction.pipeline import ExtractionOrchestrator
from narrative_engine.extraction.roles import (
    UNRESOLVED_ACTOR_TOKEN,
    ActorRole,
    is_known_role,
)
from narrative_engine.models import Actor, ArcPhase, ArcType, Episode
from narrative_engine.retrieval.embeddings import EmbeddingGenerator
from narrative_engine.taxonomy.residue import (
    episode_residue,
    role_residue_by_era,
    role_residue_by_source,
)

UTC = timezone.utc


def _render(episode: Episode) -> str:
    return EmbeddingGenerator().render_structural_template(episode)


def _episode_1907() -> Episode:
    return Episode(
        title="Panic of 1907",
        summary="Knickerbocker Trust collapse triggers bank runs",
        location="New York",
        scope_id="United States",
        start_date=datetime(1907, 10, 1, tzinfo=UTC),
        arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        arc_phase=ArcPhase.PANIC,
        actors=[
            Actor(
                name="J.P. Morgan",
                role="Banker who organized the rescue",
                canonical_role=ActorRole.FINANCIER.value,
                role_fit_confidence=0.9,
            ),
            Actor(
                name="Knickerbocker Trust",
                role="Failed trust company",
                canonical_role=None,  # below floor -> residue
                role_fit_confidence=0.3,
            ),
        ],
        initiating_conditions=[
            "J.P. Morgan's syndicate had fueled copper speculation",
            "Trust companies in New York held thin reserves",
        ],
        escalation_mechanics=[
            "Runs spread from Knickerbocker Trust in October 1907",
            "Call money rates spiked across the United States",
        ],
        tension="Private liquidity vs systemic panic in America",
    )


class TestRoleTokensOnly:
    def test_canonical_tokens_render_free_text_never_does(self):
        rendered = _render(_episode_1907())
        assert "Actor roles: financier, unresolved_actor" in rendered
        assert "Banker who organized the rescue" not in rendered
        assert "Failed trust company" not in rendered

    def test_unresolved_actors_share_one_token(self):
        episode = _episode_1907()
        episode.actors = [
            Actor(name="A", role="x"),
            Actor(name="B", role="y"),
        ]
        rendered = _render(episode)
        assert f"Actor roles: {UNRESOLVED_ACTOR_TOKEN}" in rendered
        assert "x" not in rendered.split("Actor roles:")[1].split("\n")[0].replace(
            UNRESOLVED_ACTOR_TOKEN, ""
        )


class TestProperNounScrub:
    def test_no_identity_marker_survives(self):
        rendered = _render(_episode_1907())
        for marker in [
            "J.P. Morgan",
            "Knickerbocker",
            "New York",
            "United States",
            "America",
            "1907",
            "October",
        ]:
            assert marker not in rendered, f"{marker!r} leaked into structural render"

    def test_resolved_names_become_role_tokens_not_holes(self):
        rendered = _render(_episode_1907())
        # "J.P. Morgan's syndicate" keeps its structural meaning as
        # "financier's syndicate", it isn't deleted.
        assert "financier's syndicate had fueled copper speculation" in rendered

    def test_place_and_date_tokens(self):
        rendered = _render(_episode_1907())
        assert "<PLACE>" in rendered
        assert "<DATE>" in rendered

    def test_permutation_invariance(self):
        """Identical structure under different names/places/dates renders
        identically — the operational meaning of place/date-blind."""
        a = _episode_1907()

        b = _episode_1907()
        b.location = "Vienna"
        b.scope_id = "Austria-Hungary"
        b.start_date = datetime(1873, 5, 1, tzinfo=UTC)
        b.actors[0] = b.actors[0].model_copy(update={"name": "Rothschild"})
        b.actors[1] = b.actors[1].model_copy(update={"name": "Creditanstalt"})
        b.initiating_conditions = [
            "Rothschild's syndicate had fueled copper speculation",
            "Trust companies in Vienna held thin reserves",
        ]
        b.escalation_mechanics = [
            "Runs spread from Creditanstalt in May 1873",
            "Call money rates spiked across Austria-Hungary",
        ]
        b.tension = "Private liquidity vs systemic panic in Austria"

        # Not identical (b's tension says "Austria" which is an
        # Austria-Hungary alias -> <PLACE>; a's says "America" -> <PLACE>).
        assert _render(a) == _render(b)


class TestExtractionRoleFloor:
    def _orchestrator(self) -> ExtractionOrchestrator:
        return ExtractionOrchestrator(pipeline=AsyncMock())

    def test_floor_and_vocabulary_enforced(self):
        orchestrator = self._orchestrator()
        assert (
            orchestrator._resolve_canonical_role(
                {"canonical_role": "financier", "role_fit_confidence": 0.9}
            )
            == "financier"
        )
        # Below floor -> None
        assert (
            orchestrator._resolve_canonical_role(
                {"canonical_role": "financier", "role_fit_confidence": 0.2}
            )
            is None
        )
        # Unknown vocabulary value -> None (never invent roles)
        assert (
            orchestrator._resolve_canonical_role(
                {"canonical_role": "space_wizard", "role_fit_confidence": 0.99}
            )
            is None
        )
        # Missing confidence -> None (absence of evidence discipline)
        assert (
            orchestrator._resolve_canonical_role({"canonical_role": "financier"})
            is None
        )

    def test_vocabulary_membership_helper(self):
        assert is_known_role("kingmaker")
        assert not is_known_role("protagonist")  # old free-text style


class TestResidueMetric:
    def test_episode_residue(self):
        episode = _episode_1907()
        assert episode_residue(episode) == 0.5  # 1 of 2 unresolved
        assert episode_residue(Episode(title="t", summary="s")) is None

    def test_residue_by_source_and_era(self):
        resolved_actor = Actor(name="A", role="r", canonical_role="financier")
        unresolved_actor = Actor(name="B", role="r")

        old = Episode(
            title="old",
            summary="s",
            start_date=datetime(1850, 1, 1, tzinfo=UTC),
            extracted_from=["book-old"],
            actors=[resolved_actor, unresolved_actor],
        )
        modern = Episode(
            title="modern",
            summary="s",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            extracted_from=["news-2024"],
            actors=[unresolved_actor, unresolved_actor],
        )

        by_source = role_residue_by_source([old, modern])
        assert by_source["book-old"] == 0.5
        assert by_source["news-2024"] == 1.0  # the Sec 10.5 modern-text alarm

        by_era = role_residue_by_era([old, modern])
        assert by_era["1850"] == 0.5
        assert by_era["2000"] == 1.0
