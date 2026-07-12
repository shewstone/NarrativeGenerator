"""API endpoint tests (T8, docs/tickets/T8-dashboard-and-review-ui.md).

Uses httpx.ASGITransport so requests run on the same event loop as the
db_session fixture (a sync TestClient would run the app on a second loop
and asyncpg connections cannot cross loops).
"""

from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio

from narrative_engine.api.app import create_app, get_session
from narrative_engine.models import (
    ArcPhase,
    ArcType,
    Cycle,
    CycleMembership,
    CycleScale,
    EdgeKind,
    Episode,
    EpisodeLink,
    LinkStatus,
    MechanismTag,
    ReviewStatus,
    SourceDocument,
    SourceDocumentStatus,
)
from narrative_engine.storage.repositories import (
    CycleMembershipRepository,
    CycleRepository,
    EpisodeLinkRepository,
    EpisodeRepository,
    SourceDocumentRepository,
)

UTC = timezone.utc


@pytest_asyncio.fixture
async def client(db_session):
    app = create_app(start_watcher=False)

    async def override_session():
        yield db_session

    app.dependency_overrides[get_session] = override_session
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestDashboardShell:
    @pytest.mark.asyncio
    async def test_dashboard_serves(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        assert "NARRATIVE ENGINE" in response.text
        assert "Arc Space" in response.text
        assert "Evidence inspector" in response.text
        assert "/api/arc-space" in response.text

    @pytest.mark.asyncio
    async def test_health(self, client):
        response = await client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert set(body) >= {"documents", "episodes", "arc_instances", "pending_reviews"}


class TestDocumentsEndpoint:
    @pytest.mark.asyncio
    async def test_queue_lists_documents_with_duplicates_flagged(self, client, db_session):
        repo = SourceDocumentRepository(db_session)
        original = SourceDocument(filename="book.txt", content_hash="c" * 64, size_bytes=10)
        await repo.create(original)
        duplicate = SourceDocument(
            filename="book-again.txt",
            content_hash="c" * 64,
            status="duplicate",
            duplicate_of=original.id,
        )
        await repo.create(duplicate)

        body = (await client.get("/api/documents")).json()
        by_name = {d["filename"]: d for d in body}
        assert by_name["book.txt"]["status"] == "queued"
        assert by_name["book-again.txt"]["status"] == "duplicate"
        assert by_name["book-again.txt"]["duplicate_of"] == str(original.id)


class TestRetryEndpoint:
    @pytest.mark.asyncio
    async def test_failed_document_can_be_retried(self, client, db_session):
        repo = SourceDocumentRepository(db_session)
        failed = SourceDocument(
            filename="broken.epub",
            content_hash="d" * 64,
            status="failed",
            error="EPUB parsing requires ebooklib",
            chunks_created=10,
            chunks_processed=4,
            episodes_created=12,
        )
        await repo.create(failed)

        response = await client.post(f"/api/documents/{failed.id}/retry")
        assert response.status_code == 200

        queued = await repo.get_by_id(failed.id)
        assert queued is not None
        assert queued.status == SourceDocumentStatus.QUEUED
        assert queued.chunks_processed == 4
        assert queued.episodes_created == 12

    @pytest.mark.asyncio
    async def test_extraction_pending_is_retryable(self, client, db_session):
        """Ingested before an LLM key existed -> retry re-picks it for
        extraction (the "key arrived later" path)."""
        repo = SourceDocumentRepository(db_session)
        pending = SourceDocument(
            filename="book.epub",
            content_hash="f" * 64,
            status="completed",
            extraction_ran=False,
        )
        await repo.create(pending)

        response = await client.post(f"/api/documents/{pending.id}/retry")
        assert response.status_code == 200
        queued = await repo.get_by_id(pending.id)
        assert queued is not None
        assert queued.status == SourceDocumentStatus.QUEUED

    @pytest.mark.asyncio
    async def test_fully_extracted_and_duplicate_are_not_retryable(self, client, db_session):
        repo = SourceDocumentRepository(db_session)
        done = SourceDocument(
            filename="done.txt",
            content_hash="e" * 64,
            status="completed",
            extraction_ran=True,
        )
        await repo.create(done)
        duplicate = SourceDocument(
            filename="dup.txt", content_hash="e" * 64, status="duplicate"
        )
        await repo.create(duplicate)

        for doc in (done, duplicate):
            response = await client.post(f"/api/documents/{doc.id}/retry")
            assert response.status_code == 409
            assert await repo.get_by_id(doc.id) is not None


class TestArcInstancesEndpoint:
    @pytest.mark.asyncio
    async def test_phase_coverage_payload(self, client, db_session):
        episode = Episode(
            title="Panic beat",
            summary="s",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.PANIC,
            start_date=datetime(1907, 10, 1, tzinfo=UTC),
        )
        await EpisodeRepository(db_session).create(episode)

        instance = Cycle(
            name="credit_boom_and_bust, United States, 1907–1907",
            scale=CycleScale.EPISODIC,
            is_arc_instance=True,
            dominant_arc_types=[ArcType.CREDIT_BOOM_AND_BUST],
            scope_id="us",
        )
        await CycleRepository(db_session).create(instance)
        await CycleMembershipRepository(db_session).create(
            CycleMembership(
                episode_id=episode.id,
                cycle_id=instance.id,
                link_status=LinkStatus.INFERRED,
                review_status=ReviewStatus.PENDING,
            )
        )

        body = (await client.get("/api/arc-instances")).json()
        assert len(body) == 1
        arc = body[0]
        assert arc["arc_type"] == "credit_boom_and_bust"
        assert "panic" in arc["covered_phases"]
        # Gap visibility: expected phases include ones NOT covered.
        assert set(arc["expected_phases"]) - set(arc["covered_phases"])
        assert arc["episodes"][0]["link_status"] == "inferred"


class TestArcSpaceEndpoint:
    @pytest.mark.asyncio
    async def test_returns_projected_nodes_and_interpretable_edges(self, client, db_session):
        repo = EpisodeRepository(db_session)
        first = Episode(
            title="Credit expansion",
            summary="Banks expanded leverage.",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.BOOM,
            phase_confidence=0.91,
            mechanism_tags=[MechanismTag.CREDIT_EXPANSION, MechanismTag.DEBT_OVERHANG],
            start_date=datetime(1906, 1, 1, tzinfo=UTC),
        )
        second = Episode(
            title="Banking panic",
            summary="Depositors rushed to withdraw funds.",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.PANIC,
            phase_confidence=0.94,
            mechanism_tags=[MechanismTag.CREDIT_EXPANSION, MechanismTag.FISCAL_DISTRESS],
            start_date=datetime(1907, 10, 1, tzinfo=UTC),
        )
        await repo.create(first)
        await repo.create(second)
        await repo.update_embedding(first.id, [1.0, 0.0] + [0.0] * 382, kind="structural")
        await repo.update_embedding(second.id, [0.9, 0.1] + [0.0] * 382, kind="structural")
        await repo.update_embedding(first.id, [1.0, 0.0] + [0.0] * 382, kind="surface")
        await repo.update_embedding(second.id, [0.8, 0.2] + [0.0] * 382, kind="surface")

        cycle = Cycle(
            name="Credit cycle, United States, 1906–1907",
            scale=CycleScale.EPISODIC,
            is_arc_instance=True,
            dominant_arc_types=[ArcType.CREDIT_BOOM_AND_BUST],
        )
        await CycleRepository(db_session).create(cycle)
        for episode in (first, second):
            await CycleMembershipRepository(db_session).create(
                CycleMembership(
                    episode_id=episode.id,
                    cycle_id=cycle.id,
                    link_status=LinkStatus.INFERRED,
                    review_status=ReviewStatus.PENDING,
                )
            )

        rejected_cycle = Cycle(
            name="Rejected candidate arc",
            scale=CycleScale.EPISODIC,
            is_arc_instance=True,
        )
        await CycleRepository(db_session).create(rejected_cycle)
        for episode in (first, second):
            await CycleMembershipRepository(db_session).create(
                CycleMembership(
                    episode_id=episode.id,
                    cycle_id=rejected_cycle.id,
                    link_status=LinkStatus.INFERRED,
                    review_status=ReviewStatus.REJECTED,
                )
            )
        await EpisodeLinkRepository(db_session).create(
            EpisodeLink(
                source_episode_id=first.id,
                target_episode_id=second.id,
                edge_kind=EdgeKind.SAME_EVENT_AS,
                link_status=LinkStatus.INFERRED,
                review_status=ReviewStatus.REJECTED,
            )
        )

        response = await client.get("/api/arc-space?k=1")
        assert response.status_code == 200
        body = response.json()
        assert body["projection"]["algorithm"] == "pca"
        assert body["projection"]["source_dimensions"] == 384
        assert len(body["nodes"]) == 2
        assert {node["title"] for node in body["nodes"]} == {
            "Credit expansion",
            "Banking panic",
        }
        assert all(set(node) >= {"x", "y", "z", "mechanisms", "phase"} for node in body["nodes"])

        knn = next(edge for edge in body["edges"] if edge["kind"] == "structural_neighbor")
        assert knn["structural_similarity"] > 0.9
        assert knn["shared_mechanisms"] == ["credit_expansion"]
        assert "explanation" in knn

        trajectory = next(edge for edge in body["edges"] if edge["kind"] == "arc_sequence")
        assert trajectory["arc_name"] == cycle.name
        assert trajectory["review_status"] == "pending"
        assert not any(
            edge.get("arc_name") == rejected_cycle.name for edge in body["edges"]
        )
        assert not any(edge["kind"] == "same_event_as" for edge in body["edges"])


class TestReviewFlow:
    @pytest.mark.asyncio
    async def test_membership_approve_persists(self, client, db_session):
        episode = Episode(title="ep", summary="s")
        await EpisodeRepository(db_session).create(episode)
        cycle = Cycle(name="inst", scale=CycleScale.EPISODIC, is_arc_instance=True)
        await CycleRepository(db_session).create(cycle)
        membership = CycleMembership(
            episode_id=episode.id,
            cycle_id=cycle.id,
            link_status=LinkStatus.INFERRED,
            review_status=ReviewStatus.PENDING,
        )
        await CycleMembershipRepository(db_session).create(membership)

        queue = (await client.get("/api/review-queue")).json()
        assert len(queue["memberships"]) == 1

        response = await client.post(
            f"/api/review/membership/{membership.id}", json={"decision": "approved"}
        )
        assert response.status_code == 200
        assert response.json()["review_status"] == "approved"

        assert (await client.get("/api/review-queue")).json()["memberships"] == []

    @pytest.mark.asyncio
    async def test_link_reject_and_validation(self, client, db_session):
        repo_e = EpisodeRepository(db_session)
        a = Episode(title="a", summary="s")
        b = Episode(title="b", summary="s")
        await repo_e.create(a)
        await repo_e.create(b)
        link = EpisodeLink(
            source_episode_id=a.id,
            target_episode_id=b.id,
            edge_kind=EdgeKind.SAME_EVENT_AS,
            link_status=LinkStatus.INFERRED,
            review_status=ReviewStatus.PENDING,
        )
        await EpisodeLinkRepository(db_session).create(link)

        bad = await client.post(f"/api/review/link/{link.id}", json={"decision": "maybe"})
        assert bad.status_code == 422

        ok = await client.post(f"/api/review/link/{link.id}", json={"decision": "rejected"})
        assert ok.status_code == 200
        assert ok.json()["review_status"] == "rejected"

    @pytest.mark.asyncio
    async def test_unknown_ids_404(self, client):
        response = await client.post(
            "/api/review/membership/00000000-0000-0000-0000-000000000000",
            json={"decision": "approved"},
        )
        assert response.status_code == 404
