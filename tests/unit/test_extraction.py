"""Unit tests for extraction pipeline."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from tenacity import wait_none

from narrative_engine.extraction.client import (
    ExtractionPipeline,
    LLMError,
    OpenAIClient,
)
from narrative_engine.extraction.config import ExtractionPipelineConfig, LLMConfig
from narrative_engine.extraction.pipeline import ExtractionOrchestrator, PipelineResult


class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_default_config(self):
        """Test default LLM configuration."""
        config = LLMConfig()

        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-5"
        assert config.temperature == 0.0
        assert config.max_tokens == 4000
        assert config.request_timeout_seconds == 90.0

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("NE_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("NE_LLM_MODEL", "claude-3-opus")
        monkeypatch.setenv("NE_LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("NE_LLM_MAX_TOKENS", "8000")
        monkeypatch.setenv("NE_LLM_REASONING_EFFORT", "none")

        config = LLMConfig.from_env()

        assert config.provider == "anthropic"
        assert config.model == "claude-3-opus"
        assert config.temperature == 0.5
        assert config.max_tokens == 8000
        assert config.reasoning_effort == "none"

    def test_config_reads_openai_compatible_base_url(self, monkeypatch):
        monkeypatch.setenv("NE_LLM_BASE_URL", "https://api.venice.ai/api/v1")

        config = LLMConfig.from_env()

        assert config.base_url == "https://api.venice.ai/api/v1"

    def test_config_uses_key_for_selected_provider(self, monkeypatch):
        monkeypatch.delenv("NE_LLM_API_KEY", raising=False)
        monkeypatch.setenv("NE_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

        assert LLMConfig.from_env().api_key == "openai-key"


class TestExtractionPipelineConfig:
    """Tests for extraction pipeline configuration."""

    def test_default_pipeline_config(self):
        """Test default pipeline configuration."""
        config = ExtractionPipelineConfig()

        assert config.enable_segmentation is True
        assert config.enable_extraction is True
        assert config.enable_classification is True
        assert config.enable_linking is True
        assert config.segmentation_model == "claude-haiku-4-5"
        assert config.scope_model == "claude-haiku-4-5"
        assert config.extraction_model == "claude-sonnet-5"
        assert config.reconciliation_model == "claude-sonnet-5"

    def test_pipeline_config_from_env(self, monkeypatch):
        """Test pipeline configuration from environment."""
        monkeypatch.setenv("NE_LLM_PROVIDER", "openai")
        monkeypatch.setenv("NE_ENABLE_SEGMENTATION", "false")
        monkeypatch.setenv("NE_SEG_MODEL", "gpt-4")

        config = ExtractionPipelineConfig.from_env()

        assert config.enable_segmentation is False
        assert config.segmentation_model == "gpt-4"

    def test_openai_provider_gets_openai_stage_defaults(self, monkeypatch):
        monkeypatch.setenv("NE_LLM_PROVIDER", "openai")
        for var in (
            "NE_SEG_MODEL",
            "NE_SCOPE_MODEL",
            "NE_EXTRACT_MODEL",
            "NE_RECONCILE_MODEL",
            "NE_CLASSIFY_MODEL",
            "NE_LINK_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)

        config = ExtractionPipelineConfig.from_env()

        assert all(
            not model.startswith("claude-")
            for model in (
                config.segmentation_model,
                config.scope_model,
                config.extraction_model,
                config.reconciliation_model,
                config.classification_model,
                config.linking_model,
            )
        )

    def test_linking_prefilter_config_from_env(self, monkeypatch):
        monkeypatch.setenv("NE_LINK_NEIGHBOR_WINDOW", "2")
        monkeypatch.setenv("NE_LINK_MAX_YEAR_GAP", "40")
        monkeypatch.setenv("NE_LINK_MIN_LEXICAL_OVERLAP", "0.35")

        config = ExtractionPipelineConfig.from_env()

        assert config.linking_neighbor_window == 2
        assert config.linking_max_year_gap == 40
        assert config.linking_min_lexical_overlap == 0.35

    def test_scope_confidence_floor_from_env(self, monkeypatch):
        monkeypatch.setenv("NE_TAU_SCOPE", "0.72")

        assert ExtractionPipelineConfig.from_env().scope_confidence_floor == 0.72

    def test_rejects_invalid_evidence_floor(self, monkeypatch):
        monkeypatch.setenv("NE_TAU_SCOPE", "1.2")

        with pytest.raises(ValueError, match="TAU_SCOPE"):
            ExtractionPipelineConfig.from_env()

    def test_rejects_provider_stage_model_mismatch(self, monkeypatch):
        monkeypatch.setenv("NE_LLM_PROVIDER", "openai")
        monkeypatch.setenv("NE_SEG_MODEL", "claude-haiku-4-5")

        with pytest.raises(ValueError, match="openai.*claude-haiku-4-5"):
            ExtractionPipelineConfig.from_env()


class TestOpenAIClient:
    """Tests for OpenAI client."""

    def test_uses_configured_openai_compatible_endpoint(self, monkeypatch):
        constructor = MagicMock()
        monkeypatch.setattr("narrative_engine.extraction.client.openai.AsyncOpenAI", constructor)

        OpenAIClient(
            LLMConfig(
                provider="openai",
                model="openai-gpt-52",
                api_key="venice-test-key",
                base_url="https://api.venice.ai/api/v1",
            )
        )

        constructor.assert_called_once_with(
            api_key="venice-test-key",
            base_url="https://api.venice.ai/api/v1",
            timeout=90.0,
            max_retries=0,
        )

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"result": "test"})
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        return mock_response

    @pytest.mark.asyncio
    async def test_complete_success(self, mock_openai_response):
        """Test successful completion."""
        config = LLMConfig(api_key="test-key")
        client = OpenAIClient(config)

        # Mock the OpenAI client
        mock_chat = AsyncMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_openai_response)
        client.client = MagicMock()
        client.client.chat = mock_chat

        result = await client.complete("Test prompt")

        assert result["content"] == json.dumps({"result": "test"})
        assert result["model"] == "gpt-4"
        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_complete_retries_connection_failure(self, mock_openai_response):
        """A dropped provider connection is retried without failing the chunk."""
        client = OpenAIClient(LLMConfig(api_key="test-key"))
        request = httpx.Request("POST", "https://example.test/chat/completions")
        mock_create = AsyncMock(
            side_effect=[openai.APIConnectionError(request=request), mock_openai_response]
        )
        client.client = MagicMock()
        client.client.chat.completions.create = mock_create

        result = await client.complete.retry_with(wait=wait_none())(client, "Test prompt")

        assert result["content"] == json.dumps({"result": "test"})
        assert mock_create.await_count == 2

    @pytest.mark.asyncio
    async def test_complete_forwards_reasoning_effort(self, mock_openai_response):
        """Test forwarding Venice's OpenAI-compatible reasoning extension."""
        config = LLMConfig(api_key="test-key", reasoning_effort="none")
        client = OpenAIClient(config)

        mock_create = AsyncMock(return_value=mock_openai_response)
        client.client = MagicMock()
        client.client.chat.completions.create = mock_create

        await client.complete("Test prompt")

        assert mock_create.await_args.kwargs["extra_body"] == {"reasoning_effort": "none"}

    @pytest.mark.asyncio
    async def test_complete_disables_thinking_explicitly_for_venice(self, mock_openai_response):
        """Venice receives its provider-specific no-thinking control as well."""
        config = LLMConfig(
            api_key="test-key",
            base_url="https://api.venice.ai/api/v1",
            reasoning_effort="none",
        )
        client = OpenAIClient(config)
        mock_create = AsyncMock(return_value=mock_openai_response)
        client.client.chat.completions.create = mock_create

        await client.complete("test prompt")

        assert mock_create.await_args.kwargs["extra_body"] == {
            "reasoning_effort": "none",
            "venice_parameters": {
                "disable_thinking": True,
                "strip_thinking_response": True,
            },
        }

    @pytest.mark.asyncio
    async def test_complete_with_json_parsing(self, mock_openai_response):
        """Test completion with JSON parsing."""
        config = LLMConfig(api_key="test-key")
        client = OpenAIClient(config)

        mock_chat = AsyncMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_openai_response)
        client.client = MagicMock()
        client.client.chat = mock_chat

        result = await client.complete_with_json("Test prompt")

        assert result == {"result": "test"}

    @pytest.mark.asyncio
    async def test_complete_json_decode_error(self):
        """Test handling of invalid JSON response."""
        config = LLMConfig(api_key="test-key")
        client = OpenAIClient(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"
        mock_response.model = "gpt-4"
        mock_response.usage = None

        mock_chat = AsyncMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_response)
        client.client = MagicMock()
        client.client.chat = mock_chat

        with pytest.raises(LLMError) as exc_info:
            await client.complete_with_json("Test prompt")

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_close_releases_sdk_transport(self):
        client = OpenAIClient(LLMConfig(api_key="test-key"))
        close = AsyncMock()
        client.client = MagicMock(close=close)

        await client.aclose()

        close.assert_awaited_once()


class TestExtractionPipeline:
    """Tests for extraction pipeline stages."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_segmentation_stage(self, mock_llm_client):
        """Test segmentation stage."""
        mock_llm_client.complete_with_json.return_value = {
            "episodes": [
                {
                    "number": 1,
                    "summary": "Test episode",
                    "beginning": "Start",
                    "tension": "Conflict",
                    "status": "resolved",
                }
            ]
        }

        pipeline = ExtractionPipeline(client=mock_llm_client)
        result = await pipeline.segment("Test text")

        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["summary"] == "Test episode"
        assert result["episodes"][0]["text"] == "Test text"
        assert mock_llm_client.complete_with_json.await_args.kwargs["max_tokens"] == 1_000

    @pytest.mark.asyncio
    async def test_segmentation_materializes_validated_source_spans(self, mock_llm_client):
        source = "Alpha rose to power. Beta later displaced it."
        second_start = source.index("Beta")
        mock_llm_client.complete_with_json.return_value = {
            "episodes": [
                {
                    "number": 1,
                    "summary": "Alpha rises",
                    "start_char": 0,
                    "end_char": second_start - 1,
                    "start_quote": "Alpha rose",
                    "end_quote": "power.",
                },
                {
                    "number": 2,
                    "summary": "Beta displaces Alpha",
                    "start_char": second_start,
                    "end_char": len(source),
                    "start_quote": "Beta later",
                    "end_quote": "it.",
                },
            ]
        }

        result = await ExtractionPipeline(client=mock_llm_client).segment(source)

        assert [episode["text"] for episode in result["episodes"]] == [
            "Alpha rose to power.",
            "Beta later displaced it.",
        ]

    @pytest.mark.asyncio
    async def test_segmentation_collapses_unresolvable_spans_to_one_extraction(self, mock_llm_client):
        source = "Alpha rose to power. Beta later displaced it."
        mock_llm_client.complete_with_json.return_value = {
            "episodes": [
                {"number": 1, "summary": "Alpha rises"},
                {"number": 2, "summary": "Beta displaces Alpha"},
            ]
        }

        result = await ExtractionPipeline(client=mock_llm_client).segment(source)

        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["text"] == source
        assert result["episodes"][0]["segmentation_fallback"] == "unresolved_source_spans"

    @pytest.mark.asyncio
    async def test_segmentation_recovers_multiple_events_from_ordered_start_quotes(self, mock_llm_client):
        source = "Preface. Alpha rose to power.\n\nBeta later displaced it."
        mock_llm_client.complete_with_json.return_value = {
            "episodes": [
                {
                    "number": 1,
                    "summary": "Alpha rises",
                    "start_quote": "Alpha rose to power.",
                    "end_quote": "not an exact ending",
                },
                {
                    "number": 2,
                    "summary": "Beta displaces Alpha",
                    "start_quote": "Beta later displaced it.",
                    "end_quote": "also inaccurate",
                },
            ]
        }

        result = await ExtractionPipeline(client=mock_llm_client).segment(source)

        assert len(result["episodes"]) == 2
        assert "Alpha rose" in result["episodes"][0]["text"]
        assert result["episodes"][1]["text"].startswith("Beta later")
        assert all(
            episode["boundary_resolution"] == "ordered_starts"
            for episode in result["episodes"]
        )

    @pytest.mark.asyncio
    async def test_segmentation_tolerates_quote_typography_and_whitespace(self, mock_llm_client):
        source = (
            "Preface. The leader’s reform—after long debate—began here.\n\n"
            "The opposition’s counter-movement then displaced it."
        )
        mock_llm_client.complete_with_json.return_value = {
            "episodes": [
                {
                    "number": 1,
                    "summary": "Reform begins",
                    "start_quote": "The leader's reform after long debate began here",
                    "end_quote": "inexact ending",
                },
                {
                    "number": 2,
                    "summary": "Opposition responds",
                    "start_quote": "The opposition's counter movement then displaced it",
                    "end_quote": "also inexact",
                },
            ]
        }

        result = await ExtractionPipeline(client=mock_llm_client).segment(source)

        assert len(result["episodes"]) == 2
        assert "leader’s reform" in result["episodes"][0]["text"]
        assert result["episodes"][1]["text"].startswith("The opposition’s")
        assert all(
            episode["boundary_resolution"] == "ordered_starts"
            for episode in result["episodes"]
        )

    @pytest.mark.asyncio
    async def test_extraction_stage(self, mock_llm_client):
        """Test extraction stage."""
        mock_llm_client.complete_with_json.return_value = {
            "title": "1929 Crash",
            "summary": "Stock market crash",
            "actors": [{"name": "Retail Investors", "role": "crowd"}],
            "setting": {"location": "United States", "time_period": "1929"},
            "initiating_conditions": ["Speculation"],
            "escalation_mechanics": ["Panic"],
            "tension": "Financial collapse",
            "resolution": "Market bottomed",
            "consequences": ["Great Depression"],
        }

        pipeline = ExtractionPipeline(client=mock_llm_client)
        result = await pipeline.extract(
            segment_text="Test",
            segment_summary="Test summary",
        )

        assert result["title"] == "1929 Crash"
        assert len(result["actors"]) == 1
        assert result["actors"][0]["name"] == "Retail Investors"

    @pytest.mark.asyncio
    async def test_classification_stage(self, mock_llm_client):
        """Test classification stage."""
        mock_llm_client.complete_with_json.return_value = {
            "arc_type": "credit_boom_and_bust",
            "arc_phase": "panic",
            "phase_confidence": 0.95,
            "rationale": "Clear panic signs",
            "secondary_arcs": [],
        }

        pipeline = ExtractionPipeline(client=mock_llm_client)
        result = await pipeline.classify(
            episode_summary="Crash summary",
            full_text="Full text",
        )

        assert result["arc_type"] == "credit_boom_and_bust"
        assert result["arc_phase"] == "panic"
        assert result["phase_confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_classification_second_pass(self, mock_llm_client):
        """Test second-pass classification."""
        config = ExtractionPipelineConfig(two_pass_classification=True)
        mock_llm_client.complete_with_json.return_value = {
            "arc_type": "credit_boom_and_bust",
            "arc_phase": "panic",
            "phase_confidence": 0.92,
            "rationale": "Refined",
            "changed_from_initial": True,
            "reason_for_change": "Similar episodes",
        }

        pipeline = ExtractionPipeline(client=mock_llm_client, config=config)
        result = await pipeline.classify_second_pass(
            episode_summary="Summary",
            initial_classification={"arc_type": "hubris_nemesis"},
            similar_episodes=[{"title": "Similar", "arc_type": "credit_boom_and_bust"}],
        )

        assert result["arc_type"] == "credit_boom_and_bust"
        assert result["changed_from_initial"] is True


class TestExtractionOrchestrator:
    """Tests for extraction orchestrator."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock extraction pipeline."""
        pipeline = AsyncMock()
        return pipeline

    def test_default_pipeline_receives_environment_config(self, monkeypatch):
        monkeypatch.setenv("NE_SEG_MODEL", "configured-segmenter")
        monkeypatch.setenv("NE_LLM_API_KEY", "test-key")

        orchestrator = ExtractionOrchestrator()

        assert orchestrator.pipeline.config is orchestrator.config
        assert orchestrator.pipeline.config.segmentation_model == "configured-segmenter"

    @pytest.mark.asyncio
    async def test_process_text_full_pipeline(self, mock_pipeline):
        """Test full pipeline execution."""
        # Mock segmentation
        mock_pipeline.segment.return_value = {"episodes": [{"number": 1, "summary": "Test", "text": "Test"}]}

        # Mock extraction
        mock_pipeline.extract.return_value = {
            "title": "Test Episode",
            "summary": "Test summary",
            "actors": [],
            "setting": {},
            "initiating_conditions": [],
            "escalation_mechanics": [],
            "consequences": [],
        }

        # Mock classification
        mock_pipeline.classify.return_value = {
            "arc_type": "hero_journey",
            "arc_phase": "setup",
            "phase_confidence": 0.8,
            "rationale": "Test",
            "secondary_arcs": [],
        }

        orchestrator = ExtractionOrchestrator(pipeline=mock_pipeline)

        # Mock session
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        result = await orchestrator.process_text(
            text="Test text",
            source_chunk_id="test-chunk",
            session=mock_session,
        )

        assert isinstance(result, PipelineResult)
        assert result.source_chunk_id == "test-chunk"
        assert len(result.episodes) == 1
        assert result.episodes[0].title == "Test Episode"
        assert result.errors == []
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_real_session_persists_passage_and_stage_audit(self, mock_pipeline, db_session):
        from sqlalchemy import select

        from narrative_engine.storage.orm_models import ExtractionRecordORM, SourcePassageORM

        source = "A reform coalition organized in 1903."
        mock_pipeline.segment.return_value = {
            "episodes": [{"number": 1, "summary": "Coalition organizes", "text": source}]
        }
        mock_pipeline.extract.return_value = {
            "title": "Coalition organizes",
            "summary": "A reform coalition organized.",
            "actors": [],
            "setting": {"start_date": "1903"},
        }
        mock_pipeline.classify.return_value = {
            "arc_type": "reform_then_reaction",
            "arc_phase": "setup",
            "phase_confidence": 0.8,
        }

        result = await ExtractionOrchestrator(pipeline=mock_pipeline).process_text(
            source,
            "audit-work_0",
            db_session,
        )

        records = (await db_session.execute(select(ExtractionRecordORM))).scalars().all()
        passages = (await db_session.execute(select(SourcePassageORM))).scalars().all()
        assert result.errors == []
        assert {record.pipeline_stage for record in records} == {
            "segmentation",
            "extraction",
            "classification",
        }
        assert len(passages) == 1
        assert passages[0].work_id == "audit-work"
        assert passages[0].text == source

    @pytest.mark.asyncio
    async def test_document_reconciliation_updates_contextual_phase(self, mock_pipeline, db_session):
        from narrative_engine.models import Episode
        from narrative_engine.storage.repositories import EpisodeRepository

        first = Episode(
            title="Movement organizes",
            summary="Organizations form.",
            start_year=1900,
            scope_id="uk_womens_suffrage_movement",
            scope_name="Women's Suffrage Movement",
            scope_confidence=0.9,
            arc_type="reform_then_reaction",
            arc_phase="rising_action",
            phase_confidence=0.7,
            extracted_from=["reconcile-work_0"],
        )
        second = Episode(
            title="Voting reform enacted",
            summary="The franchise expands.",
            start_year=1918,
            scope_id="uk_womens_suffrage_movement",
            scope_name="Women's Suffrage Movement",
            scope_confidence=0.9,
            arc_type="reform_then_reaction",
            arc_phase="climax",
            phase_confidence=0.7,
            extracted_from=["reconcile-work_1"],
        )
        repository = EpisodeRepository(db_session)
        await repository.create(first)
        await repository.create(second)
        mock_pipeline.reconcile_phases.return_value = {
            "episodes": [
                {
                    "episode_id": str(first.id),
                    "arc_type": "reform_then_reaction",
                    "arc_phase": "setup",
                    "confidence": 0.91,
                    "reason": "Organization precedes escalation.",
                },
                {
                    "episode_id": str(second.id),
                    "arc_type": "reform_then_reaction",
                    "arc_phase": "resolution",
                    "confidence": 0.88,
                    "reason": "The reform settled the immediate demand.",
                },
            ]
        }

        updated = await ExtractionOrchestrator(pipeline=mock_pipeline).reconcile_document_phases(
            ["reconcile-work_0", "reconcile-work_1"],
            db_session,
        )

        assert {episode.arc_phase.value for episode in updated} == {"setup", "resolution"}
        assert (await repository.get_by_id(first.id)).arc_phase.value == "setup"
        assert (await repository.get_by_id(second.id)).arc_phase.value == "resolution"

    @pytest.mark.asyncio
    async def test_cross_document_linking_creates_reviewable_edge(self, mock_pipeline, db_session):
        from sqlalchemy import select

        from narrative_engine.models import Episode
        from narrative_engine.storage.orm_models import EpisodeLinkORM
        from narrative_engine.storage.repositories import EpisodeRepository

        earlier = Episode(
            title="Convention adopts reform",
            summary="The convention adopted a suffrage reform.",
            start_year=1917,
            scope_id="uk_womens_suffrage_movement",
            scope_name="Women's Suffrage Movement",
            extracted_from=["older-source_2"],
        )
        newer = Episode(
            title="Voting reform enacted",
            summary="The reform became law in the following year.",
            start_year=1918,
            scope_id="uk_womens_suffrage_movement",
            scope_name="Women's Suffrage Movement",
            extracted_from=["new-source_0"],
        )
        repository = EpisodeRepository(db_session)
        await repository.create(earlier)
        await repository.create(newer)
        mock_pipeline.link.return_value = {
            "relationship": "same_event",
            "confidence": 0.87,
            "reasoning": "Two sources describe the same reform sequence.",
        }

        linked = await ExtractionOrchestrator(pipeline=mock_pipeline).link_document_candidates(
            ["new-source_0"],
            db_session,
        )

        edges = (await db_session.execute(select(EpisodeLinkORM))).scalars().all()
        assert linked == 1
        assert len(edges) == 1
        assert edges[0].edge_kind == "same_event_as"
        assert edges[0].review_status == "pending"

    @pytest.mark.asyncio
    async def test_process_text_persists_attested_causal_links(self, mock_pipeline):
        from narrative_engine.storage.orm_models import EpisodeLinkORM

        source = "Cause text. Effect text."
        mock_pipeline.segment.return_value = {
            "episodes": [
                {"number": 1, "summary": "Cause", "text": "Cause text."},
                {"number": 2, "summary": "Effect", "text": "Effect text."},
            ]
        }
        mock_pipeline.extract.side_effect = [
            {"title": "Cause", "summary": "Cause", "actors": [], "setting": {}},
            {"title": "Effect", "summary": "Effect", "actors": [], "setting": {}},
        ]
        mock_pipeline.classify.return_value = {
            "arc_type": "hero_journey",
            "arc_phase": "setup",
            "phase_confidence": 0.8,
        }
        mock_pipeline.link.return_value = {
            "relationship": "causes",
            "confidence": 0.9,
            "reasoning": "Explicit causal statement",
            "evidence_quote": "Cause",
        }
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        result = await ExtractionOrchestrator(
            pipeline=mock_pipeline,
            config=ExtractionPipelineConfig(enable_chunk_linking=True),
        ).process_text(source, "chunk", session)

        links = [call.args[0] for call in session.add.call_args_list if isinstance(call.args[0], EpisodeLinkORM)]
        assert result.errors == []
        assert len(links) == 1
        assert links[0].edge_kind == "causes"
        assert links[0].link_status == "attested"
        assert links[0].evidence == "Cause"

    @pytest.mark.asyncio
    async def test_process_text_rejects_causal_link_without_quote(self, mock_pipeline):
        source = "A. B."
        mock_pipeline.segment.return_value = {
            "episodes": [
                {"number": 1, "summary": "A", "text": "A."},
                {"number": 2, "summary": "B", "text": "B."},
            ]
        }
        mock_pipeline.extract.side_effect = [
            {"title": "A", "summary": "A", "actors": [], "setting": {}},
            {"title": "B", "summary": "B", "actors": [], "setting": {}},
        ]
        mock_pipeline.classify.return_value = {
            "arc_type": "hero_journey",
            "arc_phase": "setup",
            "phase_confidence": 0.8,
        }
        mock_pipeline.link.return_value = {
            "relationship": "causes",
            "confidence": 0.9,
            "reasoning": "Unsupported assertion",
        }
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        result = await ExtractionOrchestrator(
            pipeline=mock_pipeline,
            config=ExtractionPipelineConfig(enable_chunk_linking=True),
        ).process_text(source, "chunk", session)

        assert result.errors == []
        assert mock_pipeline.link.await_count == 1
        from narrative_engine.storage.orm_models import EpisodeLinkORM

        links = [call.args[0] for call in session.add.call_args_list if isinstance(call.args[0], EpisodeLinkORM)]
        assert links == []

    @pytest.mark.asyncio
    async def test_linking_prefilter_skips_implausible_non_neighbor_pairs(self, mock_pipeline):
        source = "Alpha event. Beta event. Gamma event."
        mock_pipeline.segment.return_value = {
            "episodes": [
                {"number": 1, "summary": "Alpha", "text": "Alpha event."},
                {"number": 2, "summary": "Beta", "text": "Beta event."},
                {"number": 3, "summary": "Gamma", "text": "Gamma event."},
            ]
        }
        mock_pipeline.extract.side_effect = [
            {
                "title": "Alpha",
                "summary": "Roman grain reforms",
                "actors": [],
                "setting": {"location": "Rome", "start_date": "0100"},
            },
            {
                "title": "Beta",
                "summary": "Mughal succession dispute",
                "actors": [],
                "setting": {"location": "Delhi", "start_date": "1600"},
            },
            {
                "title": "Gamma",
                "summary": "Japanese technology boom",
                "actors": [],
                "setting": {"location": "Tokyo", "start_date": "2000"},
            },
        ]
        mock_pipeline.link.return_value = {"relationship": "unrelated", "confidence": 0.99}
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        config = ExtractionPipelineConfig(enable_classification=False)

        result = await ExtractionOrchestrator(pipeline=mock_pipeline, config=config).process_text(
            source,
            "chunk",
            session,
        )

        assert result.errors == []
        assert mock_pipeline.link.await_count == 2

    def test_linking_prefilter_keeps_distant_pair_with_shared_actor(self, mock_pipeline):
        from narrative_engine.models import Actor, Episode

        first = Episode(
            title="First crisis",
            summary="A government falls",
            start_date=datetime(1800, 1, 1, tzinfo=timezone.utc),
            actors=[Actor(name="House of Example", role="ruler")],
        )
        second = Episode(
            title="Later restoration",
            summary="A dynasty returns",
            start_date=datetime(1900, 1, 1, tzinfo=timezone.utc),
            actors=[Actor(name="house OF example", role="claimant")],
        )

        assert ExtractionOrchestrator(pipeline=mock_pipeline)._is_link_candidate(first, second, distance=4)

    @pytest.mark.asyncio
    async def test_link_pair_rejects_quote_not_present_in_episode_text(self, mock_pipeline):
        from narrative_engine.models import Episode

        mock_pipeline.link.return_value = {
            "relationship": "causes",
            "confidence": 0.9,
            "evidence_quote": "A hallucinated quotation",
        }
        orchestrator = ExtractionOrchestrator(pipeline=mock_pipeline)

        linked = await orchestrator._link_episode_pair(
            Episode(title="A", summary="Documented cause"),
            Episode(title="B", summary="Documented effect"),
            AsyncMock(),
        )

        assert linked is False

    @pytest.mark.asyncio
    async def test_process_text_with_errors(self, mock_pipeline):
        """Test pipeline with stage errors."""
        # Make segmentation fail
        mock_pipeline.segment.side_effect = Exception("Segmentation failed")

        orchestrator = ExtractionOrchestrator(pipeline=mock_pipeline)

        mock_session = AsyncMock()

        result = await orchestrator.process_text(
            text="Test text",
            source_chunk_id="test-chunk",
            session=mock_session,
        )

        assert len(result.episodes) == 0
        assert len(result.errors) > 0
        assert "Segmentation failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_process_text_falls_back_when_segmentation_llm_fails(self, mock_pipeline):
        """Optional segmentation must not discard an extractable chunk."""
        mock_pipeline.segment.side_effect = LLMError("Response truncated")
        mock_pipeline.extract.return_value = {
            "title": "Fallback episode",
            "summary": "The full bounded chunk was extracted.",
            "actors": [],
            "setting": {},
        }
        mock_pipeline.classify.return_value = {}
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        result = await ExtractionOrchestrator(pipeline=mock_pipeline).process_text(
            text="A source chunk that remains safe to extract.",
            source_chunk_id="chunk-1",
            session=session,
        )

        assert result.errors == []
        assert [episode.title for episode in result.episodes] == ["Fallback episode"]
        assert mock_pipeline.extract.await_args.kwargs["segment_text"] == (
            "A source chunk that remains safe to extract."
        )

    @pytest.mark.asyncio
    async def test_classify_episode_mechanism_tags(self, mock_pipeline):
        """Unknown mechanism tags from the LLM are skipped, not fatal."""
        from narrative_engine.models import Episode, MechanismTag

        mock_pipeline.classify.return_value = {
            "arc_type": "credit_boom_and_bust",
            "arc_phase": "panic",
            "phase_confidence": 0.9,
            "rationale": "Test",
            "secondary_arcs": [],
            "mechanism_tags": ["credit_expansion", "asset_bubble", "not_a_real_mechanism"],
        }

        orchestrator = ExtractionOrchestrator(pipeline=mock_pipeline)
        episode = Episode(title="Test", summary="Test")

        await orchestrator._classify_episode(episode)

        assert episode.mechanism_tags == [
            MechanismTag.CREDIT_EXPANSION,
            MechanismTag.ASSET_BUBBLE,
        ]

    @pytest.mark.asyncio
    async def test_classify_episode_normalizes_observed_renewal_phase(self, mock_pipeline):
        """Semantic phase drift must not silently erase an otherwise valid arc."""
        from narrative_engine.models import ArcPhase, ArcType, Episode

        mock_pipeline.classify.return_value = {
            "arc_type": "decadence_and_renewal",
            "arc_phase": "renewal",
            "phase_confidence": 0.9,
            "secondary_arcs": [
                {
                    "type": "decadence_and_renewal",
                    "phase": "renewal",
                    "confidence": 0.8,
                }
            ],
        }
        episode = Episode(title="Reorganization", summary="Institutions recover.")

        await ExtractionOrchestrator(pipeline=mock_pipeline)._classify_episode(episode)

        assert episode.arc_phase == ArcPhase.RESOLUTION
        assert episode.secondary_arcs == [
            (ArcType.DECADENCE_AND_RENEWAL, ArcPhase.RESOLUTION, 0.8)
        ]

    @pytest.mark.asyncio
    async def test_invalid_phase_clears_the_incomplete_legacy_arc(self, mock_pipeline):
        """A high-confidence type without an orderable phase is not an arc."""
        from narrative_engine.models import ChangePattern, Episode

        mock_pipeline.classify.return_value = {
            "change_pattern": "tension_and_contestation",
            "pattern_confidence": 0.9,
            "arc_type": "siege_and_collapse",
            "arc_phase": "resistance",
            "phase_confidence": 0.9,
            "secondary_arcs": [],
        }
        episode = Episode(title="Resistance", summary="A group resisted an incumbent.")

        await ExtractionOrchestrator(pipeline=mock_pipeline)._classify_episode(episode)

        assert episode.change_pattern == ChangePattern.TENSION_AND_CONTESTATION
        assert episode.arc_type is None
        assert episode.arc_phase is None
        assert episode.classification_state.value == "classified"

    @pytest.mark.asyncio
    async def test_classify_episode_tolerates_scalar_collection_fields(self, mock_pipeline):
        """A malformed optional list from an LLM must not fail the document."""
        from narrative_engine.models import (
            Episode,
            MechanismFamily,
            MechanismTag,
            SituationDomain,
        )

        mock_pipeline.classify.return_value = {
            "domains": "political",
            "mechanism_families": "competition_displacement",
            "mechanism_tags": "identity_polarization",
            "secondary_arcs": "none",
        }
        episode = Episode(title="Unification", summary="A polity was unified.")

        await ExtractionOrchestrator(pipeline=mock_pipeline)._classify_episode(episode)

        assert episode.domains == [SituationDomain.POLITICAL]
        assert episode.mechanism_families == [MechanismFamily.COMPETITION_DISPLACEMENT]
        assert episode.mechanism_tags == [MechanismTag.IDENTITY_POLARIZATION]
        assert episode.secondary_arcs == []

    @pytest.mark.asyncio
    async def test_extract_segment_tolerates_scalar_and_malformed_collections(self, mock_pipeline):
        mock_pipeline.extract.return_value = {
            "title": "A reform begins",
            "summary": "A reform coalition formed.",
            "setting": "China",
            "actors": ["not-an-actor", {"name": "Reformers", "role": "challenger"}],
            "initiating_conditions": "Institutional pressure",
            "escalation_mechanics": None,
            "consequences": ["A new coalition formed", 42],
        }

        episode = await ExtractionOrchestrator(pipeline=mock_pipeline)._extract_segment(
            {"text": "Source text", "summary": "Reform"},
            "Source text",
            "chunk-1",
        )

        assert episode is not None
        assert episode.location is None
        assert episode.initiating_conditions == ["Institutional pressure"]
        assert episode.escalation_mechanics == []
        assert episode.consequences == ["A new coalition formed"]
        assert [actor.name for actor in episode.actors] == ["Reformers"]
        assert episode.source_passages[0].text == "Source text"
        assert episode.source_passages[0].passage_id.startswith("chunk-1:")

    @pytest.mark.asyncio
    async def test_extracts_unregistered_movement_inside_parent_scope(self, mock_pipeline):
        from narrative_engine.models import ScopeKind

        mock_pipeline.extract.return_value = {
            "title": "A reform discourse spreads",
            "summary": "A loose set of political ideas gained institutional uptake.",
            "setting": {"location": "United States"},
            "actors": [],
            "focal_scope": {
                "name": "Example reform discourse",
                "kind": "discourse",
                "parent_name": "United States",
                "confidence": 0.82,
                "evidence_quote": "the new language spread through universities",
                "boundary_note": "A contested umbrella label, not a membership organization.",
            },
        }
        orchestrator = ExtractionOrchestrator(pipeline=mock_pipeline)

        episode = await orchestrator._extract_segment(
            {"text": "Source text", "summary": "Ideas spread"},
            "Source text",
            "chunk-1",
        )

        assert episode is not None
        assert episode.scope_id is None
        assert episode.scope_name == "Example reform discourse"
        assert episode.scope_kind == ScopeKind.DISCOURSE
        assert episode.parent_scope_name == "United States"
        assert episode.scope_confidence == 0.82
        assert episode.scope_notes and "contested" in episode.scope_notes

    @pytest.mark.asyncio
    async def test_constrained_scope_stage_accepts_only_high_confidence_candidate(self, mock_pipeline):
        mock_pipeline.extract.return_value = {
            "title": "Militant suffrage expands",
            "summary": "British suffragists escalated their campaign.",
            "setting": {"location": "United Kingdom", "start_date": "1903"},
            "actors": [],
            "focal_scope": {
                "name": "British suffragists",
                "kind": "movement",
                "parent_name": "United Kingdom",
                "confidence": 0.91,
                "evidence_quote": "British suffragists escalated",
            },
        }
        mock_pipeline.canonicalize_scope.return_value = {
            "scope_id": "uk_womens_suffrage_movement",
            "confidence": 0.94,
            "reason": "The raw plural denotes the registered movement.",
        }

        episode = await ExtractionOrchestrator(pipeline=mock_pipeline)._extract_segment(
            {"text": "British suffragists escalated their campaign.", "summary": "Campaign"},
            "British suffragists escalated their campaign.",
            "suffrage_1",
        )

        assert episode is not None
        assert episode.scope_id == "uk_womens_suffrage_movement"
        assert episode.scope_notes and episode.scope_notes.startswith("Registry match:")

    @pytest.mark.asyncio
    async def test_classifies_scale_neutral_group_pattern(self, mock_pipeline):
        from narrative_engine.models import (
            ChangePattern,
            Episode,
            MechanismFamily,
            SituationDomain,
            SituationScale,
        )

        mock_pipeline.classify.return_value = {
            "change_pattern": "emergence_and_gathering",
            "pattern_confidence": 0.9,
            "pattern_rationale": "Local cells consolidate into a movement.",
            "situation_scale": "organization",
            "domains": ["political", "organizational", "not-a-domain"],
            "configuration": {
                "capacity": 0.7,
                "cohesion": 0.6,
                "pressure": 0.2,
                "legitimacy": -0.1,
                "adaptability": 0.8,
                "agency": -0.4,
                "invented_axis": 1.0,
            },
            "mechanism_families": [
                "cooperation_alignment",
                "contagion_diffusion",
                "not-a-family",
            ],
            "mechanism_tags": [],
            "arc_type": None,
            "arc_phase": None,
            "phase_confidence": 0.0,
        }
        orchestrator = ExtractionOrchestrator(pipeline=mock_pipeline)
        episode = Episode(title="Movement forms", summary="Local cells unite.")

        await orchestrator._classify_episode(episode)

        assert episode.change_pattern == ChangePattern.EMERGENCE_AND_GATHERING
        assert episode.situation_scale == SituationScale.ORGANIZATION
        assert episode.domains == [
            SituationDomain.POLITICAL,
            SituationDomain.ORGANIZATIONAL,
        ]
        assert episode.configuration.capacity == 0.7
        assert episode.mechanism_families == [
            MechanismFamily.COOPERATION_ALIGNMENT,
            MechanismFamily.CONTAGION_DIFFUSION,
        ]
        assert episode.classification_state.value == "classified"

    @pytest.mark.asyncio
    async def test_parse_date_range(self, mock_pipeline):
        """Test date range parsing."""
        orchestrator = ExtractionOrchestrator(pipeline=mock_pipeline)

        from narrative_engine.models import Episode

        episode = Episode(title="Test", summary="Test")

        # Test year range
        result = await orchestrator._parse_dates(episode, "1921-1923", "range")
        assert result.start_date.year == 1921
        assert result.end_date.year == 1923

        # Test single year
        result = await orchestrator._parse_dates(episode, "1929", "year")
        assert result.start_date.year == 1929

    @pytest.mark.asyncio
    async def test_parse_month_name_containing_to_as_single_date(self, mock_pipeline):
        from narrative_engine.models import Episode

        episode = await ExtractionOrchestrator(pipeline=mock_pipeline)._parse_dates(
            Episode(title="Preface", summary="Publication date"),
            "October 1869",
            "month",
        )

        assert episode.start_date is not None
        assert (episode.start_date.year, episode.start_date.month) == (1869, 10)
        assert episode.end_date is None
        assert episode.date_precision == "month"

    @pytest.mark.asyncio
    async def test_extract_segment_uses_llm_normalized_dates(self, mock_pipeline):
        mock_pipeline.extract.return_value = {
            "title": "Publication",
            "summary": "The work was published.",
            "setting": {
                "location": "London",
                "time_period_label": "October 1869",
                "start_date": "1869-10",
                "end_date": None,
                "date_precision": "month",
                "date_basis": "explicit",
                "date_confidence": 0.99,
            },
            "actors": [],
        }

        episode = await ExtractionOrchestrator(pipeline=mock_pipeline)._extract_segment(
            {"summary": "Publication", "text": "Published in October 1869."},
            "Published in October 1869.",
            "chunk-1",
        )

        assert episode is not None
        assert episode.start_date is not None
        assert (episode.start_date.year, episode.start_date.month) == (1869, 10)
        assert episode.end_date is None
        assert episode.start_year == 1869
        assert episode.end_year is None
        assert episode.date_precision == "month"

    @pytest.mark.asyncio
    async def test_extract_segment_preserves_bce_years_without_datetime(self, mock_pipeline):
        mock_pipeline.extract.return_value = {
            "title": "Battle of Kadesh",
            "summary": "Egypt and the Hittites fought at Kadesh.",
            "setting": {
                "location": "Kadesh",
                "time_period_label": "1274 BCE",
                "start_date": "-1274",
                "end_date": "-1274",
                "date_precision": "year",
            },
            "actors": [],
        }

        episode = await ExtractionOrchestrator(pipeline=mock_pipeline)._extract_segment(
            {"summary": "Battle", "text": "The battle took place in 1274 BCE."},
            "The battle took place in 1274 BCE.",
            "chunk-bce",
        )

        assert episode.start_year == -1274
        assert episode.end_year == -1274
        assert episode.start_date is None
        assert episode.end_date is None

    @pytest.mark.asyncio
    async def test_process_batch(self, mock_pipeline):
        """Test batch processing."""
        mock_pipeline.segment.return_value = {"episodes": []}

        orchestrator = ExtractionOrchestrator(pipeline=mock_pipeline)

        mock_session = AsyncMock()

        chunks = [
            {"id": "chunk-1", "text": "Text 1"},
            {"id": "chunk-2", "text": "Text 2"},
        ]

        results = await orchestrator.process_batch(chunks, mock_session)

        assert len(results) == 2
        assert results[0].source_chunk_id == "chunk-1"
        assert results[1].source_chunk_id == "chunk-2"


class TestPrompts:
    """Tests for prompt templates."""

    def test_segmentation_prompt_structure(self):
        """Test segmentation prompt contains required elements."""
        from narrative_engine.extraction.prompts import get_segmentation_prompt

        prompt = get_segmentation_prompt("Sample text")

        assert "episode" in prompt.lower()
        assert "json" in prompt.lower()
        assert "Sample text" in prompt
        assert "beginning" in prompt.lower()
        assert "tension" in prompt.lower()
        assert "start_char" in prompt
        assert "end_char" in prompt
        assert "verbatim" in prompt.lower()

    def test_extraction_prompt_structure(self):
        """Test extraction prompt contains required fields."""
        from narrative_engine.extraction.prompts import get_extraction_prompt

        prompt = get_extraction_prompt("Segment text", "Summary")

        assert "title" in prompt.lower()
        assert "actors" in prompt.lower()
        assert "setting" in prompt.lower()
        assert "initiating_conditions" in prompt.lower()
        assert "escalation" in prompt.lower()
        assert "time_period_label" in prompt
        assert "start_date" in prompt
        assert "date_basis" in prompt
        assert "never invent" in prompt.lower()
        assert "json" in prompt.lower()

    def test_classification_prompt_structure(self):
        """Test classification prompt contains arc types."""
        from narrative_engine.extraction.prompts import get_classification_prompt

        prompt = get_classification_prompt("Summary", "Full text")

        assert "arc_type" in prompt.lower() or "arc type" in prompt.lower()
        assert "phase" in prompt.lower()
        assert "confidence" in prompt.lower()
        assert "credit_boom_and_bust" in prompt or "boom" in prompt.lower()
