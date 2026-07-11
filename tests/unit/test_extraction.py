"""Unit tests for extraction pipeline."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

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

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("NE_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("NE_LLM_MODEL", "claude-3-opus")
        monkeypatch.setenv("NE_LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("NE_LLM_MAX_TOKENS", "8000")

        config = LLMConfig.from_env()

        assert config.provider == "anthropic"
        assert config.model == "claude-3-opus"
        assert config.temperature == 0.5
        assert config.max_tokens == 8000

    def test_config_reads_openai_compatible_base_url(self, monkeypatch):
        monkeypatch.setenv("NE_LLM_BASE_URL", "https://api.venice.ai/api/v1")

        config = LLMConfig.from_env()

        assert config.base_url == "https://api.venice.ai/api/v1"


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
        assert config.extraction_model == "claude-sonnet-5"

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
        for var in ("NE_SEG_MODEL", "NE_EXTRACT_MODEL", "NE_CLASSIFY_MODEL", "NE_LINK_MODEL"):
            monkeypatch.delenv(var, raising=False)

        config = ExtractionPipelineConfig.from_env()

        assert all(
            not model.startswith("claude-")
            for model in (
                config.segmentation_model,
                config.extraction_model,
                config.classification_model,
                config.linking_model,
            )
        )

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
    async def test_process_text_persists_attested_causal_links(self, mock_pipeline):
        from narrative_engine.storage.orm_models import EpisodeLinkORM

        mock_pipeline.segment.return_value = {
            "episodes": [
                {"number": 1, "summary": "Cause", "text": "Cause text"},
                {"number": 2, "summary": "Effect", "text": "Effect text"},
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

        result = await ExtractionOrchestrator(pipeline=mock_pipeline).process_text("Text", "chunk", session)

        links = [call.args[0] for call in session.add.call_args_list if isinstance(call.args[0], EpisodeLinkORM)]
        assert result.errors == []
        assert len(links) == 1
        assert links[0].edge_kind == "causes"
        assert links[0].link_status == "attested"
        assert links[0].evidence == "Cause"

    @pytest.mark.asyncio
    async def test_process_text_rejects_causal_link_without_quote(self, mock_pipeline):
        mock_pipeline.segment.return_value = {
            "episodes": [
                {"number": 1, "summary": "A", "text": "A"},
                {"number": 2, "summary": "B", "text": "B"},
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

        result = await ExtractionOrchestrator(pipeline=mock_pipeline).process_text("Text", "chunk", session)

        assert any("evidence quote" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_link_pair_rejects_quote_not_present_in_episode_text(self, mock_pipeline):
        from narrative_engine.models import Episode

        mock_pipeline.link.return_value = {
            "relationship": "causes",
            "confidence": 0.9,
            "evidence_quote": "A hallucinated quotation",
        }
        orchestrator = ExtractionOrchestrator(pipeline=mock_pipeline)

        with pytest.raises(ValueError, match="not present"):
            await orchestrator._link_episode_pair(
                Episode(title="A", summary="Documented cause"),
                Episode(title="B", summary="Documented effect"),
                AsyncMock(),
            )

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

    def test_extraction_prompt_structure(self):
        """Test extraction prompt contains required fields."""
        from narrative_engine.extraction.prompts import get_extraction_prompt

        prompt = get_extraction_prompt("Segment text", "Summary")

        assert "title" in prompt.lower()
        assert "actors" in prompt.lower()
        assert "setting" in prompt.lower()
        assert "initiating_conditions" in prompt.lower()
        assert "escalation" in prompt.lower()
        assert "json" in prompt.lower()

    def test_classification_prompt_structure(self):
        """Test classification prompt contains arc types."""
        from narrative_engine.extraction.prompts import get_classification_prompt

        prompt = get_classification_prompt("Summary", "Full text")

        assert "arc_type" in prompt.lower() or "arc type" in prompt.lower()
        assert "phase" in prompt.lower()
        assert "confidence" in prompt.lower()
        assert "credit_boom_and_bust" in prompt or "boom" in prompt.lower()
