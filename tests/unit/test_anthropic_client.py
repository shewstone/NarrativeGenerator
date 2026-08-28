"""AnthropicClient tests (T9, docs/tickets/T9-anthropic-llm-client.md)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from narrative_engine.extraction.client import (
    AnthropicClient,
    ExtractionPipeline,
    LLMError,
    OpenAIClient,
)
from narrative_engine.extraction.config import ExtractionPipelineConfig, LLMConfig


def _response(
    text='{"ok": true}',
    stop_reason="end_turn",
    model="claude-sonnet-5",
):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def _client(response=None, config=None):
    fake_sdk = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=response or _response())),
        close=AsyncMock(),
    )
    client = AnthropicClient(
        config or LLMConfig(provider="anthropic", model="claude-sonnet-5"),
        client=fake_sdk,
    )
    return client, fake_sdk.messages.create


class TestRequestShape:
    @pytest.mark.asyncio
    async def test_no_sampling_parameters_ever(
        self,
    ):
        """Current Claude models 400 on temperature/top_p — must never be sent,
        even when a caller passes temperature (classify passes 0.0)."""
        client, create = _client()

        await client.complete("prompt", temperature=0.0)

        kwargs = create.call_args.kwargs
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "top_k" not in kwargs
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["max_tokens"] == 4000

    @pytest.mark.asyncio
    async def test_model_and_max_tokens_overridable(self):
        client, create = _client()

        await client.complete("prompt", model="claude-haiku-4-5", max_tokens=512)

        kwargs = create.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_usage_mapped_to_pipeline_shape(self):
        client, _ = _client()

        result = await client.complete("prompt")

        assert result["usage"] == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }

    @pytest.mark.asyncio
    async def test_close_releases_sdk_transport(self):
        client, _ = _client()

        await client.aclose()

        client.client.close.assert_awaited_once()


class TestStopReasons:
    @pytest.mark.asyncio
    async def test_refusal_surfaces_as_llm_error(self):
        client, _ = _client(_response(stop_reason="refusal"))

        with pytest.raises(LLMError, match="refused"):
            await client.complete("prompt")

    @pytest.mark.asyncio
    async def test_truncation_surfaces_as_llm_error(self):
        client, _ = _client(_response(stop_reason="max_tokens"))

        with pytest.raises(LLMError, match="truncated"):
            await client.complete("prompt")


class TestJsonParsing:
    @pytest.mark.asyncio
    async def test_plain_json(self):
        client, _ = _client(_response('{"arc_type": "credit_boom_and_bust"}'))
        result = await client.complete_with_json("prompt")
        assert result == {"arc_type": "credit_boom_and_bust"}

    @pytest.mark.asyncio
    async def test_fenced_json_is_stripped(self):
        client, _ = _client(_response('```json\n{"episodes": []}\n```'))
        result = await client.complete_with_json("prompt")
        assert result == {"episodes": []}

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        client, _ = _client(_response("I could not produce JSON, sorry."))
        with pytest.raises(LLMError, match="Invalid JSON"):
            await client.complete_with_json("prompt")


class TestProviderFactory:
    def test_anthropic_provider_constructs_client(self, monkeypatch):
        monkeypatch.setenv("NE_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        pipeline = ExtractionPipeline()
        assert isinstance(pipeline.client, AnthropicClient)

    def test_openai_path_still_works(self, monkeypatch):
        monkeypatch.setenv("NE_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        pipeline = ExtractionPipeline()
        assert isinstance(pipeline.client, OpenAIClient)


class TestDefaults:
    def test_no_retired_model_defaults(self, monkeypatch):
        for var in (
            "NE_LLM_PROVIDER",
            "NE_LLM_MODEL",
            "NE_SEG_MODEL",
            "NE_EXTRACT_MODEL",
            "NE_CLASSIFY_MODEL",
            "NE_LINK_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)

        llm = LLMConfig.from_env()
        assert llm.provider == "anthropic"
        assert llm.model == "claude-sonnet-5"

        stages = ExtractionPipelineConfig.from_env()
        # Sec 7 routing: cheap segmentation, strong everything else
        assert stages.segmentation_model == "claude-haiku-4-5"
        assert stages.extraction_model == "claude-sonnet-5"
        assert stages.classification_model == "claude-sonnet-5"
        assert stages.linking_model == "claude-sonnet-5"
        # The retired claude-3-opus-20240229 must be gone everywhere
        assert "20240229" not in repr(llm) + repr(stages)
