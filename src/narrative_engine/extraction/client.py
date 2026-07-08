"""LLM client for extraction pipeline."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Type, TypeVar
from abc import ABC, abstractmethod

import openai
import structlog
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from narrative_engine.extraction.config import LLMConfig

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM operations."""
    
    pass


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Type[T]] = None,
    ) -> Dict[str, Any]:
        """Send completion request to LLM."""
        pass
    
    @abstractmethod
    async def complete_with_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send completion request and parse JSON response."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
        )
    
    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Type[T]] = None,
    ) -> Dict[str, Any]:
        """Send completion request to OpenAI."""
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise historical data extraction system. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if response_format else None,
            )
            
            content = response.choices[0].message.content
            if not content:
                raise LLMError("Empty response from OpenAI")
            
            return {
                "content": content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            }
        
        except openai.BadRequestError as e:
            logger.error("OpenAI bad request", error=str(e), model=model)
            raise LLMError(f"Bad request: {e}") from e
        except openai.AuthenticationError as e:
            logger.error("OpenAI authentication failed", error=str(e))
            raise LLMError("Authentication failed—check API key") from e
        except Exception as e:
            logger.error("OpenAI completion failed", error=str(e))
            raise LLMError(f"Completion failed: {e}") from e
    
    async def complete_with_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send completion request and parse JSON."""
        result = await self.complete(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=dict,  # Request JSON mode
        )
        
        try:
            return json.loads(result["content"])
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", content=result["content"][:200])
            raise LLMError(f"Invalid JSON response: {e}") from e


class ExtractionPipeline:
    """Main extraction pipeline coordinating LLM calls."""
    
    def __init__(
        self,
        client: Optional[LLMClient] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.client = client or self._create_default_client()
        self.config = config
        self.logger = structlog.get_logger()
    
    def _create_default_client(self) -> LLMClient:
        """Create default LLM client from environment."""
        llm_config = LLMConfig.from_env()
        
        if llm_config.provider == "openai":
            return OpenAIClient(llm_config)
        # elif llm_config.provider == "anthropic":
        #     return AnthropicClient(llm_config)
        else:
            raise LLMError(f"Unsupported provider: {llm_config.provider}")
    
    async def segment(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage 1: Segment text into episodes."""
        from narrative_engine.extraction.prompts import get_segmentation_prompt
        
        self.logger.info("Starting segmentation", text_length=len(text))
        
        prompt = get_segmentation_prompt(text)
        result = await self.client.complete_with_json(
            prompt=prompt,
            model=model or (self.config.segmentation_model if self.config else None),
        )
        
        self.logger.info(
            "Segmentation complete",
            episodes_found=len(result.get("episodes", [])),
        )
        
        return result
    
    async def extract(
        self,
        segment_text: str,
        segment_summary: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage 2: Extract structured data from segment."""
        from narrative_engine.extraction.prompts import get_extraction_prompt
        
        self.logger.info("Starting extraction", segment_summary=segment_summary[:50])
        
        prompt = get_extraction_prompt(segment_text, segment_summary)
        result = await self.client.complete_with_json(
            prompt=prompt,
            model=model or (self.config.extraction_model if self.config else None),
        )
        
        self.logger.info("Extraction complete", episode_title=result.get("title", "Unknown"))
        
        return result
    
    async def classify(
        self,
        episode_summary: str,
        full_text: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage 3: Classify arc type and phase."""
        from narrative_engine.extraction.prompts import get_classification_prompt
        
        self.logger.info("Starting classification", summary=episode_summary[:50])
        
        prompt = get_classification_prompt(episode_summary, full_text)
        result = await self.client.complete_with_json(
            prompt=prompt,
            model=model or (self.config.classification_model if self.config else None),
            temperature=0.0,  # Deterministic for classification
        )
        
        self.logger.info(
            "Classification complete",
            arc_type=result.get("arc_type"),
            arc_phase=result.get("arc_phase"),
            confidence=result.get("phase_confidence"),
        )
        
        return result
    
    async def classify_second_pass(
        self,
        episode_summary: str,
        initial_classification: Dict[str, Any],
        similar_episodes: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Second-pass classification with nearest-neighbor guidance."""
        from narrative_engine.extraction.prompts import get_classification_second_pass_prompt
        
        if not self.config or not self.config.two_pass_classification:
            return initial_classification
        
        self.logger.info("Starting second-pass classification")
        
        prompt = get_classification_second_pass_prompt(
            episode_summary,
            initial_classification,
            similar_episodes,
        )
        result = await self.client.complete_with_json(
            prompt=prompt,
            model=model or (self.config.classification_model if self.config else None),
            temperature=0.0,
        )
        
        self.logger.info(
            "Second-pass complete",
            changed=result.get("changed_from_initial", False),
        )
        
        return result
    
    async def link(
        self,
        episode1: Dict[str, Any],
        episode2: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage 4: Determine relationship between two episodes."""
        from narrative_engine.extraction.prompts import get_linking_prompt
        
        self.logger.info(
            "Starting linking",
            episode1=episode1.get("title", "Unknown"),
            episode2=episode2.get("title", "Unknown"),
        )
        
        prompt = get_linking_prompt(episode1, episode2)
        result = await self.client.complete_with_json(
            prompt=prompt,
            model=model or (self.config.linking_model if self.config else None),
        )
        
        self.logger.info(
            "Linking complete",
            relationship=result.get("relationship"),
            confidence=result.get("confidence"),
        )
        
        return result
