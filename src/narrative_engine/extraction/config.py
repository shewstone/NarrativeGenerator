"""Configuration for LLM extraction pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str = "openai"  # openai, anthropic
    model: str = "gpt-4"
    temperature: float = 0.0  # Low for deterministic extraction
    max_tokens: int = 4000
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls, prefix: str = "NE_") -> LLMConfig:
        """Create config from environment variables."""
        provider = os.getenv(f"{prefix}LLM_PROVIDER", "openai")

        # Default models by provider
        default_models = {
            "openai": "gpt-4",
            "anthropic": "claude-3-opus-20240229",
        }

        return cls(
            provider=provider,
            model=os.getenv(f"{prefix}LLM_MODEL", default_models.get(provider, "gpt-4")),
            temperature=float(os.getenv(f"{prefix}LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv(f"{prefix}LLM_MAX_TOKENS", "4000")),
            api_key=os.getenv(f"{prefix}LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        )


@dataclass(frozen=True)
class ExtractionPipelineConfig:
    """Configuration for extraction pipeline stages."""

    # Stage enablement
    enable_segmentation: bool = True
    enable_extraction: bool = True
    enable_classification: bool = True
    enable_linking: bool = True

    # Model routing by stage (cheap → expensive)
    segmentation_model: str = "gpt-3.5-turbo"  # Cheap, fast
    extraction_model: str = "gpt-4"  # Good structured output
    classification_model: str = "gpt-4"  # Requires judgment
    linking_model: str = "gpt-4"  # Complex reasoning

    # Chunk settings
    chunk_size_tokens: int = 6000  # ~2-8k tokens per chunk
    chunk_overlap_tokens: int = 500  # Context overlap

    # Classification settings
    classification_temperature: float = 0.0
    two_pass_classification: bool = True

    # Entity resolution settings
    similarity_threshold: float = 0.85  # For same-event detection

    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000

    @classmethod
    def from_env(cls, prefix: str = "NE_") -> ExtractionPipelineConfig:
        """Create config from environment variables."""
        return cls(
            enable_segmentation=os.getenv(f"{prefix}ENABLE_SEGMENTATION", "true").lower() == "true",
            enable_extraction=os.getenv(f"{prefix}ENABLE_EXTRACTION", "true").lower() == "true",
            enable_classification=os.getenv(f"{prefix}ENABLE_CLASSIFICATION", "true").lower() == "true",
            enable_linking=os.getenv(f"{prefix}ENABLE_LINKING", "true").lower() == "true",
            segmentation_model=os.getenv(f"{prefix}SEG_MODEL", "gpt-3.5-turbo"),
            extraction_model=os.getenv(f"{prefix}EXTRACT_MODEL", "gpt-4"),
            classification_model=os.getenv(f"{prefix}CLASSIFY_MODEL", "gpt-4"),
            linking_model=os.getenv(f"{prefix}LINK_MODEL", "gpt-4"),
            chunk_size_tokens=int(os.getenv(f"{prefix}CHUNK_SIZE", "6000")),
            chunk_overlap_tokens=int(os.getenv(f"{prefix}CHUNK_OVERLAP", "500")),
        )


@dataclass(frozen=True)
class PromptVersions:
    """Versioned prompt templates."""

    segmentation_version: str = "v1.0.0"
    extraction_version: str = "v1.0.0"
    classification_version: str = "v1.0.0"
    linking_version: str = "v1.0.0"


# Taxonomies are versioned artifacts (design doc Sec 7): re-running with a
# new version is a batch job, not a rewrite. Single source of truth so a
# version bump doesn't require hunting down every literal.
CURRENT_TAXONOMY_VERSION = "arc-v0.1.0"

# Default arc taxonomy for prompts
DEFAULT_ARC_TAXONOMY = {
    "rise_and_overextension": {
        "description": "Growth phase followed by exceeding sustainable limits",
        "phases": ["emergence", "acceleration", "overextension", "correction"],
    },
    "hubris_nemesis": {
        "description": "Excessive pride leading to downfall",
        "phases": ["rise", "hubris", "challenge", "nemesis", "catharsis"],
    },
    "reform_then_reaction": {
        "description": "Change triggering backlash and reversal",
        "phases": ["status_quo", "reform", "resistance", "reaction", "equilibrium"],
    },
    "decadence_and_renewal": {
        "description": "Decline followed by regeneration",
        "phases": ["florescence", "decadence", "crisis", "renewal", "growth"],
    },
    "siege_and_collapse": {
        "description": "External pressure leading to breakdown",
        "phases": ["threat_emergence", "resistance", "siege", "collapse", "aftermath"],
    },
    "succession_crisis": {
        "description": "Leadership transition causing instability",
        "phases": ["predecessor", "transition", "contestation", "resolution", "consolidation"],
    },
    "credit_boom_and_bust": {
        "description": "Financial expansion and contraction (Minsky-Kindleberger)",
        "phases": ["boom", "euphoria", "distress", "panic", "revulsion"],
    },
    "generational_forgetting": {
        "description": "Lessons lost between generations leading to repeated mistakes",
        "phases": [
            "crisis_memory",
            "institutionalization",
            "generational_shift",
            "erosion",
            "repetition",
        ],
    },
    "hero_journey": {
        "description": "Classic departure-initiation-return arc",
        "phases": ["departure", "initiation", "ordeal", "return", "mastery"],
    },
    "tragedy": {
        "description": "Fatal flaw leading to inevitable downfall",
        "phases": ["exposition", "rising_action", "climax", "falling_action", "catastrophe"],
    },
    "comedy": {
        "description": "Confusion leading to recognition and union",
        "phases": ["normality", "confusion", "complication", "clarification", "union"],
    },
    "rebirth": {
        "description": "Death and transformation leading to new life",
        "phases": ["fullness", "death", "winter", "awakening", "rebirth"],
    },
    "voyage_return": {
        "description": "Journey to strange lands and return transformed",
        "phases": ["departure", "trials", "encounter", "return", "integration"],
    },
    "rags_to_riches": {
        "description": "Rise from obscurity to success, threat, and final triumph",
        "phases": ["initial_state", "acquisition", "peak", "loss", "final_success"],
    },
}


# Standard narrative phases
STANDARD_PHASES = [
    "setup",
    "rising_action",
    "climax",
    "falling_action",
    "resolution",
]


FINANCIAL_PHASES = [
    "boom",
    "euphoria",
    "distress",
    "panic",
    "revulsion",
]
