"""Configuration for LLM extraction pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from narrative_engine.models import MechanismTag


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str = "anthropic"  # anthropic, openai
    model: str = "claude-sonnet-5"
    # NOTE: temperature applies to the OpenAI path only. Current Claude
    # models (Sonnet 5, Opus 4.8/4.7) removed sampling parameters — the
    # AnthropicClient never sends it (a request carrying it returns 400).
    temperature: float = 0.0
    max_tokens: int = 4000
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @classmethod
    def from_env(cls, prefix: str = "NE_") -> LLMConfig:
        """Create config from environment variables."""
        provider = os.getenv(f"{prefix}LLM_PROVIDER", "anthropic")

        # Default models by provider (T9: Sonnet chosen 2026-07-11 —
        # near-Opus extraction quality at Sonnet cost; per-stage overrides
        # below make upgrading any single stage a one-variable change)
        default_models = {
            "anthropic": "claude-sonnet-5",
            "openai": "gpt-4",
        }

        return cls(
            provider=provider,
            model=os.getenv(f"{prefix}LLM_MODEL", default_models.get(provider, "claude-sonnet-5")),
            temperature=float(os.getenv(f"{prefix}LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv(f"{prefix}LLM_MAX_TOKENS", "4000")),
            api_key=os.getenv(f"{prefix}LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv(f"{prefix}LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        )


@dataclass(frozen=True)
class ExtractionPipelineConfig:
    """Configuration for extraction pipeline stages."""

    # Stage enablement
    enable_segmentation: bool = True
    enable_extraction: bool = True
    enable_classification: bool = True
    enable_linking: bool = True

    # Model routing by stage (design doc Sec 7: "cheap model for
    # segmentation, strong model for extraction/classification/synthesis")
    segmentation_model: str = "claude-haiku-4-5"  # Cheap, fast
    extraction_model: str = "claude-sonnet-5"  # Structured episode records
    classification_model: str = "claude-sonnet-5"  # Arc/phase judgment
    linking_model: str = "claude-sonnet-5"  # Causal-evidence extraction

    # Chunk settings
    chunk_size_tokens: int = 6000  # ~2-8k tokens per chunk
    chunk_overlap_tokens: int = 500  # Context overlap

    # Classification settings
    classification_temperature: float = 0.0
    two_pass_classification: bool = True
    # tau_class (design doc Sec 6.2 stage 4): classification is not a forced
    # choice. If no canonical arc clears this confidence floor, the episode
    # gets NO arc assignment and classification_state="unclassified".
    # UNTUNED HYPOTHESIS: tune against the analog fixture, not intuition
    # (Sec 9) -- too low pollutes the analog base, too high starves it.
    classification_confidence_floor: float = 0.5

    # tau_role (T2; Sec 10.5): same no-forced-choice discipline for actor
    # roles. Below this fit confidence, canonical_role stays None and the
    # mention counts as vocabulary residue. UNTUNED HYPOTHESIS, like
    # tau_class.
    role_fit_floor: float = 0.5

    # Entity resolution settings
    similarity_threshold: float = 0.85  # For same-event detection

    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000

    @classmethod
    def from_env(cls, prefix: str = "NE_") -> ExtractionPipelineConfig:
        """Create config from environment variables."""
        provider = os.getenv(f"{prefix}LLM_PROVIDER", "anthropic").lower()
        stage_defaults = {
            "anthropic": {
                "segmentation": "claude-haiku-4-5",
                "strong": "claude-sonnet-5",
            },
            "openai": {
                "segmentation": "gpt-4o-mini",
                "strong": "gpt-4o",
            },
        }
        if provider not in stage_defaults:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        defaults = stage_defaults[provider]
        config = cls(
            enable_segmentation=os.getenv(f"{prefix}ENABLE_SEGMENTATION", "true").lower() == "true",
            enable_extraction=os.getenv(f"{prefix}ENABLE_EXTRACTION", "true").lower() == "true",
            enable_classification=os.getenv(f"{prefix}ENABLE_CLASSIFICATION", "true").lower() == "true",
            enable_linking=os.getenv(f"{prefix}ENABLE_LINKING", "true").lower() == "true",
            segmentation_model=os.getenv(f"{prefix}SEG_MODEL", defaults["segmentation"]),
            extraction_model=os.getenv(f"{prefix}EXTRACT_MODEL", defaults["strong"]),
            classification_model=os.getenv(f"{prefix}CLASSIFY_MODEL", defaults["strong"]),
            linking_model=os.getenv(f"{prefix}LINK_MODEL", defaults["strong"]),
            chunk_size_tokens=int(os.getenv(f"{prefix}CHUNK_SIZE", "6000")),
            chunk_overlap_tokens=int(os.getenv(f"{prefix}CHUNK_OVERLAP", "500")),
            classification_confidence_floor=float(os.getenv(f"{prefix}TAU_CLASS", "0.5")),
            role_fit_floor=float(os.getenv(f"{prefix}TAU_ROLE", "0.5")),
        )
        incompatible_prefix = "claude-" if provider == "openai" else "gpt-"
        for model in (
            config.segmentation_model,
            config.extraction_model,
            config.classification_model,
            config.linking_model,
        ):
            if model.startswith(incompatible_prefix):
                raise ValueError(f"LLM provider {provider!r} is incompatible with stage model {model!r}")
        return config


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


# Mechanism vocabulary is a versioned artifact too (design doc Sec 3.8),
# same convention as CURRENT_TAXONOMY_VERSION above.
CURRENT_MECHANISM_VOCAB_VERSION = "mechanism-v0.1.0"

# Short descriptions for the classification prompt -- one per MechanismTag.
MECHANISM_DESCRIPTIONS = {
    MechanismTag.ELITE_OVERPRODUCTION.value: "More elite aspirants than elite positions available",
    MechanismTag.ELITE_INTRA_COMPETITION.value: "Elites turning on each other for scarce positions/resources",
    MechanismTag.POPULAR_IMMISERATION.value: "Falling living standards or well-being for the general population",
    MechanismTag.FISCAL_DISTRESS.value: "State revenue failing to cover state obligations",
    MechanismTag.STATE_FRAGILITY.value: "Weakening state capacity to enforce order or deliver services",
    MechanismTag.GEOPOLITICAL_PRESSURE.value: "External military, diplomatic, or economic pressure from rivals",
    MechanismTag.INSTITUTIONAL_DECAY.value: "Institutions losing effectiveness or legitimacy over time",
    MechanismTag.BUREAUCRATIC_SCLEROSIS.value: "Administrative rigidity slowing response to new conditions",
    MechanismTag.REGULATORY_CAPTURE.value: "Regulators serving the interests of the regulated",
    MechanismTag.CORRUPTION_SPIRAL.value: "Self-reinforcing growth of corrupt practice",
    MechanismTag.CREDIT_EXPANSION.value: "Rapid growth in lending/leverage",
    MechanismTag.DEBT_OVERHANG.value: "Accumulated debt suppressing future growth or investment",
    MechanismTag.CURRENCY_CRISIS.value: "Sudden loss of confidence in a currency's value",
    MechanismTag.ASSET_BUBBLE.value: "Asset prices detached from fundamentals",
    MechanismTag.GENERATIONAL_FORGETTING.value: "Lessons from a past crisis lost as generations turn over",
    MechanismTag.COHESION_EROSION.value: "Weakening of shared identity or mutual trust within a group",
    MechanismTag.IDENTITY_POLARIZATION.value: "Hardening of group identity into opposed factions",
    MechanismTag.CULTURAL_DECADENCE.value: "Declining vigor or discipline in cultural norms",
    MechanismTag.HUBRIS_CULTURE.value: "Overconfidence born of past success",
    MechanismTag.REFORM_RESISTANCE.value: "Entrenched interests blocking needed change",
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
