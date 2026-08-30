"""Configuration for LLM extraction pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from narrative_engine.models import ChangePattern, MechanismFamily, MechanismTag


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
    request_timeout_seconds: float = 90.0
    # Venice reasoning models otherwise spend tokens narrating simple JSON
    # decisions and may leave message.content empty. Omit this for providers
    # or models that do not support the OpenAI-compatible extension.
    reasoning_effort: Optional[str] = None

    @classmethod
    def from_env(cls, prefix: str = "NE_") -> LLMConfig:
        """Create config from environment variables."""
        provider = os.getenv(f"{prefix}LLM_PROVIDER", "anthropic")
        provider_api_key = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(provider)

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
            api_key=os.getenv(f"{prefix}LLM_API_KEY") or (os.getenv(provider_api_key) if provider_api_key else None),
            base_url=os.getenv(f"{prefix}LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            request_timeout_seconds=float(
                os.getenv(f"{prefix}LLM_REQUEST_TIMEOUT_SECONDS", "90")
            ),
            reasoning_effort=os.getenv(f"{prefix}LLM_REASONING_EFFORT") or None,
        )


@dataclass(frozen=True)
class ExtractionPipelineConfig:
    """Configuration for extraction pipeline stages."""

    # Stage enablement
    enable_segmentation: bool = True
    enable_extraction: bool = True
    enable_classification: bool = True
    enable_linking: bool = True
    # Legacy all-pairs linking inside every chunk duplicates the bounded
    # document-level linker and can dominate ingestion cost for an eight-event
    # segment. Keep it available for focused callers, but disable it in the
    # local batch configuration.
    enable_chunk_linking: bool = True

    # Model routing by stage (design doc Sec 7: "cheap model for
    # segmentation, strong model for extraction/classification/synthesis")
    segmentation_model: str = "claude-haiku-4-5"  # Cheap, fast
    scope_model: str = "claude-haiku-4-5"  # Constrained registry selection
    extraction_model: str = "claude-sonnet-5"  # Structured episode records
    reconciliation_model: str = "claude-sonnet-5"  # Document-level phase judgment
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

    # Scope is a hard composition partition, so a wrong focal scope is more
    # damaging than an unresolved one. Low-confidence claims remain visible
    # on the episode but are not promoted to canonical scope ids.
    scope_confidence_floor: float = 0.6

    # Entity resolution settings
    similarity_threshold: float = 0.85  # For same-event detection
    # Always inspect nearby episodes. More distant pairs must share an actor,
    # place, temporal window, or enough meaningful vocabulary before an LLM
    # call is made, preventing quadratic all-pairs spend on obvious negatives.
    linking_neighbor_window: int = 1
    linking_max_year_gap: float = 25.0
    linking_min_lexical_overlap: float = 0.2
    cross_link_max_pairs: int = 20

    # Independent episode calls within one chunk are I/O-bound. A small
    # bounded fan-out removes serial network latency without producing a
    # burst large enough to overwhelm Venice or the shared DB session.
    stage_concurrency: int = 3

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
            enable_chunk_linking=os.getenv(f"{prefix}ENABLE_CHUNK_LINKING", "true").lower()
            == "true",
            segmentation_model=os.getenv(f"{prefix}SEG_MODEL", defaults["segmentation"]),
            scope_model=os.getenv(f"{prefix}SCOPE_MODEL", defaults["segmentation"]),
            extraction_model=os.getenv(f"{prefix}EXTRACT_MODEL", defaults["strong"]),
            reconciliation_model=os.getenv(f"{prefix}RECONCILE_MODEL", defaults["strong"]),
            classification_model=os.getenv(f"{prefix}CLASSIFY_MODEL", defaults["strong"]),
            linking_model=os.getenv(f"{prefix}LINK_MODEL", defaults["strong"]),
            chunk_size_tokens=int(os.getenv(f"{prefix}CHUNK_SIZE", "6000")),
            chunk_overlap_tokens=int(os.getenv(f"{prefix}CHUNK_OVERLAP", "500")),
            classification_confidence_floor=float(os.getenv(f"{prefix}TAU_CLASS", "0.5")),
            role_fit_floor=float(os.getenv(f"{prefix}TAU_ROLE", "0.5")),
            scope_confidence_floor=float(os.getenv(f"{prefix}TAU_SCOPE", "0.6")),
            linking_neighbor_window=int(os.getenv(f"{prefix}LINK_NEIGHBOR_WINDOW", "1")),
            linking_max_year_gap=float(os.getenv(f"{prefix}LINK_MAX_YEAR_GAP", "25")),
            linking_min_lexical_overlap=float(os.getenv(f"{prefix}LINK_MIN_LEXICAL_OVERLAP", "0.2")),
            cross_link_max_pairs=int(os.getenv(f"{prefix}CROSS_LINK_MAX_PAIRS", "20")),
            stage_concurrency=int(os.getenv(f"{prefix}STAGE_CONCURRENCY", "3")),
        )
        if config.linking_neighbor_window < 0:
            raise ValueError("LINK_NEIGHBOR_WINDOW must be non-negative")
        if config.linking_max_year_gap < 0:
            raise ValueError("LINK_MAX_YEAR_GAP must be non-negative")
        if not 0.0 <= config.linking_min_lexical_overlap <= 1.0:
            raise ValueError("LINK_MIN_LEXICAL_OVERLAP must be between 0 and 1")
        if config.cross_link_max_pairs < 0:
            raise ValueError("CROSS_LINK_MAX_PAIRS must be non-negative")
        if not 1 <= config.stage_concurrency <= 8:
            raise ValueError("STAGE_CONCURRENCY must be between 1 and 8")
        for name, value in (
            ("TAU_CLASS", config.classification_confidence_floor),
            ("TAU_ROLE", config.role_fit_floor),
            ("TAU_SCOPE", config.scope_confidence_floor),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        incompatible_prefix = "claude-" if provider == "openai" else "gpt-"
        for model in (
            config.segmentation_model,
            config.scope_model,
            config.extraction_model,
            config.reconciliation_model,
            config.classification_model,
            config.linking_model,
        ):
            if model.startswith(incompatible_prefix):
                raise ValueError(f"LLM provider {provider!r} is incompatible with stage model {model!r}")
        return config


@dataclass(frozen=True)
class PromptVersions:
    """Versioned prompt templates."""

    segmentation_version: str = "v1.3.0"
    scope_version: str = "v1.0.0"
    extraction_version: str = "v2.1.0"
    classification_version: str = "v2.0.0"
    reconciliation_version: str = "v1.0.0"
    linking_version: str = "v1.0.0"


# Taxonomies are versioned artifacts (design doc Sec 7): re-running with a
# new version is a batch job, not a rewrite. Single source of truth so a
# version bump doesn't require hunting down every literal.
CURRENT_TAXONOMY_VERSION = "arc-v0.1.0"
CURRENT_SITUATION_ONTOLOGY_VERSION = "situation-v1.0.0"

# Primary, scale-neutral change vocabulary. These are deliberately phrased
# without market, state, civilization, or literary-protagonist assumptions.
CHANGE_PATTERN_DESCRIPTIONS = {
    ChangePattern.EMERGENCE_AND_GATHERING.value: "A new identity, capability, or coalition takes shape",
    ChangePattern.EXPANSION_AND_CONSOLIDATION.value: "Reach grows while gains, rules, or relationships are stabilized",
    ChangePattern.SATURATION_AND_OVERREACH.value: "Growth meets limits and commitments exceed sustainable capacity",
    ChangePattern.TENSION_AND_CONTESTATION.value: "Competing claims or pressures become active and visible",
    ChangePattern.FRAGMENTATION_AND_RELEASE.value: "A previously held-together structure separates or disperses",
    ChangePattern.RETREAT_AND_PRESERVATION.value: "Exposure is reduced to protect a viable core",
    ChangePattern.TURNING_AND_REORIENTATION.value: "A threshold or reversal redirects attention, strategy, or identity",
    ChangePattern.RENEWAL_AND_INTEGRATION.value: "Capacity and coherence are rebuilt in a revised form",
    ChangePattern.SUCCESSION_AND_TRANSFER.value: "Authority, responsibility, memory, or resources pass to new holders",
}

MECHANISM_FAMILY_DESCRIPTIONS = {
    MechanismFamily.AMPLIFICATION_FEEDBACK.value: "A change reinforces itself through a feedback loop",
    MechanismFamily.RESOURCE_STRAIN.value: "Demand on time, energy, money, people, or material exceeds supply",
    MechanismFamily.LEGITIMACY_EROSION.value: "Trust in a person, norm, leadership, or order declines",
    MechanismFamily.COORDINATION_FAILURE.value: "Actors cannot align information, incentives, or action",
    MechanismFamily.BOUNDARY_PRESSURE.value: "External conditions or neighboring actors stress the focal scope",
    MechanismFamily.SUCCESSION_DYNAMICS.value: "Transfer of authority or identity creates rivalry or discontinuity",
    MechanismFamily.INSTITUTIONAL_LOCK_IN.value: "Established commitments constrain adaptation",
    MechanismFamily.ADAPTATION_LEARNING.value: "Feedback changes behavior, structure, or capability",
    MechanismFamily.CONTAGION_DIFFUSION.value: "Ideas, behavior, confidence, or disruption spread between actors",
    MechanismFamily.COOPERATION_ALIGNMENT.value: "Mutual adjustment increases shared capacity or cohesion",
    MechanismFamily.COMPETITION_DISPLACEMENT.value: "One actor, practice, or coalition gains at another's expense",
    MechanismFamily.MEMORY_LOSS.value: "Relevant experience or safeguards fade, allowing recurrence",
}

CONFIGURATION_DIMENSIONS = {
    "capacity": "-1 depleted/constrained; +1 abundant/growing",
    "cohesion": "-1 fragmented; +1 integrated",
    "pressure": "-1 latent/low; +1 acute",
    "legitimacy": "-1 contested; +1 accepted",
    "adaptability": "-1 rigid; +1 flexible",
    "agency": "-1 distributed; +1 concentrated",
}

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
