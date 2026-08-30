"""Versioned prompt templates for LLM extraction pipeline."""

from __future__ import annotations

from typing import Dict, List

from narrative_engine.extraction.config import (
    CHANGE_PATTERN_DESCRIPTIONS,
    CONFIGURATION_DIMENSIONS,
    DEFAULT_ARC_TAXONOMY,
    MECHANISM_DESCRIPTIONS,
    MECHANISM_FAMILY_DESCRIPTIONS,
)
from narrative_engine.models import ScopeKind, SituationDomain, SituationScale

# Prompt versions for tracking
PROMPT_VERSIONS = {
    "segmentation": "1.3.0",  # 1.3.0: bounded, source-backed, cost-capped events
    "extraction": "2.1.0",  # durable focal scope from bounded neighboring context
    "scope": "1.0.0",  # constrained selection from registry candidates
    "classification": "2.0.0",  # configuration/change-pattern ontology
    "reconciliation": "1.0.0",  # document-level chronological phase review
    "linking": "1.0.0",
}


def _role_vocabulary_block() -> str:
    """Serialized controlled role vocabulary for the extraction prompt."""
    from narrative_engine.extraction.roles import ActorRole

    return ", ".join(role.value for role in ActorRole)


def get_segmentation_prompt(text: str) -> str:
    """Prompt for identifying episode boundaries.

    Stage 1: Split text into discrete narrative units (episodes).
    """
    return f"""You are a historical narrative analyzer. Your task is to identify distinct episodes (bounded narrative units) in the provided text.  # noqa: E501

An **episode** is a self-contained historical situation with:
- A clear beginning (initiating conditions)
- A tension or conflict
- A resolution or ongoing development
- Specific actors and time period

**Instructions:**
1. Identify the historically meaningful episodes in the text. Return at most
   8. Combine incidental anecdotes or repeated descriptions into the nearest
   causally coherent episode; do not treat the entire chapter as one episode
   merely because it has one heading.
2. For each episode, provide:
   - Episode number (1, 2, 3...)
   - One-line summary (20 words max)
   - Beginning state (what kicked it off, 20 words max)
   - Key tension (what's at stake, 20 words max)
   - Current status (resolved or ongoing)
   - start_char: zero-based character offset where the episode begins
   - end_char: zero-based, end-exclusive character offset where it ends
   - start_quote: an exact 4-12 word verbatim quote from the beginning of the span
   - end_quote: an exact 4-12 word verbatim quote from the end of the span

Character offsets refer only to the text between the delimiters below.
Spans must be ordered, non-overlapping, and copied from the source. Prefer
boundaries at paragraph breaks. Never paraphrase the boundary quotes.

**Output format:** Return JSON with this structure:
{{"episodes": [{{"number": 1, "summary": "...", "beginning": "...", "tension": "...", "status": "resolved|ongoing", "start_char": 0, "end_char": 123, "start_quote": "exact source text", "end_quote": "exact source text"}}]}}

If no distinct episodes found, return empty array.

**Text to analyze:**
---
{text}
---

Return only valid JSON."""


def get_extraction_prompt(
    segment_text: str,
    segment_summary: str,
    narrative_context: str | None = None,
) -> str:
    """Prompt for extracting structured data from an episode.

    Stage 2: Pull actors, conditions, mechanics, resolution from segment.
    """
    scope_kinds = ", ".join(kind.value for kind in ScopeKind)
    return f"""You are extracting structured data from a narrative segment. The subject may be a person, relationship, group, organization, polity, civilization, or system. Extract all relevant information without privileging finance or state-level events.

**Context:** This segment describes: {segment_summary}

**Nearby event summaries from the same source chunk:**
{narrative_context or 'No additional context supplied.'}

Use the nearby summaries only to identify the durable trajectory that this
episode updates. Every factual field and evidence quote must still be grounded
in the segment text below; do not import facts from a neighboring event.

**Instructions:** Extract the following fields and return as JSON:

1. **title** (string): A concise title for this episode (max 10 words)

2. **summary** (string): A 2-3 sentence summary of what happened

3. **actors** (array): List of significant actors with:
   - name: Actor name
   - role: Their role in this episode, in your own words (free text)
   - canonical_role: The best-fitting STRUCTURAL position from this controlled
     vocabulary (or null if none genuinely fits — do not force a choice):
     {_role_vocabulary_block()}
     Roles name structural positions, not costumes: a "court" is any power
     center where proximity to a principal outweighs formal office (an
     administration's inner circle, a founder-CEO's kitchen cabinet, a
     politburo). Roles can be filled by COLLECTIVE actors — an investor
     syndicate can be a kingmaker, a movement can be a pretender.
   - role_fit_confidence: 0.0-1.0, how well the canonical_role fits

4. **setting** (object):
   - location: Where it took place
   - time_period_label: Preserve the source's original wording for when it occurred
   - start_date: Normalize CE dates to "YYYY", "YYYY-MM", or "YYYY-MM-DD".
     Represent BCE years with a leading minus (for example, "-1274" means
     1274 BCE); there is no year zero. Use null when legendary, disputed, or
     not safely resolvable
   - end_date: Same normalized formats, or null for a single date/unknown end
   - date_precision: "day", "month", "year", "range", or "unknown"
   - date_basis: "explicit", "inferred", "legendary", or "unknown"
   - date_confidence: 0.0-1.0 confidence in the normalized date. Never invent a
     date for mythic eras or vague labels; preserve the label and use null dates.

5. **initiating_conditions** (array): What started this episode? (3-5 bullet points)

6. **escalation_mechanics** (array): How did tension build? What dynamics drove it? (3-5 bullet points)

7. **tension** (string): The core conflict or what's at stake (1 sentence)

8. **resolution** (string or null): How it ended, or "ongoing" if not resolved

9. **consequences** (array): What happened afterward? Immediate and downstream effects (3-5 bullet points)

10. **focal_scope** (object): The durable bounded subject whose formation,
    growth, contestation, decline, or renewal this episode is evidence about.
    Ask: "whose longer trajectory would this event be one data point in?"
    Prefer continuity across nearby events over whichever actor performs the
    immediate action. An office-holder, temporary assembly, law, battle, or
    crowd normally belongs in actors/title, not focal_scope, unless its own
    durable trajectory is genuinely the subject. Choose a subgroup, party,
    movement, or family of ideas when the text tracks that subgroup's path;
    do not default to its parent polity. When an event could describe both a
    challenger's rise and an incumbent's decline, follow the trajectory
    emphasized across nearby summaries and record the ambiguity in
    boundary_note rather than switching among transient actors.
   - name: The conventional full name supported by the source, or null. Prefer
     a durable subject that can recur across episodes; never use an event title
     or ad-hoc description as a scope name. Keep naming consistent (for
     example, "Julius Caesar", not alternating between "Caesar" and "Julius
     Caesar").
   - kind: One of {scope_kinds}, or null
   - parent_name: Nearest containing group, organization, polity, or other
     larger scope when supported (for example faction -> party -> polity)
   - confidence: 0.0-1.0. Do not infer a parent merely from stereotypes.
   - evidence_quote: Short exact quote supporting the scope identification
   - boundary_note: Explain ambiguity or contested labels (for example, when a
     source treats a loose family of ideas as if it were a coherent group)

**Text to analyze:**
---
{segment_text}
---

Return only valid JSON matching this schema. Use null for unknown fields."""


def get_scope_canonicalization_prompt(
    raw_name: str,
    raw_kind: str | None,
    parent_name: str | None,
    evidence_quote: str | None,
    candidates: List[Dict],
) -> str:
    """Constrain entity normalization to retrieved registry candidates."""
    return f"""You are resolving one source-backed historical subject against a versioned scope registry.

Raw subject: {raw_name}
Raw kind: {raw_kind or 'unknown'}
Containing subject: {parent_name or 'unknown'}
Evidence quote: {evidence_quote or 'not supplied'}

Candidate registry entries:
{candidates}

Choose a candidate only when it denotes the same durable subject—not merely a
related person, parent polity, location, event, or similarly named group. If
none is the same subject, return null. Never invent an id.

Return only JSON:
{{
  "scope_id": "one candidate id or null",
  "confidence": 0.0,
  "reason": "brief identity justification"
}}"""


def get_classification_prompt(episode_summary: str, full_text: str) -> str:
    """Prompt for a scale-neutral reading plus optional legacy arc labels.

    Stage 3: Assign arc type, phase, and confidence.
    """
    arc_descriptions = "\n".join(
        [
            f"- {key}: {value['description']}\n  Typical trajectory: {', '.join(value['phases'])}"
            for key, value in DEFAULT_ARC_TAXONOMY.items()
        ]
    )

    mechanism_descriptions = "\n".join(
        [f"- {tag}: {description}" for tag, description in MECHANISM_DESCRIPTIONS.items()]
    )

    change_descriptions = "\n".join(
        [f"- {pattern}: {description}" for pattern, description in CHANGE_PATTERN_DESCRIPTIONS.items()]
    )
    family_descriptions = "\n".join(
        [f"- {family}: {description}" for family, description in MECHANISM_FAMILY_DESCRIPTIONS.items()]
    )
    dimensions = "\n".join(
        [f"- {dimension}: {description}" for dimension, description in CONFIGURATION_DIMENSIONS.items()]
    )
    scales = ", ".join(scale.value for scale in SituationScale)
    domains = ", ".join(domain.value for domain in SituationDomain)

    return f"""You are a scale-neutral change-pattern classifier. Analyze the situation whether it concerns a person, relationship, faction, organization, polity, civilization, or system. Finance is one possible domain, not the default frame.

**Episode summary:**
{episode_summary}

**Full episode text (for context):**
---
{full_text}
---

**Primary change patterns:**
{change_descriptions}

**Situation scales:** {scales}

**Domains** (choose every clearly material facet): {domains}

**Configuration axes** (score -1.0 to +1.0; use null when unsupported):
{dimensions}

**Broad mechanism families:**
{family_descriptions}

**Optional legacy arc types** (compatibility/specialised interpretation):
{arc_descriptions}

**Standard narrative phases:**
- setup: Initial conditions, exposition
- rising_action: Building tension, escalation
- climax: Peak moment, turning point
- falling_action: Consequences unfold
- resolution: Final outcome, denouement

**Financial cycle phases (for credit_boom_and_bust):**
- boom: Expansion phase
- euphoria: Peak optimism, peak speculation
- distress: First signs of trouble
- panic: Crash, rapid decline
- revulsion: Despair, avoidance of asset class

**Optional detailed mechanisms** (tag only clearly supported mechanisms):
{mechanism_descriptions}

**Instructions:**
1. Identify the primary scale-neutral change pattern and its confidence.
2. Identify the focal scale and material domains. A faction's rise is a
   group-scale situation even if the surrounding context is civilizational.
3. Score the six configuration axes; use null instead of inventing evidence.
4. Tag broad mechanism families and any clearly supported detailed mechanisms.
5. Add a legacy arc reading only when it genuinely helps. Legacy arc_type and
   arc_phase may be null; never force a financial, civilizational, or literary
   label. arc_phase must be one of setup, rising_action, climax,
   falling_action, resolution, boom, euphoria, distress, panic, revulsion.

**Output format (JSON):**
{{
  "change_pattern": "tension_and_contestation",
  "pattern_confidence": 0.88,
  "pattern_rationale": "A challenger faction is openly contesting the incumbent coalition",
  "situation_scale": "group",
  "domains": ["organizational", "political"],
  "configuration": {{
    "capacity": 0.2,
    "cohesion": -0.6,
    "pressure": 0.8,
    "legitimacy": -0.4,
    "adaptability": 0.1,
    "agency": 0.5
  }},
  "mechanism_families": ["legitimacy_erosion", "competition_displacement"],
  "mechanism_tags": ["identity_polarization"],
  "arc_type": "reform_then_reaction",
  "arc_phase": "rising_action",
  "phase_confidence": 0.67,
  "rationale": "Useful secondary interpretation",
  "secondary_arcs": []
}}

Return only valid JSON. Null is preferable to a weak forced label."""


def get_classification_second_pass_prompt(
    episode_summary: str,
    initial_classification: Dict,
    similar_episodes: List[Dict],
) -> str:
    """Prompt for second-pass classification with nearest-neighbor guidance.

    Improves label stability across corpus.
    """
    similar_text = "\n".join(
        [
            f"- {ep.get('title', 'Unknown')}: classified as {ep.get('arc_type', 'unknown')}, {ep.get('arc_phase', 'unknown')}"
            for ep in similar_episodes[:5]
        ]
    )

    return f"""You are refining a narrative classification. Review the initial classification in light of similar episodes.

**Episode:**
{episode_summary}

**Initial classification:**
- Arc type: {initial_classification.get("arc_type", "unknown")}
- Phase: {initial_classification.get("arc_phase", "unknown")}
- Confidence: {initial_classification.get("phase_confidence", 0)}

**Similar episodes (already classified):**
{similar_text}

**Instructions:**
1. Does the initial classification align with similar episodes?
2. Should the arc type or phase be adjusted?
3. Provide refined classification with updated confidence

**Output format (JSON):**
{{
  "arc_type": "...",
  "arc_phase": "...",
  "phase_confidence": 0.0-1.0,
  "rationale": "...",
  "changed_from_initial": true/false,
  "reason_for_change": "..." (if changed)
}}

Return only valid JSON."""


def get_phase_reconciliation_prompt(scope_name: str, episodes: List[Dict]) -> str:
    """Review per-event labels with the surrounding document chronology."""
    arc_types = ", ".join(DEFAULT_ARC_TAXONOMY)
    return f"""You are reviewing a chronological set of source-backed historical episodes about one focal subject: {scope_name}.

Episodes, in chronological order:
{episodes}

For each episode, decide whether its existing legacy arc type and narrative
phase make sense relative to the surrounding trajectory. This is a
reconciliation pass, not a request to force one grand story onto unrelated
events. Preserve null when the evidence does not establish a legacy arc.

Allowed arc types: {arc_types}
Allowed phases: setup, rising_action, climax, falling_action, resolution,
boom, euphoria, distress, panic, revulsion.

Rules:
- A phase is relative to the focal subject's trajectory, not the chapter's
  position or the emotional intensity of one paragraph.
- Multiple independent sources may legitimately document the same phase.
- Use falling_action for consequences unfolding after a turning point and
  resolution only for a comparatively settled outcome.
- Never change dates, scope, actors, or factual summaries.
- Use confidence below 0.5 or null labels when context remains ambiguous.

Return only JSON:
{{
  "episodes": [
    {{
      "episode_id": "uuid from input",
      "arc_type": "allowed value or null",
      "arc_phase": "allowed value or null",
      "confidence": 0.0,
      "reason": "brief contextual justification"
    }}
  ]
}}"""


def get_linking_prompt(episode1: Dict, episode2: Dict) -> str:
    """Prompt for entity resolution and causal linking.

    Stage 4: Determine if two episodes describe same event or are causally related.
    """
    def evidence_text(episode: Dict) -> str:
        passages = episode.get("source_passages")
        if isinstance(passages, list) and passages and isinstance(passages[0], dict):
            text = passages[0].get("text")
            if isinstance(text, str):
                return text[:1600]
        return str(episode.get("summary", "Unknown"))

    return f"""You are analyzing the relationship between two historical episodes for entity resolution.

**Episode 1:**
- Title: {episode1.get("title", "Unknown")}
- Summary: {episode1.get("summary", "Unknown")}
- Time: {episode1.get("start_year") or episode1.get("start_date") or "Unknown"}
- Source evidence: {evidence_text(episode1)}

**Episode 2:**
- Title: {episode2.get("title", "Unknown")}
- Summary: {episode2.get("summary", "Unknown")}
- Time: {episode2.get("start_year") or episode2.get("start_date") or "Unknown"}
- Source evidence: {evidence_text(episode2)}

**Possible relationships:**
1. **same_event**: Episodes describe the same historical event (different sources, same occurrence)
2. **causes**: Episode 1 caused or led to Episode 2
3. **caused_by**: Episode 2 caused or led to Episode 1
4. **related**: Related but no direct causal link (same era, connected themes)
5. **unrelated**: Distinct events with no meaningful connection

**Instructions:**
- Determine the relationship type
- Provide confidence (0.0-1.0)
- Explain reasoning
- If causal, specify mechanism if apparent
- If causal, provide a verbatim quote from the supplied source evidence
  that explicitly supports the direction of causation. Without such a quote,
  return "related" rather than making a causal claim.

**Output format (JSON):**
{{
  "relationship": "same_event|causes|caused_by|related|unrelated",
  "confidence": 0.0-1.0,
  "reasoning": "...",
  "evidence_quote": "verbatim supporting quote" (required if causal),
  "mechanism": "..." (if causal)
}}

Return only valid JSON."""


def get_causal_linking_prompt(source_episode: Dict, target_episodes: List[Dict]) -> str:
    """Prompt for finding causal connections from source to potential targets.

    Identifies downstream consequences.
    """
    targets_text = "\n".join(
        [
            f"{i + 1}. {ep.get('title', 'Unknown')}: {ep.get('summary', 'Unknown')[:100]}..."
            for i, ep in enumerate(target_episodes[:10])
        ]
    )

    return f"""You are identifying causal connections between historical events.

**Source episode:**
Title: {source_episode.get("title", "Unknown")}
Summary: {source_episode.get("summary", "Unknown")}

**Potential downstream episodes:**
{targets_text}

**Instructions:**
For each potential target episode, determine:
1. Is there a causal connection from source to target?
2. If yes, what is the mechanism?
3. Confidence in causal claim (0.0-1.0)

**Output format (JSON):**
{{
  "causal_links": [
    {{"target_index": 1, "is_causal": true, "mechanism": "...", "confidence": 0.85}},
    {{"target_index": 2, "is_causal": false, "confidence": 0.3}}
  ]
}}

Be conservative—only mark as causal if there's clear evidence of influence."""
