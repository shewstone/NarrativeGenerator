"""Versioned prompt templates for LLM extraction pipeline."""

from __future__ import annotations

from typing import Dict, List

from narrative_engine.extraction.config import DEFAULT_ARC_TAXONOMY, MECHANISM_DESCRIPTIONS

# Prompt versions for tracking
PROMPT_VERSIONS = {
    "segmentation": "1.0.0",
    "extraction": "1.1.0",  # 1.1.0: controlled role vocabulary (T2)
    "classification": "1.0.0",
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
1. Identify all distinct episodes in the text
2. For each episode, provide:
   - Episode number (1, 2, 3...)
   - One-line summary (20 words max)
   - Beginning state (what kicked it off)
   - Key tension (what's at stake)
   - Current status (resolved or ongoing)

**Output format:** Return JSON with this structure:
{{"episodes": [{{"number": 1, "summary": "...", "beginning": "...", "tension": "...", "status": "resolved|ongoing"}}]}}

If no distinct episodes found, return empty array.

**Text to analyze:**
---
{text}
---

Return only valid JSON."""


def get_extraction_prompt(segment_text: str, segment_summary: str) -> str:
    """Prompt for extracting structured data from an episode.

    Stage 2: Pull actors, conditions, mechanics, resolution from segment.
    """
    return f"""You are extracting structured historical data from a narrative segment. Extract all relevant information.

**Context:** This segment describes: {segment_summary}

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
   - time_period: When (e.g., "1921-1923", "October 1929", "Q4 2008")
   - date_precision: "year", "month", "day", or "range"

5. **initiating_conditions** (array): What started this episode? (3-5 bullet points)

6. **escalation_mechanics** (array): How did tension build? What dynamics drove it? (3-5 bullet points)

7. **tension** (string): The core conflict or what's at stake (1 sentence)

8. **resolution** (string or null): How it ended, or "ongoing" if not resolved

9. **consequences** (array): What happened afterward? Immediate and downstream effects (3-5 bullet points)

**Text to analyze:**
---
{segment_text}
---

Return only valid JSON matching this schema. Use null for unknown fields."""


def get_classification_prompt(episode_summary: str, full_text: str) -> str:
    """Prompt for classifying arc type and phase.

    Stage 3: Assign arc type, phase, and confidence.
    """
    arc_descriptions = "\n".join(
        [
            f"- {key}: {value['description']}\n  Phases: {', '.join(value['phases'])}"
            for key, value in DEFAULT_ARC_TAXONOMY.items()
        ]
    )

    mechanism_descriptions = "\n".join(
        [f"- {tag}: {description}" for tag, description in MECHANISM_DESCRIPTIONS.items()]
    )

    return f"""You are a narrative pattern classifier. Analyze this historical episode and classify its archetypal structure.

**Episode summary:**
{episode_summary}

**Full episode text (for context):**
---
{full_text}
---

**Available arc types:**
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

**Available mechanisms** (structural drivers -- tag any that are clearly at work in this episode; leave out mechanisms that don't apply):
{mechanism_descriptions}

**Instructions:**
1. Identify the PRIMARY arc type (most dominant pattern)
2. Identify the current phase in that arc
3. Consider if secondary arcs apply (episodes often instantiate multiple patterns)
4. Tag the mechanisms clearly present, from the vocabulary above only
5. Provide confidence score (0.0-1.0) and rationale

**Output format (JSON):**
{{
  "arc_type": "credit_boom_and_bust",
  "arc_phase": "panic",
  "phase_confidence": 0.92,
  "rationale": "Clear panic phase: bank runs, asset fire sales, contagion",
  "secondary_arcs": [
    {{"type": "hubris_nemesis", "phase": "nemesis", "confidence": 0.75}}
  ],
  "mechanism_tags": ["credit_expansion", "asset_bubble"]
}}

Return only valid JSON. Be decisive—choose the best fit even if imperfect."""


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
- Arc type: {initial_classification.get('arc_type', 'unknown')}
- Phase: {initial_classification.get('arc_phase', 'unknown')}
- Confidence: {initial_classification.get('phase_confidence', 0)}

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


def get_linking_prompt(episode1: Dict, episode2: Dict) -> str:
    """Prompt for entity resolution and causal linking.

    Stage 4: Determine if two episodes describe same event or are causally related.
    """
    return f"""You are analyzing the relationship between two historical episodes for entity resolution.

**Episode 1:**
- Title: {episode1.get('title', 'Unknown')}
- Summary: {episode1.get('summary', 'Unknown')}
- Time: {episode1.get('time_period', 'Unknown')}

**Episode 2:**
- Title: {episode2.get('title', 'Unknown')}
- Summary: {episode2.get('summary', 'Unknown')}
- Time: {episode2.get('time_period', 'Unknown')}

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
- If causal, provide a verbatim quote from the supplied episode summaries
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
            f"{i+1}. {ep.get('title', 'Unknown')}: {ep.get('summary', 'Unknown')[:100]}..."
            for i, ep in enumerate(target_episodes[:10])
        ]
    )

    return f"""You are identifying causal connections between historical events.

**Source episode:**
Title: {source_episode.get('title', 'Unknown')}
Summary: {source_episode.get('summary', 'Unknown')}

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
