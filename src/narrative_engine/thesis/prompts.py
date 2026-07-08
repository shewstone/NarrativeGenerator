"""Prompts for thesis generation and refinement."""

THESIS_GENERATION_PROMPT = """You are a narrative forecasting analyst. Given a query about a current situation and a set of historical analogs, synthesize a probabilistic forecast.

Query: {query}

Current Situation:
- Title: {episode_title}
- Summary: {episode_summary}
- Arc Type: {arc_type}
- Current Phase: {arc_phase}

Historical Analogs (ranked by relevance):
{analogs}

Task:
1. Identify the dominant pattern across analogs
2. Generate 2-3 alternative continuations with probabilities
3. Identify key indicators to watch
4. Note major uncertainties

Output JSON:
{{
    "dominant_pattern": "Brief description of the most likely outcome",
    "continuations": [
        {{"description": "Outcome 1", "probability": 0.6, "confidence": "high"}},
        {{"description": "Outcome 2", "probability": 0.3, "confidence": "medium"}},
        {{"description": "Outcome 3", "probability": 0.1, "confidence": "low"}}
    ],
    "key_indicators": ["indicator1", "indicator2", "indicator3"],
    "uncertainties": ["uncertainty1", "uncertainty2"],
    "rationale": "Brief explanation of reasoning"
}}

Rules:
- Probabilities must sum to 1.0
- Be specific about outcomes (avoid vague language)
- Cite specific analogs in rationale
- Express appropriate uncertainty
"""

THESIS_REFINEMENT_PROMPT = """Review and refine the following thesis based on feedback or new information.

Original Thesis:
{original_thesis}

New Context:
{new_context}

Refinement Type: {refinement_type}

Task:
- Update probabilities if needed
- Add/remove alternative outcomes
- Revise confidence level
- Update watch conditions

Output revised thesis in the same JSON format.
"""

BACKTESTING_PROMPT = """Evaluate this thesis against actual outcomes for backtesting.

Thesis (made at {thesis_date}):
{thesis}

Actual Outcome (observed at {outcome_date}):
{actual_outcome}

Task:
1. Which continuation was closest to actual outcome?
2. Calculate Brier score for the thesis
3. Identify what was missed or mispredicted
4. Note lessons for future forecasts

Output:
{{
    "matched_continuation": "Which predicted outcome matched",
    "brier_score": 0.XX,
    "accuracy_assessment": "accurate|partial|missed",
    "missed_factors": ["factor1", "factor2"],
    "lessons": "Brief lessons learned"
}}
"""

CONTINUATION_CLUSTERING_PROMPT = """Given multiple historical outcomes, cluster them into similar patterns.

Outcomes:
{outcomes}

Task:
Group these into 2-5 clusters based on similarity of outcome type (not just text similarity).

Output:
{{
    "clusters": [
        {{
            "label": "Pattern name (e.g., 'Soft Landing', 'Hard Crash')",
            "outcome_indices": [0, 2, 5],
            "description": "What characterizes this pattern"
        }}
    ]
}}
"""

THESIS_NARRATIVE_PROMPT = """You are a narrative forecasting analyst. Interpret the following algorithmic forecast results and provide a compelling narrative synthesis.

Algorithmic Forecast Results:
- Dominant Continuation: {dominant_continuation}
- Probability: {probability}
- Alternative Scenarios: {alternatives}
- Confidence: {confidence}
- Key Uncertainties: {uncertainties}

Historical Analogs Supporting This Forecast:
{analogs}

Task:
Write a 2-3 paragraph narrative synthesis that:
1. Explains the dominant pattern in accessible terms
2. Describes what makes this situation similar to the historical analogs
3. Acknowledges the key uncertainties and alternative paths
4. Uses specific examples from the analogs

Tone: Analytical but accessible. Like a good financial journalist or historian explaining the pattern.

Output:
{{
    "narrative_summary": "The 2-3 paragraph synthesis",
    "key_pattern": "One-sentence summary of the dominant pattern",
    "analog_strength": "Why these analogs are relevant",
    "dissenting_view": "Brief acknowledgment of why it might be different this time"
}}
"""


def format_analogs(analogs: list) -> str:
    """Format analogs for prompt."""
    lines = []
    for i, analog in enumerate(analogs[:5], 1):
        episode = analog.episode
        lines.append(f"{i}. {episode.title} (Relevance: {analog.combined_score:.2f})")
        lines.append(f"   Arc: {episode.arc_type.value if episode.arc_type else 'unknown'}")
        lines.append(f"   Phase: {episode.arc_phase.value if episode.arc_phase else 'unknown'}")
        lines.append(f"   Outcome: {episode.resolution or 'unknown'}")
        lines.append(f"   Consequences: {', '.join(episode.consequences[:2]) if episode.consequences else 'unknown'}")
        lines.append("")
    return "\n".join(lines)
