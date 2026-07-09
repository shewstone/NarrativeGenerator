"""Baseline forecasters for the evaluation harness.

Design doc Sec 6.6: "Baselines: persistence ('things continue'), bare LLM
with no retrieval system, simple reference-class forecasting. If the
machinery can't beat the bare LLM, the structure isn't paying rent."

Until these run alongside the thesis pipeline in every backtest, a good
Brier score is uninterpretable -- it may be measuring the model's memory
of history rather than anything the arc machinery adds.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from narrative_engine.models import Episode

logger = structlog.get_logger()


@dataclass
class BaselinePrediction:
    """A baseline's answer, shaped for the same scoring path as a Thesis."""

    baseline_name: str
    predicted_continuation: str
    probability: float
    rationale: str


class PersistenceBaseline:
    """'Things continue as they are' -- the floor every forecast must beat."""

    name = "persistence"

    def predict(self, query_episode: Episode) -> BaselinePrediction:
        tension = query_episode.tension or "current conditions"
        return BaselinePrediction(
            baseline_name=self.name,
            predicted_continuation=(
                f"The situation persists: {tension} continues without "
                "structural resolution over the forecast horizon"
            ),
            probability=0.5,
            rationale="Persistence baseline: no change predicted, by construction",
        )


class BareLLMBaseline:
    """The model alone, no retrieval, no arcs, no analogs.

    This is the baseline that decides whether the whole apparatus pays
    rent (Sec 6.6): the LLM already knows a great deal of history, so the
    system must add value over simply asking it.
    """

    name = "bare_llm"

    PROMPT = """You are forecasting the continuation of a present-day situation.
Do not assume access to any retrieval system; answer from general knowledge.

Situation: {title}
Summary: {summary}
Tension: {tension}

Respond with JSON: {{"continuation": "<one-sentence most likely continuation>",
"probability": <0.0-1.0>, "rationale": "<one sentence>"}}"""

    def __init__(self, llm_client=None) -> None:
        # Any object with `async complete(prompt) -> str` works; defaults to
        # the extraction pipeline's client so both use the same model config.
        self._client = llm_client

    async def predict(self, query_episode: Episode) -> BaselinePrediction:
        import json

        client = self._client
        if client is None:
            from narrative_engine.extraction.client import ExtractionPipeline

            # Reuse the extraction pipeline's provider selection/config.
            client = ExtractionPipeline().client

        prompt = self.PROMPT.format(
            title=query_episode.title,
            summary=query_episode.summary,
            tension=query_episode.tension or "unknown",
        )

        raw = await client.complete(prompt)

        try:
            parsed = json.loads(raw)
            return BaselinePrediction(
                baseline_name=self.name,
                predicted_continuation=str(parsed.get("continuation", raw[:200])),
                probability=float(parsed.get("probability", 0.5)),
                rationale=str(parsed.get("rationale", "")),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Bare-LLM baseline returned non-JSON; using raw text")
            return BaselinePrediction(
                baseline_name=self.name,
                predicted_continuation=raw[:200] if raw else "no prediction",
                probability=0.5,
                rationale="unparseable model output",
            )
