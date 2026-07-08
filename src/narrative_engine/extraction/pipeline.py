"""Orchestration for the full extraction pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from narrative_engine.extraction.client import ExtractionPipeline, LLMClient
from narrative_engine.extraction.config import ExtractionPipelineConfig
from narrative_engine.models import Episode, Actor, ArcType, ArcPhase, SourcePassage
from narrative_engine.storage.repositories import (
    EpisodeRepository,
    RepositoryFactory,
)

logger = structlog.get_logger()


class PipelineResult:
    """Result of a full extraction pipeline run."""

    def __init__(
        self,
        source_chunk_id: str,
        episodes: List[Episode],
        processing_time_ms: int,
        errors: List[str],
    ) -> None:
        self.source_chunk_id = source_chunk_id
        self.episodes = episodes
        self.processing_time_ms = processing_time_ms
        self.errors = errors
        self.created_at = datetime.utcnow()


class ExtractionOrchestrator:
    """Orchestrates the full extraction pipeline from raw text to database."""

    def __init__(
        self,
        pipeline: Optional[ExtractionPipeline] = None,
        config: Optional[ExtractionPipelineConfig] = None,
    ) -> None:
        self.pipeline = pipeline or ExtractionPipeline(config=config)
        self.config = config or ExtractionPipelineConfig.from_env()
        self.logger = structlog.get_logger()

    async def process_text(
        self,
        text: str,
        source_chunk_id: str,
        session: AsyncSession,
    ) -> PipelineResult:
        """Process raw text through full pipeline and store results.

        Stage 1: Segmentation → Stage 2: Extraction → Stage 3: Classification
        """
        import time

        start_time = time.time()

        episodes: List[Episode] = []
        errors: List[str] = []

        try:
            # Stage 1: Segmentation
            if self.config.enable_segmentation:
                self.logger.info("Stage 1: Segmentation", chunk_id=source_chunk_id)
                segmentation_result = await self.pipeline.segment(text)

                segments = segmentation_result.get("episodes", [])
                self.logger.info(f"Found {len(segments)} segments")
            else:
                # If segmentation disabled, treat whole text as one segment
                segments = [{"number": 1, "summary": text[:200], "text": text}]

            # Stage 2: Extraction
            for segment in segments:
                try:
                    episode = await self._extract_segment(
                        segment,
                        text,  # Full context
                        source_chunk_id,
                    )

                    if episode:
                        episodes.append(episode)

                except Exception as e:
                    self.logger.error(
                        "Extraction failed for segment",
                        segment=segment.get("number"),
                        error=str(e),
                    )
                    errors.append(f"Segment {segment.get('number')}: {str(e)}")

            # Stage 3: Classification
            if self.config.enable_classification:
                self.logger.info("Stage 3: Classification")

                for episode in episodes:
                    try:
                        await self._classify_episode(episode)
                    except Exception as e:
                        self.logger.error(
                            "Classification failed",
                            episode=episode.title,
                            error=str(e),
                        )
                        errors.append(f"Classification for {episode.title}: {str(e)}")

            # Store in database
            await self._store_episodes(episodes, session)

        except Exception as e:
            self.logger.error("Pipeline failed", error=str(e), chunk_id=source_chunk_id)
            errors.append(f"Pipeline: {str(e)}")

        processing_time_ms = int((time.time() - start_time) * 1000)

        return PipelineResult(
            source_chunk_id=source_chunk_id,
            episodes=episodes,
            processing_time_ms=processing_time_ms,
            errors=errors,
        )

    async def _extract_segment(
        self,
        segment: Dict[str, Any],
        full_text: str,
        source_chunk_id: str,
    ) -> Optional[Episode]:
        """Extract structured data from a single segment."""
        if not self.config.enable_extraction:
            return None

        segment_text = segment.get("text", full_text)
        segment_summary = segment.get("summary", segment_text[:200])

        # Call LLM for extraction
        extraction_result = await self.pipeline.extract(
            segment_text=segment_text,
            segment_summary=segment_summary,
        )

        # Build Episode from extraction result
        episode = Episode(
            title=extraction_result.get("title", "Untitled"),
            summary=extraction_result.get("summary", ""),
            location=extraction_result.get("setting", {}).get("location"),
            initiating_conditions=extraction_result.get("initiating_conditions", []),
            escalation_mechanics=extraction_result.get("escalation_mechanics", []),
            tension=extraction_result.get("tension"),
            resolution=extraction_result.get("resolution"),
            consequences=extraction_result.get("consequences", []),
            extracted_from=[source_chunk_id],
        )

        # Parse dates
        setting = extraction_result.get("setting", {})
        if setting.get("time_period"):
            episode = await self._parse_dates(
                episode, setting["time_period"], setting.get("date_precision", "year")
            )

        # Parse actors
        actors_data = extraction_result.get("actors", [])
        episode.actors = [
            Actor(
                name=a.get("name", "Unknown"),
                role=a.get("role", "unknown"),
                attributes=a.get("attributes", {}),
            )
            for a in actors_data
        ]

        return episode

    async def _classify_episode(self, episode: Episode) -> None:
        """Classify arc type and phase for an episode."""
        # First pass classification
        classification = await self.pipeline.classify(
            episode_summary=episode.summary,
            full_text=f"{episode.title}\n{episode.summary}",
        )

        # Update episode with classification
        arc_type_str = classification.get("arc_type")
        arc_phase_str = classification.get("arc_phase")

        if arc_type_str:
            try:
                episode.arc_type = ArcType(arc_type_str)
            except ValueError:
                self.logger.warning(f"Unknown arc type: {arc_type_str}")

        if arc_phase_str:
            try:
                episode.arc_phase = ArcPhase(arc_phase_str)
            except ValueError:
                self.logger.warning(f"Unknown arc phase: {arc_phase_str}")

        episode.phase_confidence = classification.get("phase_confidence", 0.0)
        episode.arc_rationale = classification.get("rationale")

        # Handle secondary arcs
        secondary = classification.get("secondary_arcs", [])
        for sec in secondary:
            try:
                episode.secondary_arcs.append(
                    (
                        ArcType(sec.get("type", "unknown")),
                        ArcPhase(sec.get("phase", "unknown")),
                        sec.get("confidence", 0.5),
                    )
                )
            except ValueError:
                pass

        # TODO: Second-pass classification with nearest neighbors
        # Requires vector search for similar episodes

    async def _parse_dates(
        self,
        episode: Episode,
        time_period: str,
        precision: str,
    ) -> Episode:
        """Parse date strings into datetime objects."""
        # Simple parsing—could be enhanced with dateparser
        from dateutil import parser as date_parser

        try:
            # Try to parse as range (e.g., "1921-1923" or "1921 to 1923")
            if "-" in time_period or "to" in time_period:
                parts = time_period.replace("to", "-").split("-")
                if len(parts) >= 2:
                    start = date_parser.parse(parts[0].strip(), fuzzy=True)
                    end = date_parser.parse(parts[1].strip(), fuzzy=True)
                    episode.start_date = start
                    episode.end_date = end
                    episode.date_precision = "range"
            else:
                # Single date
                date = date_parser.parse(time_period, fuzzy=True)
                episode.start_date = date
                episode.date_precision = precision

        except Exception as e:
            self.logger.warning(f"Failed to parse date: {time_period}", error=str(e))

        return episode

    async def _store_episodes(
        self,
        episodes: List[Episode],
        session: AsyncSession,
    ) -> None:
        """Store extracted episodes in database."""
        factory = RepositoryFactory(session)

        for episode in episodes:
            try:
                created = await factory.episodes.create(episode)
                self.logger.info(f"Stored episode: {created.title}")
            except Exception as e:
                self.logger.error(f"Failed to store episode: {episode.title}", error=str(e))
                raise

    async def process_batch(
        self,
        chunks: List[Dict[str, str]],  # [{"id": "...", "text": "..."}, ...]
        session: AsyncSession,
    ) -> List[PipelineResult]:
        """Process multiple text chunks."""
        results = []

        for chunk in chunks:
            result = await self.process_text(
                text=chunk["text"],
                source_chunk_id=chunk["id"],
                session=session,
            )
            results.append(result)

        return results
