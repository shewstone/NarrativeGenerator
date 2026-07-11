"""Orchestration for the full extraction pipeline."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from narrative_engine.extraction.client import ExtractionPipeline
from narrative_engine.extraction.config import ExtractionPipelineConfig
from narrative_engine.models import (
    Actor,
    ArcPhase,
    ArcType,
    ClassificationState,
    Episode,
    MechanismTag,
    utcnow,
)
from narrative_engine.storage.repositories import (
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
        self.created_at = utcnow()


class ExtractionOrchestrator:
    """Orchestrates the full extraction pipeline from raw text to database."""

    def __init__(
        self,
        pipeline: Optional[ExtractionPipeline] = None,
        config: Optional[ExtractionPipelineConfig] = None,
    ) -> None:
        self.config = config or ExtractionPipelineConfig.from_env()
        self.pipeline = pipeline or ExtractionPipeline(config=self.config)
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

            if self.config.enable_linking and len(episodes) > 1:
                self.logger.info("Stage 4: Linking")
                for index, source in enumerate(episodes):
                    for target in episodes[index + 1 :]:
                        try:
                            await self._link_episode_pair(source, target, session)
                        except Exception as e:
                            self.logger.error(
                                "Linking failed",
                                source=source.title,
                                target=target.title,
                                error=str(e),
                            )
                            errors.append(f"Linking {source.title} -> {target.title}: {str(e)}")

        except Exception as e:
            self.logger.error("Pipeline failed", error=str(e), chunk_id=source_chunk_id)
            errors.append(f"Pipeline: {str(e)}")

        # Round up: a pipeline that ran reports at least 1ms, never a
        # truncated-to-zero artifact for sub-millisecond (e.g. mocked) runs.
        processing_time_ms = max(1, int((time.time() - start_time) * 1000))

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
            episode = await self._parse_dates(episode, setting["time_period"], setting.get("date_precision", "year"))

        # Parse actors. canonical_role passes the tau_role fit floor or stays
        # None (no forced choice — T2); unknown vocabulary values are treated
        # as unresolved rather than invented roles entering the render.
        actors_data = extraction_result.get("actors", [])
        episode.actors = [
            Actor(
                name=a.get("name", "Unknown"),
                role=a.get("role", "unknown"),
                canonical_role=self._resolve_canonical_role(a),
                role_fit_confidence=a.get("role_fit_confidence"),
                attributes=a.get("attributes", {}),
            )
            for a in actors_data
        ]

        return episode

    def _resolve_canonical_role(self, actor_data: Dict[str, Any]) -> Optional[str]:
        """Apply the tau_role floor to an extracted canonical_role claim."""
        from narrative_engine.extraction.roles import is_known_role

        candidate = actor_data.get("canonical_role")
        if not candidate or not is_known_role(candidate):
            return None
        confidence = actor_data.get("role_fit_confidence")
        if confidence is None or confidence < self.config.role_fit_floor:
            return None
        return candidate

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
            with suppress(ValueError):
                episode.secondary_arcs.append(
                    (
                        ArcType(sec.get("type", "unknown")),
                        ArcPhase(sec.get("phase", "unknown")),
                        sec.get("confidence", 0.5),
                    )
                )

        # Handle mechanism tags (design doc Sec 3.8): unrecognized tags are
        # skipped, not fatal -- the LLM occasionally drifts from the
        # vocabulary given in the prompt.
        for tag in classification.get("mechanism_tags", []):
            with suppress(ValueError):
                episode.mechanism_tags.append(MechanismTag(tag))

        # tau_class floor (design doc Sec 6.2 stage 4): classification is
        # NOT a forced choice. If the best canonical arc doesn't clear the
        # floor -- or the LLM produced no usable label at all -- the episode
        # carries no arc assignment rather than its least-bad label.
        # Secondary arcs are dropped too: they rank below the primary, so
        # they cannot clear a floor the primary failed. Confidence and
        # rationale are kept for audit; unclassified episodes are excluded
        # from the arc-conditioned analog base (repositories.
        # search_by_embedding) and feed the discovery trigger (Sec 3.4)
        # when that lands.
        floor = self.config.classification_confidence_floor
        if episode.arc_type is None or episode.phase_confidence < floor:
            if episode.arc_type is not None:
                self.logger.info(
                    "Episode failed tau_class floor; marking unclassified",
                    arc_type=episode.arc_type.value,
                    confidence=episode.phase_confidence,
                    floor=floor,
                )
            episode.arc_type = None
            episode.arc_phase = None
            episode.secondary_arcs = []
            episode.classification_state = ClassificationState.UNCLASSIFIED
        else:
            episode.classification_state = ClassificationState.CLASSIFIED

        # TODO: Second-pass classification with nearest neighbors
        # Requires vector search for similar episodes. NOTE (Sec 6.2 stage
        # 4): when this lands, unclassified episodes must be excluded from
        # the neighbor pool so low-confidence labels never propagate.

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

    async def _link_episode_pair(
        self,
        source: Episode,
        target: Episode,
        session: AsyncSession,
    ) -> None:
        """Persist a supported identity/causal relationship for one pair."""
        from narrative_engine.storage.orm_models import EpisodeLinkORM

        result = await self.pipeline.link(source.model_dump(mode="json"), target.model_dump(mode="json"))
        relationship = result.get("relationship")
        if relationship not in {"same_event", "causes", "caused_by"}:
            return
        if float(result.get("confidence", 0.0)) < 0.5:
            return

        evidence = result.get("evidence_quote")
        if relationship in {"causes", "caused_by"} and not evidence:
            raise ValueError("Causal link requires a verbatim evidence quote")
        if evidence:
            supplied_text = "\n".join((source.title, source.summary, target.title, target.summary))
            if evidence not in supplied_text:
                raise ValueError("Evidence quote is not present in the supplied episode text")

        source_id, target_id = source.id, target.id
        if relationship == "caused_by":
            source_id, target_id = target_id, source_id
        edge_kind = "same_event_as" if relationship == "same_event" else "causes"
        session.add(
            EpisodeLinkORM(
                source_episode_id=source_id,
                target_episode_id=target_id,
                edge_kind=edge_kind,
                link_status="attested",
                evidence=evidence or result.get("reasoning"),
                review_status="pending",
            )
        )
        await session.flush()

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
