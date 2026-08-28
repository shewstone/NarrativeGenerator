"""Orchestration for the full extraction pipeline."""

from __future__ import annotations

import re
from contextlib import suppress
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from narrative_engine.extraction.client import ExtractionPipeline, materialize_segment_spans
from narrative_engine.extraction.config import ExtractionPipelineConfig
from narrative_engine.models import (
    Actor,
    ArcPhase,
    ArcType,
    ChangePattern,
    ClassificationState,
    Episode,
    MechanismFamily,
    MechanismTag,
    ScopeKind,
    SituationConfiguration,
    SituationDomain,
    SituationScale,
    utcnow,
)
from narrative_engine.storage.repositories import (
    RepositoryFactory,
)

logger = structlog.get_logger()

_LINK_STOP_WORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "before",
    "being",
    "between",
    "during",
    "episode",
    "from",
    "into",
    "over",
    "that",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "were",
    "when",
    "where",
    "which",
    "while",
    "with",
}


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

    async def aclose(self) -> None:
        """Release the LLM client's underlying HTTP connection pool."""
        close = getattr(self.pipeline, "aclose", None)
        if close is not None:
            await close()

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

                segments = materialize_segment_spans(text, segmentation_result.get("episodes"))
                if segments and segments[0].get("segmentation_fallback"):
                    self.logger.warning(
                        "Segmentation spans could not be resolved; extracting chunk once",
                        chunk_id=source_chunk_id,
                    )
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
                total_pairs = len(episodes) * (len(episodes) - 1) // 2
                candidate_pairs = 0
                for index, source in enumerate(episodes):
                    for target_index, target in enumerate(episodes[index + 1 :], start=index + 1):
                        if not self._is_link_candidate(source, target, target_index - index):
                            continue
                        candidate_pairs += 1
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
                self.logger.info(
                    "Linking prefilter complete",
                    total_pairs=total_pairs,
                    candidate_pairs=candidate_pairs,
                    skipped_pairs=total_pairs - candidate_pairs,
                )

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
            setting_description=extraction_result.get("setting", {}).get("description"),
            initiating_conditions=extraction_result.get("initiating_conditions", []),
            escalation_mechanics=extraction_result.get("escalation_mechanics", []),
            tension=extraction_result.get("tension"),
            resolution=extraction_result.get("resolution"),
            consequences=extraction_result.get("consequences", []),
            extracted_from=[source_chunk_id],
        )

        self._apply_focal_scope(episode, extraction_result.get("focal_scope"))

        # Parse dates
        setting = extraction_result.get("setting", {})
        if "start_date" in setting:
            episode = self._apply_normalized_dates(episode, setting)
        elif setting.get("time_period"):
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

    def _apply_focal_scope(self, episode: Episode, focal_scope: Any) -> None:
        """Record a source-backed focal subject and resolve known scopes.

        New parties, factions, movements, and ideas retain their raw names;
        only high-confidence exact aliases become canonical registry ids.
        """
        if not isinstance(focal_scope, dict):
            return

        name = focal_scope.get("name")
        if isinstance(name, str) and name.strip():
            episode.scope_name = name.strip()
        parent_name = focal_scope.get("parent_name")
        if isinstance(parent_name, str) and parent_name.strip():
            episode.parent_scope_name = parent_name.strip()
        evidence = focal_scope.get("evidence_quote")
        if isinstance(evidence, str) and evidence.strip():
            episode.scope_evidence = evidence.strip()
        boundary_note = focal_scope.get("boundary_note")
        if isinstance(boundary_note, str) and boundary_note.strip():
            episode.scope_notes = boundary_note.strip()

        kind = focal_scope.get("kind")
        if isinstance(kind, str):
            with suppress(ValueError):
                episode.scope_kind = ScopeKind(kind)

        confidence = self._bounded_confidence(focal_scope.get("confidence"))
        episode.scope_confidence = confidence
        if episode.scope_name and confidence is not None and confidence >= self.config.scope_confidence_floor:
            from narrative_engine.scopes import get_registry, resolve_scope

            episode.scope_id = resolve_scope(episode.scope_name)
            canonical = get_registry().get(episode.scope_id) if episode.scope_id else None
            if canonical is not None:
                episode.scope_kind = canonical.kind
                if canonical.parent_scope_id and not episode.parent_scope_name:
                    parent = get_registry().get(canonical.parent_scope_id)
                    episode.parent_scope_name = parent.name if parent else None

    @staticmethod
    def _bounded_confidence(value: Any, default: Optional[float] = None) -> Optional[float]:
        """Parse a finite 0..1 score without trusting arbitrary LLM JSON."""
        if isinstance(value, bool):
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if 0.0 <= parsed <= 1.0 else default

    def _apply_normalized_dates(self, episode: Episode, setting: Dict[str, Any]) -> Episode:
        """Apply LLM-normalized partial ISO dates after deterministic validation."""
        import calendar
        import re
        from datetime import datetime, timezone

        def parse_partial(value: Optional[str], *, end_bound: bool) -> Optional[datetime]:
            if value is None:
                return None
            match = re.fullmatch(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", value)
            if not match:
                raise ValueError(f"invalid normalized date {value!r}")
            year = int(match.group(1))
            month = int(match.group(2) or (12 if end_bound else 1))
            day = int(match.group(3) or (calendar.monthrange(year, month)[1] if end_bound else 1))
            return datetime(year, month, day, tzinfo=timezone.utc)

        try:
            episode.start_date = parse_partial(setting.get("start_date"), end_bound=False)
            episode.end_date = parse_partial(setting.get("end_date"), end_bound=True)
            episode.date_precision = setting.get("date_precision") or "unknown"
        except (TypeError, ValueError) as e:
            self.logger.warning(
                "Rejected invalid LLM-normalized date",
                label=setting.get("time_period_label"),
                error=str(e),
            )
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
        """Classify a neutral situation pattern and optional legacy arc."""
        # First pass classification
        classification = await self.pipeline.classify(
            episode_summary=episode.summary,
            full_text=f"{episode.title}\n{episode.summary}",
        )

        # Primary, scale-neutral reading.
        pattern_str = classification.get("change_pattern")
        if pattern_str:
            with suppress(ValueError):
                episode.change_pattern = ChangePattern(pattern_str)
        episode.pattern_confidence = self._bounded_confidence(classification.get("pattern_confidence"), 0.0) or 0.0
        episode.pattern_rationale = classification.get("pattern_rationale")

        scale_str = classification.get("situation_scale")
        if scale_str:
            with suppress(ValueError):
                episode.situation_scale = SituationScale(scale_str)

        episode.domains = []
        for domain in classification.get("domains", []):
            with suppress(ValueError):
                parsed_domain = SituationDomain(domain)
                if parsed_domain not in episode.domains:
                    episode.domains.append(parsed_domain)

        configuration = classification.get("configuration")
        if isinstance(configuration, dict):
            scores = {
                key: value
                for key, value in configuration.items()
                if key in SituationConfiguration.model_fields
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and -1.0 <= float(value) <= 1.0
            }
            episode.configuration = SituationConfiguration(**scores)

        episode.mechanism_families = []
        for family in classification.get("mechanism_families", []):
            with suppress(ValueError):
                parsed_family = MechanismFamily(family)
                if parsed_family not in episode.mechanism_families:
                    episode.mechanism_families.append(parsed_family)

        # Optional legacy arc reading.
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

        episode.phase_confidence = self._bounded_confidence(classification.get("phase_confidence"), 0.0) or 0.0
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
        episode.mechanism_tags = []
        for tag in classification.get("mechanism_tags", []):
            with suppress(ValueError):
                parsed_tag = MechanismTag(tag)
                if parsed_tag not in episode.mechanism_tags:
                    episode.mechanism_tags.append(parsed_tag)

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
        pattern_clears_floor = episode.change_pattern is not None and episode.pattern_confidence >= floor
        if not pattern_clears_floor:
            episode.change_pattern = None

        arc_clears_floor = episode.arc_type is not None and episode.phase_confidence >= floor
        if not arc_clears_floor:
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

        if pattern_clears_floor or arc_clears_floor:
            episode.classification_state = ClassificationState.CLASSIFIED
        else:
            episode.classification_state = ClassificationState.UNCLASSIFIED

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
        import re

        from dateutil import parser as date_parser

        try:
            # Try to parse as range (e.g., "1921-1923" or "1921 to 1923")
            range_match = re.fullmatch(
                r"\s*(\d{3,4})\s*[-–—]\s*(\d{3,4})\s*",
                time_period,
            ) or re.fullmatch(
                r"\s*(.+?)\s+(?:to|through)\s+(.+?)\s*",
                time_period,
                flags=re.IGNORECASE,
            )
            if range_match:
                start = date_parser.parse(range_match.group(1), fuzzy=True)
                end = date_parser.parse(range_match.group(2), fuzzy=True)
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

    def _is_link_candidate(self, source: Episode, target: Episode, distance: int) -> bool:
        """Cheap, conservative gate before pairwise LLM relationship checks."""
        if distance <= self.config.linking_neighbor_window:
            return True

        source_actors = {self._normalize_link_value(actor.name) for actor in source.actors}
        target_actors = {self._normalize_link_value(actor.name) for actor in target.actors}
        if (source_actors - {""}) & (target_actors - {""}):
            return True

        source_location = self._normalize_link_value(source.location)
        target_location = self._normalize_link_value(target.location)
        if source_location and source_location == target_location:
            return True

        year_gap = self._episode_year_gap(source, target)
        if year_gap is not None and year_gap <= self.config.linking_max_year_gap:
            return True

        source_terms = self._link_terms(self._episode_link_text(source))
        target_terms = self._link_terms(self._episode_link_text(target))
        if not source_terms or not target_terms:
            return False
        shared_terms = source_terms & target_terms
        overlap = len(shared_terms) / min(len(source_terms), len(target_terms))
        return len(shared_terms) >= 2 and overlap >= self.config.linking_min_lexical_overlap

    @staticmethod
    def _normalize_link_value(value: Optional[str]) -> str:
        if not value:
            return ""
        return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))

    @staticmethod
    def _link_terms(value: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9]+", value.casefold())
            if len(term) >= 4 and term not in _LINK_STOP_WORDS
        }

    @staticmethod
    def _episode_link_text(episode: Episode) -> str:
        """Collect deterministic evidence fields used only by the cheap gate."""
        return " ".join(
            value
            for value in (
                episode.title,
                episode.summary,
                episode.tension,
                episode.resolution,
                *episode.initiating_conditions,
                *episode.escalation_mechanics,
                *episode.consequences,
            )
            if value
        )

    @staticmethod
    def _episode_year_gap(source: Episode, target: Episode) -> Optional[float]:
        """Return zero for overlapping intervals, otherwise years between them."""
        if source.start_date is None or target.start_date is None:
            return None
        source_start = source.start_date.date()
        source_end = (source.end_date or source.start_date).date()
        target_start = target.start_date.date()
        target_end = (target.end_date or target.start_date).date()
        if source_end < target_start:
            days = (target_start - source_end).days
        elif target_end < source_start:
            days = (source_start - target_end).days
        else:
            return 0.0
        return days / 365.2425

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
