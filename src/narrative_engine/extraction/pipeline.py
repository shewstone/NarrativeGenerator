"""Orchestration for the full extraction pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from contextlib import suppress
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from narrative_engine.extraction.client import (
    ExtractionPipeline,
    LLMError,
    materialize_segment_spans,
)
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
    SourcePassage,
    utcnow,
)
from narrative_engine.storage.repositories import RepositoryFactory

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

# Some otherwise-valid classifiers occasionally echo a semantic phase from an
# arc-type name instead of the legacy vocabulary stated in the prompt. Keep
# the compatibility layer narrow and evidence-backed rather than accepting an
# arbitrary new phase that downstream composition cannot order.
_ARC_PHASE_ALIASES = {
    "renewal": ArcPhase.RESOLUTION,
}
_CANONICAL_SCOPE_SELECTION_FLOOR = 0.8


def _parse_arc_phase(value: str) -> ArcPhase:
    return _ARC_PHASE_ALIASES[value] if value in _ARC_PHASE_ALIASES else ArcPhase(value)


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
        skip_segmentation: bool = False,
    ) -> PipelineResult:
        """Process raw text through full pipeline and store results.

        Stage 1: Segmentation → Stage 2: Extraction → Stage 3: Classification
        """
        start_time = time.time()

        episodes: List[Episode] = []
        errors: List[str] = []

        try:
            # Stage 1: Segmentation
            if self.config.enable_segmentation and not skip_segmentation:
                self.logger.info("Stage 1: Segmentation", chunk_id=source_chunk_id)
                try:
                    stage_started = time.perf_counter()
                    segmentation_result = await self.pipeline.segment(text)
                    raw_segments = segmentation_result.get("episodes")
                    if isinstance(raw_segments, list) and all(
                        isinstance(segment, dict) and isinstance(segment.get("text"), str)
                        for segment in raw_segments
                    ):
                        # ExtractionPipeline.segment already materializes raw
                        # model boundaries. Custom pipeline implementations
                        # may return raw offsets, handled by the fallback.
                        segments = [dict(segment) for segment in raw_segments]
                    else:
                        segments = materialize_segment_spans(text, raw_segments)
                    response_without_text = {
                        **segmentation_result,
                        "episodes": [
                            {key: value for key, value in segment.items() if key != "text"}
                            for segment in segments
                        ],
                    }
                    self._record_stage(
                        session,
                        source_chunk_id=source_chunk_id,
                        pipeline_stage="segmentation",
                        prompt_version="1.3.0",
                        model_used=self.config.segmentation_model,
                        input_data=self._text_audit_input(text),
                        output_data={
                            "response": response_without_text,
                            "materialized_segments": [
                                {
                                    "number": segment.get("number"),
                                    "start_char": segment.get("start_char"),
                                    "end_char": segment.get("end_char"),
                                    "fallback": segment.get("segmentation_fallback"),
                                }
                                for segment in segments
                            ],
                        },
                        processing_time_ms=self._elapsed_ms(stage_started),
                    )
                except LLMError as exc:
                    # Segmentation improves granularity but is not required
                    # for correctness: SmartChunker has already bounded the
                    # source. A truncated/invalid segment response should not
                    # discard an otherwise extractable book checkpoint.
                    self.logger.warning(
                        "Segmentation unavailable; extracting chunk once",
                        chunk_id=source_chunk_id,
                        error=str(exc),
                    )
                    segments = [
                        {
                            "number": 1,
                            "summary": text[:200],
                            "text": text,
                            "segmentation_fallback": "segmentation_llm_error",
                        }
                    ]
                    self._record_stage(
                        session,
                        source_chunk_id=source_chunk_id,
                        pipeline_stage="segmentation",
                        prompt_version="1.3.0",
                        model_used=self.config.segmentation_model,
                        input_data=self._text_audit_input(text),
                        output_data={"fallback": "segmentation_llm_error"},
                        error_message=str(exc),
                    )

                if segments and segments[0].get("segmentation_fallback") == "unresolved_source_spans":
                    self.logger.warning(
                        "Segmentation spans could not be resolved; extracting chunk once",
                        chunk_id=source_chunk_id,
                    )
                self.logger.info(f"Found {len(segments)} segments")
            else:
                # If segmentation disabled, treat whole text as one segment
                segments = [{"number": 1, "summary": text[:200], "text": text}]

            # Stage 2: Extraction. Supply only bounded neighboring summaries,
            # not the full chunk again for every segment. This gives focal
            # scope continuity without multiplying input-token and audit size.
            narrative_context = "\n".join(
                f"{index}. {str(item.get('summary') or '')[:240]}"
                for index, item in enumerate(segments, start=1)
            )
            extraction_slots = asyncio.Semaphore(self.config.stage_concurrency)

            async def extract_one(segment: Dict[str, Any]) -> tuple[Optional[Episode], Optional[str]]:
                async with extraction_slots:
                    try:
                        episode = await self._extract_segment(
                            segment,
                            text,  # Full context
                            source_chunk_id,
                            session=session,
                            narrative_context=narrative_context,
                        )
                        return episode, None
                    except Exception as exc:
                        self.logger.error(
                            "Extraction failed for segment",
                            segment=segment.get("number"),
                            error=str(exc),
                        )
                        return None, f"Segment {segment.get('number')}: {str(exc)}"

            extraction_results = await asyncio.gather(
                *(extract_one(segment) for segment in segments)
            )
            for episode, error in extraction_results:
                if episode is not None:
                    episodes.append(episode)
                if error is not None:
                    errors.append(error)

            # Stage 3: Classification
            if self.config.enable_classification:
                self.logger.info("Stage 3: Classification")

                classification_slots = asyncio.Semaphore(self.config.stage_concurrency)

                async def classify_one(episode: Episode) -> Optional[str]:
                    async with classification_slots:
                        try:
                            source_text = (
                                episode.source_passages[0].text
                                if episode.source_passages
                                else episode.summary
                            )
                            await self._classify_episode(
                                episode,
                                source_text=source_text,
                                source_chunk_id=source_chunk_id,
                                session=session,
                            )
                            return None
                        except Exception as exc:
                            self.logger.error(
                                "Classification failed",
                                episode=episode.title,
                                error=str(exc),
                            )
                            return f"Classification for {episode.title}: {str(exc)}"

                classification_errors = await asyncio.gather(
                    *(classify_one(episode) for episode in episodes)
                )
                errors.extend(error for error in classification_errors if error is not None)

            # Store in database
            await self._store_episodes(episodes, session)

            if (
                self.config.enable_linking
                and self.config.enable_chunk_linking
                and len(episodes) > 1
            ):
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
                            self.logger.warning(
                                "Optional chunk link rejected",
                                source=source.title,
                                target=target.title,
                                error=str(e),
                            )
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
        session: Optional[AsyncSession] = None,
        narrative_context: Optional[str] = None,
    ) -> Optional[Episode]:
        """Extract structured data from a single segment."""
        if not self.config.enable_extraction:
            return None

        segment_text = segment.get("text", full_text)
        segment_summary = segment.get("summary", segment_text[:200])

        # Call LLM for extraction
        stage_started = time.perf_counter()
        try:
            extraction_result = await self.pipeline.extract(
                segment_text=segment_text,
                segment_summary=segment_summary,
                narrative_context=narrative_context,
            )
        except Exception as exc:
            self._record_stage(
                session,
                source_chunk_id=source_chunk_id,
                pipeline_stage="extraction",
                prompt_version="2.1.0",
                model_used=self.config.extraction_model,
                input_data={
                    **self._text_audit_input(segment_text),
                    "segment_number": segment.get("number"),
                    "segment_summary": segment_summary,
                },
                output_data={},
                processing_time_ms=self._elapsed_ms(stage_started),
                error_message=str(exc),
            )
            raise
        if not isinstance(extraction_result, dict):
            raise TypeError("Extraction response must be a JSON object")

        self._record_stage(
            session,
            source_chunk_id=source_chunk_id,
            pipeline_stage="extraction",
            prompt_version="2.1.0",
            model_used=self.config.extraction_model,
            input_data={
                **self._text_audit_input(segment_text),
                "segment_number": segment.get("number"),
                "segment_summary": segment_summary,
                "start_char": segment.get("start_char"),
                "end_char": segment.get("end_char"),
            },
            output_data=extraction_result,
            processing_time_ms=self._elapsed_ms(stage_started),
        )

        setting_value = extraction_result.get("setting")
        setting = setting_value if isinstance(setting_value, dict) else {}

        # Build Episode from extraction result
        episode = Episode(
            title=extraction_result.get("title", "Untitled"),
            summary=extraction_result.get("summary", ""),
            location=setting.get("location"),
            setting_description=setting.get("description"),
            initiating_conditions=self._string_list(extraction_result.get("initiating_conditions")),
            escalation_mechanics=self._string_list(extraction_result.get("escalation_mechanics")),
            tension=extraction_result.get("tension"),
            resolution=extraction_result.get("resolution"),
            consequences=self._string_list(extraction_result.get("consequences")),
            extracted_from=[source_chunk_id],
            source_passages=[
                SourcePassage(
                    work_id=self._work_id_from_chunk_id(source_chunk_id),
                    passage_id=(
                        f"{source_chunk_id}:"
                        f"{segment.get('start_char', 0)}-{segment.get('end_char', len(segment_text))}"
                    ),
                    text=segment_text,
                )
            ],
        )

        self._apply_focal_scope(episode, extraction_result.get("focal_scope"))
        await self._canonicalize_focal_scope(
            episode,
            source_chunk_id=source_chunk_id,
            session=session,
        )

        # Parse dates
        if "start_date" in setting:
            episode = self._apply_normalized_dates(episode, setting)
        elif setting.get("time_period"):
            episode = await self._parse_dates(episode, setting["time_period"], setting.get("date_precision", "year"))

        # Parse actors. canonical_role passes the tau_role fit floor or stays
        # None (no forced choice — T2); unknown vocabulary values are treated
        # as unresolved rather than invented roles entering the render.
        actors_data = [actor for actor in self._as_list(extraction_result.get("actors")) if isinstance(actor, dict)]
        episode.actors = [
            Actor(
                name=a.get("name", "Unknown"),
                role=a.get("role", "unknown"),
                canonical_role=self._resolve_canonical_role(a),
                role_fit_confidence=self._bounded_confidence(a.get("role_fit_confidence")),
                attributes=a.get("attributes") if isinstance(a.get("attributes"), dict) else {},
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

    async def _canonicalize_focal_scope(
        self,
        episode: Episode,
        *,
        source_chunk_id: str,
        session: Optional[AsyncSession],
    ) -> None:
        """Resolve a high-confidence raw name from a bounded candidate set."""
        if (
            episode.scope_id
            or not episode.scope_name
            or episode.scope_confidence is None
            or episode.scope_confidence < self.config.scope_confidence_floor
        ):
            return

        from narrative_engine.scopes import get_registry, suggest_scopes

        raw_kind = episode.scope_kind.value if episode.scope_kind else None
        suggestions = suggest_scopes(
            episode.scope_name,
            kind=raw_kind,
            parent_name=episode.parent_scope_name,
            limit=5,
        )
        # Avoid spending a model call when lexical retrieval found no
        # meaningful identity candidate. Such names remain visible residue.
        if not suggestions or suggestions[0].score < 0.58:
            return

        registry = get_registry()
        candidate_payload = []
        for suggestion in suggestions:
            parent = (
                registry.get(suggestion.scope.parent_scope_id)
                if suggestion.scope.parent_scope_id
                else None
            )
            candidate_payload.append(
                {
                    "id": suggestion.scope.id,
                    "name": suggestion.scope.name,
                    "kind": suggestion.scope.kind.value,
                    "parent": parent.name if parent else None,
                    "retrieval_score": suggestion.score,
                }
            )

        stage_started = time.perf_counter()
        try:
            result = await self.pipeline.canonicalize_scope(
                raw_name=episode.scope_name,
                raw_kind=raw_kind,
                parent_name=episode.parent_scope_name,
                evidence_quote=episode.scope_evidence,
                candidates=candidate_payload,
            )
        except Exception as exc:
            self._record_stage(
                session,
                source_chunk_id=source_chunk_id,
                pipeline_stage="scope_canonicalization",
                prompt_version="1.0.0",
                model_used=self.config.scope_model,
                input_data={"raw_name": episode.scope_name, "candidates": candidate_payload},
                output_data={},
                processing_time_ms=self._elapsed_ms(stage_started),
                error_message=str(exc),
            )
            self.logger.warning(
                "Scope canonicalization unavailable; retaining raw scope",
                raw_scope=episode.scope_name,
                error=str(exc),
            )
            return
        if not isinstance(result, dict):
            return

        confidence = self._bounded_confidence(result.get("confidence"), 0.0) or 0.0
        selected_id = result.get("scope_id")
        allowed_ids = {candidate["id"] for candidate in candidate_payload}
        accepted = (
            isinstance(selected_id, str)
            and selected_id in allowed_ids
            and confidence >= _CANONICAL_SCOPE_SELECTION_FLOOR
        )
        self._record_stage(
            session,
            source_chunk_id=source_chunk_id,
            pipeline_stage="scope_canonicalization",
            prompt_version="1.0.0",
            model_used=self.config.scope_model,
            input_data={"raw_name": episode.scope_name, "candidates": candidate_payload},
            output_data={**result, "accepted": accepted},
            confidence=confidence,
            processing_time_ms=self._elapsed_ms(stage_started),
        )
        if not accepted:
            return

        canonical = registry.get(selected_id)
        if canonical is None:
            return
        episode.scope_id = canonical.id
        episode.scope_kind = canonical.kind
        if canonical.parent_scope_id and not episode.parent_scope_name:
            parent = registry.get(canonical.parent_scope_id)
            episode.parent_scope_name = parent.name if parent else None
        reason = result.get("reason")
        if isinstance(reason, str) and reason.strip():
            note = f"Registry match: {reason.strip()}"
            episode.scope_notes = f"{episode.scope_notes}; {note}" if episode.scope_notes else note

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

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        """Normalize a drifting JSON collection without iterating strings."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @classmethod
    def _string_list(cls, value: Any) -> List[str]:
        """Accept a scalar string or a list and discard non-string debris."""
        return [item for item in cls._as_list(value) if isinstance(item, str)]

    @staticmethod
    def _work_id_from_chunk_id(source_chunk_id: str) -> str:
        """Recover a stable work id from SmartChunker's ``work_id_N`` id."""
        return re.sub(r"_\d+$", "", source_chunk_id)

    @staticmethod
    def _episode_source_chunk(episode: Episode) -> str:
        return episode.extracted_from[0] if episode.extracted_from else "unknown"

    @staticmethod
    def _text_audit_input(text: str) -> Dict[str, Any]:
        """Describe an LLM input without duplicating source text in audit JSON.

        Exact evidence is stored once in ``source_passages``. The audit row
        retains a digest and bounded preview so requests can be correlated
        without multiplying the database footprint by every pipeline stage.
        """
        encoded = text.encode("utf-8", errors="replace")
        return {
            "text_sha256": hashlib.sha256(encoded).hexdigest(),
            "characters": len(text),
            "preview": text[:240],
        }

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(1, int((time.perf_counter() - started) * 1000))

    def _record_stage(
        self,
        session: Optional[AsyncSession],
        *,
        source_chunk_id: str,
        pipeline_stage: str,
        prompt_version: str,
        model_used: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        confidence: Optional[float] = None,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Append a transactional audit record when using a real DB session."""
        if session is None or not isinstance(session, AsyncSession):
            return
        from narrative_engine.storage.orm_models import ExtractionRecordORM

        session.add(
            ExtractionRecordORM(
                source_chunk_id=source_chunk_id,
                pipeline_stage=pipeline_stage,
                prompt_version=prompt_version,
                model_used=model_used,
                input=input_data,
                output=output_data,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                error_message=error_message,
            )
        )

    def _apply_normalized_dates(self, episode: Episode, setting: Dict[str, Any]) -> Episode:
        """Apply normalized dates while retaining signed BCE years.

        Python and PostgreSQL datetimes begin at 1 CE. Signed year columns are
        therefore authoritative for chronological ordering; compatible CE
        values also receive datetime bounds for existing consumers.
        """
        import calendar
        import re
        from datetime import datetime, timezone

        def parse_partial(value: Optional[str], *, end_bound: bool) -> tuple[Optional[int], Optional[datetime]]:
            if value is None:
                return None, None
            if not isinstance(value, str):
                raise ValueError(f"invalid normalized date {value!r}")
            match = re.fullmatch(r"(-?\d{1,4})(?:-(\d{2})(?:-(\d{2}))?)?", value)
            if not match:
                raise ValueError(f"invalid normalized date {value!r}")
            year = int(match.group(1))
            if year == 0:
                raise ValueError("historical dates do not use year zero")
            month = int(match.group(2) or (12 if end_bound else 1))
            calendar_year = year if year > 0 else 2000
            day = int(match.group(3) or (calendar.monthrange(calendar_year, month)[1] if end_bound else 1))
            # Validate month/day even for BCE values, which cannot be placed
            # in datetime. The signed year remains usable by graph ordering.
            datetime(calendar_year, month, day)
            date = datetime(year, month, day, tzinfo=timezone.utc) if year > 0 else None
            return year, date

        try:
            episode.start_year, episode.start_date = parse_partial(setting.get("start_date"), end_bound=False)
            episode.end_year, episode.end_date = parse_partial(setting.get("end_date"), end_bound=True)
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
        confidence = self._bounded_confidence(actor_data.get("role_fit_confidence"))
        if confidence is None or confidence < self.config.role_fit_floor:
            return None
        return candidate

    async def _classify_episode(
        self,
        episode: Episode,
        source_text: Optional[str] = None,
        source_chunk_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> None:
        """Classify a neutral situation pattern and optional legacy arc."""
        # First pass classification
        classification_text = source_text or f"{episode.title}\n{episode.summary}"
        stage_started = time.perf_counter()
        try:
            classification = await self.pipeline.classify(
                episode_summary=episode.summary,
                full_text=classification_text,
            )
        except Exception as exc:
            self._record_stage(
                session,
                source_chunk_id=source_chunk_id or self._episode_source_chunk(episode),
                pipeline_stage="classification",
                prompt_version="2.0.0",
                model_used=self.config.classification_model,
                input_data={
                    "episode_id": str(episode.id),
                    "summary": episode.summary,
                    **self._text_audit_input(classification_text),
                },
                output_data={},
                processing_time_ms=self._elapsed_ms(stage_started),
                error_message=str(exc),
            )
            raise
        if not isinstance(classification, dict):
            raise TypeError("Classification response must be a JSON object")

        self._record_stage(
            session,
            source_chunk_id=source_chunk_id or self._episode_source_chunk(episode),
            pipeline_stage="classification",
            prompt_version="2.0.0",
            model_used=self.config.classification_model,
            input_data={
                "episode_id": str(episode.id),
                "summary": episode.summary,
                **self._text_audit_input(classification_text),
            },
            output_data=classification,
            confidence=self._bounded_confidence(classification.get("phase_confidence")),
            processing_time_ms=self._elapsed_ms(stage_started),
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
        for domain in self._as_list(classification.get("domains")):
            with suppress(ValueError, TypeError):
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
        for family in self._as_list(classification.get("mechanism_families")):
            with suppress(ValueError, TypeError):
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
                episode.arc_phase = _parse_arc_phase(arc_phase_str)
            except ValueError:
                self.logger.warning(f"Unknown arc phase: {arc_phase_str}")

        episode.phase_confidence = self._bounded_confidence(classification.get("phase_confidence"), 0.0) or 0.0
        episode.arc_rationale = classification.get("rationale")

        # Handle secondary arcs
        for sec in self._as_list(classification.get("secondary_arcs")):
            if not isinstance(sec, dict):
                continue
            with suppress(ValueError, TypeError):
                confidence = self._bounded_confidence(sec.get("confidence"), 0.5)
                phase = sec.get("phase", "unknown")
                episode.secondary_arcs.append(
                    (
                        ArcType(sec.get("type", "unknown")),
                        _parse_arc_phase(phase),
                        confidence if confidence is not None else 0.5,
                    )
                )

        # Handle mechanism tags (design doc Sec 3.8): unrecognized tags are
        # skipped, not fatal -- the LLM occasionally drifts from the
        # vocabulary given in the prompt.
        episode.mechanism_tags = []
        for tag in self._as_list(classification.get("mechanism_tags")):
            with suppress(ValueError, TypeError):
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

        # A legacy arc is an ordered (type, phase) reading.  Keeping a type
        # after the model invents an unusable phase creates an instance that
        # composition cannot place, so both labels are required.
        arc_clears_floor = (
            episode.arc_type is not None
            and episode.arc_phase is not None
            and episode.phase_confidence >= floor
        )
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
                episode.start_year = start.year
                episode.end_year = end.year
                episode.date_precision = "range"
            else:
                # Single date
                date = date_parser.parse(time_period, fuzzy=True)
                episode.start_date = date
                episode.start_year = date.year
                episode.date_precision = precision

        except Exception as e:
            self.logger.warning(f"Failed to parse date: {time_period}", error=str(e))

        return episode

    async def reconcile_document_phases(
        self,
        chunk_ids: List[str],
        session: AsyncSession,
    ) -> List[Episode]:
        """Reconcile phase labels after all episodes in one work are visible.

        Per-episode classification remains useful for immediate checkpointing;
        this bounded second pass supplies the chronology that an isolated
        passage cannot. Only validated, above-floor assignments are applied.
        """
        from narrative_engine.composition.identity import _episode_date_key
        from narrative_engine.scopes import scope_partition_key
        from narrative_engine.storage.orm_models import EpisodeORM
        from narrative_engine.storage.repositories import EpisodeRepository

        episodes = await EpisodeRepository(session).get_for_chunks(chunk_ids)
        groups: Dict[str, List[Episode]] = {}
        for episode in episodes:
            if (
                episode.scope_confidence is not None
                and episode.scope_confidence < self.config.scope_confidence_floor
            ):
                continue
            key = scope_partition_key(episode.scope_id, episode.scope_name)
            if key:
                groups.setdefault(key, []).append(episode)

        updated: List[Episode] = []
        document_id = f"{self._work_id_from_chunk_id(chunk_ids[0])}:document" if chunk_ids else "unknown:document"
        for group in groups.values():
            if len(group) < 2:
                continue
            ordered = sorted(group, key=_episode_date_key)
            # Keep prompts bounded. Adjacent windows overlap by one episode
            # so a boundary event retains context without quadratic growth.
            windows = [ordered[index : index + 20] for index in range(0, len(ordered), 19)]
            for window in windows:
                if len(window) < 2:
                    continue
                payload = [
                    {
                        "episode_id": str(episode.id),
                        "date": episode.start_year
                        or (episode.start_date.year if episode.start_date else None),
                        "title": episode.title,
                        "summary": episode.summary,
                        "existing_arc_type": episode.arc_type.value if episode.arc_type else None,
                        "existing_phase": episode.arc_phase.value if episode.arc_phase else None,
                        "source_excerpt": (
                            episode.source_passages[0].text[:500]
                            if episode.source_passages
                            else None
                        ),
                    }
                    for episode in window
                ]
                scope_name = window[0].scope_name or window[0].scope_id or "unknown"
                stage_started = time.perf_counter()
                try:
                    result = await self.pipeline.reconcile_phases(scope_name, payload)
                except Exception as exc:
                    self._record_stage(
                        session,
                        source_chunk_id=document_id,
                        pipeline_stage="phase_reconciliation",
                        prompt_version="1.0.0",
                        model_used=self.config.reconciliation_model,
                        input_data={
                            "scope": scope_name,
                            "episode_ids": [item["episode_id"] for item in payload],
                        },
                        output_data={},
                        processing_time_ms=self._elapsed_ms(stage_started),
                        error_message=str(exc),
                    )
                    self.logger.warning(
                        "Document phase reconciliation unavailable",
                        scope=scope_name,
                        error=str(exc),
                    )
                    continue
                if not isinstance(result, dict):
                    continue
                self._record_stage(
                    session,
                    source_chunk_id=document_id,
                    pipeline_stage="phase_reconciliation",
                    prompt_version="1.0.0",
                    model_used=self.config.reconciliation_model,
                    input_data={
                        "scope": scope_name,
                        "episode_ids": [item["episode_id"] for item in payload],
                    },
                    output_data=result,
                    processing_time_ms=self._elapsed_ms(stage_started),
                )

                by_id = {str(episode.id): episode for episode in window}
                assignments = result.get("episodes")
                if not isinstance(assignments, list):
                    continue
                for assignment in assignments:
                    if not isinstance(assignment, dict):
                        continue
                    episode = by_id.get(str(assignment.get("episode_id")))
                    confidence = self._bounded_confidence(assignment.get("confidence"), 0.0) or 0.0
                    if episode is None or confidence < self.config.classification_confidence_floor:
                        continue
                    try:
                        arc_type = ArcType(assignment.get("arc_type"))
                        arc_phase = _parse_arc_phase(assignment.get("arc_phase"))
                    except (TypeError, ValueError):
                        continue

                    episode.arc_type = arc_type
                    episode.arc_phase = arc_phase
                    episode.phase_confidence = confidence
                    reason = assignment.get("reason")
                    if isinstance(reason, str) and reason.strip():
                        episode.arc_rationale = f"Document reconciliation: {reason.strip()}"
                    episode.classification_state = ClassificationState.CLASSIFIED

                    orm = await session.get(EpisodeORM, episode.id)
                    if orm is None:
                        continue
                    orm.arc_type = episode.arc_type
                    orm.arc_phase = episode.arc_phase
                    orm.phase_confidence = episode.phase_confidence
                    orm.arc_rationale = episode.arc_rationale
                    orm.classification_state = episode.classification_state.value
                    if all(existing.id != episode.id for existing in updated):
                        updated.append(episode)

        return updated

    async def link_document_candidates(
        self,
        chunk_ids: List[str],
        session: AsyncSession,
    ) -> int:
        """Link new episodes to plausible same-scope episodes in the corpus."""
        if not self.config.enable_linking or self.config.cross_link_max_pairs == 0:
            return 0

        from sqlalchemy import and_, or_, select
        from sqlalchemy.orm import selectinload

        from narrative_engine.storage.orm_models import EpisodeLinkORM, EpisodeORM
        from narrative_engine.storage.repositories import EpisodeRepository

        repository = EpisodeRepository(session)
        new_episodes = await repository.get_for_chunks(chunk_ids)
        attempted: set[frozenset] = set()
        linked = 0
        calls = 0

        for source in new_episodes:
            if calls >= self.config.cross_link_max_pairs:
                break
            if not source.scope_id and not source.scope_name:
                continue

            scope_filter = (
                or_(
                    EpisodeORM.scope_id == source.scope_id,
                    EpisodeORM.scope_name == source.scope_name,
                )
                if source.scope_id and source.scope_name
                else (
                    EpisodeORM.scope_id == source.scope_id
                    if source.scope_id
                    else EpisodeORM.scope_name == source.scope_name
                )
            )
            result = await session.execute(
                select(EpisodeORM)
                .where(EpisodeORM.id != source.id, scope_filter)
                .options(
                    selectinload(EpisodeORM.actors),
                    selectinload(EpisodeORM.source_passages),
                )
                .limit(100)
            )
            candidates = [repository._from_orm(orm) for orm in result.scalars().unique().all()]
            candidates = [
                candidate
                for candidate in candidates
                if not set(source.extracted_from) & set(candidate.extracted_from)
                and self._is_link_candidate(source, candidate, distance=10_000)
            ]
            candidates.sort(
                key=lambda candidate: self._cross_link_candidate_score(source, candidate),
                reverse=True,
            )

            for candidate in candidates[:3]:
                if calls >= self.config.cross_link_max_pairs:
                    break
                pair = frozenset((source.id, candidate.id))
                if pair in attempted:
                    continue
                attempted.add(pair)

                existing = await session.execute(
                    select(EpisodeLinkORM.id)
                    .where(
                        or_(
                            and_(
                                EpisodeLinkORM.source_episode_id == source.id,
                                EpisodeLinkORM.target_episode_id == candidate.id,
                            ),
                            and_(
                                EpisodeLinkORM.source_episode_id == candidate.id,
                                EpisodeLinkORM.target_episode_id == source.id,
                            ),
                        )
                    )
                    .limit(1)
                )
                if existing.scalar_one_or_none() is not None:
                    continue

                source_year = self._episode_start_year_value(source)
                candidate_year = self._episode_start_year_value(candidate)
                first, second = source, candidate
                if (
                    source_year is not None
                    and candidate_year is not None
                    and candidate_year < source_year
                ):
                    first, second = candidate, source
                calls += 1
                try:
                    linked += int(await self._link_episode_pair(first, second, session))
                except Exception as exc:
                    self.logger.warning(
                        "Cross-document linking candidate failed",
                        source=first.title,
                        target=second.title,
                        error=str(exc),
                    )

        self.logger.info(
            "Cross-document linking complete",
            candidates_checked=calls,
            links_created=linked,
        )
        return linked

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
    ) -> bool:
        """Persist a supported identity/causal relationship for one pair."""
        from narrative_engine.storage.orm_models import EpisodeLinkORM

        stage_started = time.perf_counter()
        try:
            result = await self.pipeline.link(
                source.model_dump(mode="json"),
                target.model_dump(mode="json"),
            )
        except Exception as exc:
            self._record_stage(
                session,
                source_chunk_id=self._episode_source_chunk(source),
                pipeline_stage="linking",
                prompt_version="1.0.0",
                model_used=self.config.linking_model,
                input_data={
                    "source_episode_id": str(source.id),
                    "target_episode_id": str(target.id),
                },
                output_data={},
                processing_time_ms=self._elapsed_ms(stage_started),
                error_message=str(exc),
            )
            raise
        if not isinstance(result, dict):
            return False
        self._record_stage(
            session,
            source_chunk_id=self._episode_source_chunk(source),
            pipeline_stage="linking",
            prompt_version="1.0.0",
            model_used=self.config.linking_model,
            input_data={
                "source_episode_id": str(source.id),
                "target_episode_id": str(target.id),
            },
            output_data=result,
            confidence=self._bounded_confidence(result.get("confidence")),
            processing_time_ms=self._elapsed_ms(stage_started),
        )
        relationship = result.get("relationship")
        if relationship not in {"same_event", "causes", "caused_by"}:
            return False
        if float(result.get("confidence", 0.0)) < 0.5:
            return False

        evidence = result.get("evidence_quote")
        if relationship in {"causes", "caused_by"} and not evidence:
            self.logger.warning(
                "Rejected unsupported causal link",
                source=source.title,
                target=target.title,
                reason="missing evidence quote",
            )
            return False
        if evidence:
            supplied_text = "\n".join(
                [
                    source.title,
                    source.summary,
                    *(passage.text for passage in source.source_passages),
                    target.title,
                    target.summary,
                    *(passage.text for passage in target.source_passages),
                ]
            )
            if evidence not in supplied_text:
                self.logger.warning(
                    "Rejected unsupported episode link",
                    source=source.title,
                    target=target.title,
                    reason="evidence quote not present in source passages",
                )
                return False

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
        return True

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
    def _episode_start_year_value(episode: Episode) -> Optional[int]:
        if episode.start_year is not None:
            return episode.start_year
        return episode.start_date.year if episode.start_date else None

    def _cross_link_candidate_score(self, source: Episode, target: Episode) -> float:
        """Rank already-prefiltered pairs before spending a linking call."""
        source_terms = self._link_terms(self._episode_link_text(source))
        target_terms = self._link_terms(self._episode_link_text(target))
        union = source_terms | target_terms
        lexical = len(source_terms & target_terms) / len(union) if union else 0.0

        source_actors = {self._normalize_link_value(actor.name) for actor in source.actors}
        target_actors = {self._normalize_link_value(actor.name) for actor in target.actors}
        actor_union = (source_actors | target_actors) - {""}
        actor_overlap = (
            len((source_actors & target_actors) - {""}) / len(actor_union)
            if actor_union
            else 0.0
        )

        year_gap = self._episode_year_gap(source, target)
        temporal = (
            max(0.0, 1.0 - year_gap / max(1.0, self.config.linking_max_year_gap))
            if year_gap is not None
            else 0.0
        )
        title_match = float(
            self._normalize_link_value(source.title) == self._normalize_link_value(target.title)
        )
        return title_match + 0.4 * lexical + 0.35 * actor_overlap + 0.25 * temporal

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
        if source.start_date is not None and target.start_date is not None:
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

        source_start_year = source.start_year
        target_start_year = target.start_year
        if source_start_year is None or target_start_year is None:
            return None
        source_end_year = source.end_year if source.end_year is not None else source_start_year
        target_end_year = target.end_year if target.end_year is not None else target_start_year
        if source_end_year < target_start_year:
            return float(target_start_year - source_end_year)
        if target_end_year < source_start_year:
            return float(source_start_year - target_end_year)
        return 0.0

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
