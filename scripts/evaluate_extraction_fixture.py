#!/usr/bin/env python3
"""Run the live LLM pipeline against the seed extraction gold fixture."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict

from narrative_engine.evaluation.extraction_fixture import EXTRACTION_CASES
from narrative_engine.evaluation.extraction_quality import PredictedEpisode, score_extraction
from narrative_engine.extraction.client import ExtractionPipeline
from narrative_engine.extraction.config import ExtractionPipelineConfig
from narrative_engine.scopes import get_registry, resolve_scope, suggest_scopes


def _year(value) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.match(r"^-?\d{1,4}", value)
    return int(match.group()) if match else None


async def _canonical_scope(
    pipeline: ExtractionPipeline,
    focal_scope: dict,
) -> str | None:
    raw_name = focal_scope.get("name")
    if not isinstance(raw_name, str):
        return None
    exact = resolve_scope(raw_name)
    if exact:
        return exact
    raw_kind = focal_scope.get("kind")
    parent_name = focal_scope.get("parent_name")
    suggestions = suggest_scopes(
        raw_name,
        kind=raw_kind if isinstance(raw_kind, str) else None,
        parent_name=parent_name if isinstance(parent_name, str) else None,
    )
    if not suggestions or suggestions[0].score < 0.58:
        return None
    registry = get_registry()
    candidates = []
    for suggestion in suggestions:
        parent = registry.get(suggestion.scope.parent_scope_id) if suggestion.scope.parent_scope_id else None
        candidates.append(
            {
                "id": suggestion.scope.id,
                "name": suggestion.scope.name,
                "kind": suggestion.scope.kind.value,
                "parent": parent.name if parent else None,
                "retrieval_score": suggestion.score,
            }
        )
    result = await pipeline.canonicalize_scope(
        raw_name,
        raw_kind if isinstance(raw_kind, str) else None,
        parent_name if isinstance(parent_name, str) else None,
        focal_scope.get("evidence_quote"),
        candidates,
    )
    allowed = {candidate["id"] for candidate in candidates}
    selected = result.get("scope_id")
    confidence = result.get("confidence")
    return (
        selected
        if selected in allowed and isinstance(confidence, (int, float)) and confidence >= 0.8
        else None
    )


async def main() -> None:
    config = ExtractionPipelineConfig.from_env()
    pipeline = ExtractionPipeline(config=config)
    reports = []
    try:
        for case in EXTRACTION_CASES:
            segmented = await pipeline.segment(case.text)
            narrative_context = "\n".join(
                f"{index}. {str(segment.get('summary') or '')[:240]}"
                for index, segment in enumerate(segmented.get("episodes", []), start=1)
            )
            predicted = []
            for segment in segmented.get("episodes", []):
                extracted = await pipeline.extract(
                    segment_text=segment["text"],
                    segment_summary=segment.get("summary", ""),
                    narrative_context=narrative_context,
                )
                focal_scope = extracted.get("focal_scope")
                scope_id = (
                    await _canonical_scope(pipeline, focal_scope)
                    if isinstance(focal_scope, dict)
                    else None
                )
                setting = extracted.get("setting")
                setting = setting if isinstance(setting, dict) else {}
                predicted.append(
                    PredictedEpisode(
                        start_char=int(segment.get("start_char", 0)),
                        end_char=int(segment.get("end_char", len(case.text))),
                        scope_id=scope_id,
                        start_year=_year(setting.get("start_date")),
                    )
                )
            score = score_extraction(case.episodes, predicted)
            reports.append({"case": case.name, **asdict(score)})
    finally:
        await pipeline.aclose()

    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
