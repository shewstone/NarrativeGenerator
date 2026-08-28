"""Scope registry and alias resolver (T5, docs/tickets/T5-scope-registry.md).

The scope partition is composition's stage-1 hard filter (design doc
Sec 6.2 stage 6): inconsistent raw scope labels from extraction ("US" vs
"United States") silently fragment arc instances. This module resolves raw
strings against a versioned alias registry.

Resolution is exact-alias-only, deliberately: a WRONG scope silently
poisons the composition partition, while an UNRESOLVED scope falls into
the v0.7 unscoped-singleton path, which is visible and safe. Same
asymmetry logic as the evidence floor — absence of evidence must never
behave like evidence.

The registry itself is versioned data, not ontology (Sec 9: scope
boundaries are contested claims). Registry changes are data changes plus a
version bump in scope_registry.json; no code change.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Dict, List, Optional

from narrative_engine.logging_config import get_logger
from narrative_engine.models import Scope

logger = get_logger(__name__)

_REGISTRY_PACKAGE = "narrative_engine.data"
_REGISTRY_RESOURCE = "scope_registry.json"

_ARTICLE_PREFIXES = ("the ",)
DEFAULT_SCOPE_CONFIDENCE_FLOOR = 0.6


def _normalize(raw: str) -> str:
    """Casefold, strip punctuation/articles, collapse whitespace."""
    text = raw.casefold().strip()
    for prefix in _ARTICLE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class ScopeRegistry:
    """In-memory registry: id -> Scope, normalized alias -> id."""

    def __init__(self, version: str, scopes: List[Scope]):
        seen_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        for scope in scopes:
            if scope.id in seen_ids:
                duplicate_ids.add(scope.id)
            seen_ids.add(scope.id)
        if duplicate_ids:
            raise ValueError(f"Duplicate scope ids: {sorted(duplicate_ids)}")
        self.version = version
        self._by_id: Dict[str, Scope] = {s.id: s for s in scopes}
        self._alias_to_id: Dict[str, str] = {}
        for scope in scopes:
            for alias in [scope.id, scope.name, *scope.aliases]:
                key = _normalize(alias)
                existing = self._alias_to_id.get(key)
                if existing and existing != scope.id:
                    raise ValueError(
                        f"Alias collision in scope registry: {alias!r} maps to both {existing!r} and {scope.id!r}"
                    )
                self._alias_to_id[key] = scope.id
        self._validate_hierarchy()

    def _validate_hierarchy(self) -> None:
        """Reject missing parents and containment cycles at load time."""
        for scope in self._by_id.values():
            if scope.parent_scope_id and scope.parent_scope_id not in self._by_id:
                raise ValueError(f"Scope {scope.id!r} has unknown parent {scope.parent_scope_id!r}")
            seen: set[str] = set()
            current: Optional[Scope] = scope
            while current is not None:
                if current.id in seen:
                    raise ValueError(f"Cycle in scope hierarchy at {current.id!r}")
                seen.add(current.id)
                current = self._by_id.get(current.parent_scope_id) if current.parent_scope_id else None

    @classmethod
    def load(cls) -> "ScopeRegistry":
        raw = resources.files(_REGISTRY_PACKAGE).joinpath(_REGISTRY_RESOURCE).read_text()
        data = json.loads(raw)
        scopes = [Scope(**entry) for entry in data["scopes"]]
        return cls(version=data["version"], scopes=scopes)

    def resolve(self, raw: Optional[str]) -> Optional[str]:
        """Resolve a raw scope string to a registry scope id, or None.

        None means "unresolved": callers must keep the episode on the
        visible unscoped/raw-string path, never guess. Unresolved strings
        are logged — they are the promotion queue for new aliases.
        """
        if not raw:
            return None
        scope_id = self._alias_to_id.get(_normalize(raw))
        if scope_id is None:
            logger.info("scope_unresolved", raw=raw)
        return scope_id

    def get(self, scope_id: str) -> Optional[Scope]:
        return self._by_id.get(scope_id)

    def all(self) -> List[Scope]:
        return list(self._by_id.values())

    def lineage(self, raw: Optional[str]) -> List[Scope]:
        """Return focal scope followed by each containing parent.

        Example: faction -> party -> polity -> civilization. Unknown raw
        labels return an empty list rather than being guessed into a tree.
        """
        scope_id = self.resolve(raw)
        current = self._by_id.get(scope_id) if scope_id else None
        result: List[Scope] = []
        while current is not None:
            result.append(current)
            current = self._by_id.get(current.parent_scope_id) if current.parent_scope_id else None
        return result

    def descendants(self, raw: Optional[str]) -> List[Scope]:
        """Return every registered scope nested under ``raw``."""
        scope_id = self.resolve(raw)
        if scope_id is None:
            return []
        return [
            scope
            for scope in self._by_id.values()
            if any(parent.id == scope_id for parent in self.lineage(scope.id)[1:])
        ]

    def is_within(self, child: Optional[str], ancestor: Optional[str]) -> bool:
        """Whether ``child`` is the same scope as, or nested under, ancestor."""
        ancestor_id = self.resolve(ancestor)
        return bool(ancestor_id and any(scope.id == ancestor_id for scope in self.lineage(child)))


@lru_cache(maxsize=1)
def get_registry() -> ScopeRegistry:
    return ScopeRegistry.load()


def resolve_scope(raw: Optional[str]) -> Optional[str]:
    """Module-level convenience wrapper over the packaged registry."""
    return get_registry().resolve(raw)


def scope_registry_version() -> str:
    return get_registry().version


def scope_partition_key(scope_id: Optional[str], scope_name: Optional[str] = None) -> Optional[str]:
    """Canonical composition key for registered and newly observed scopes.

    Unknown subgroup names remain explicit normalized partitions. They never
    fall into a shared ``None`` bucket, and can later be promoted into the
    versioned registry without losing the original extraction claim.
    """
    raw = scope_id or scope_name
    if not raw:
        return None
    return resolve_scope(raw) or f"raw:{_normalize(raw)}"
