"""Pluggable reranking hooks for Winners + QA corrections.

This module is intentionally lightweight:
- It provides a stable interface the matcher can call.
- Default behavior is no-op (doesn't change rankings).

You can later wire real data sources:
- Winners "gold standard" similarity signals
- QA correction logs: [text] + [rejected] + [accepted]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RerankContext:
    """Inputs that rerankers may use to adjust candidate ordering."""

    query_text: str
    segment: Dict[str, Any]


def rerank_candidates(
    candidates: List[Dict[str, Any]],
    *,
    ctx: RerankContext,
    winners_signal: Optional[Any] = None,
    qa_signal: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank candidates in descending desirability.

    Expected candidate keys (best-effort):
    - similarity: float (higher better)
    - video_path/path: str
    - metadata: dict

    winners_signal / qa_signal are placeholders for future integration.
    """
    # Default: keep current ordering (already similarity-sorted and penalty-adjusted).
    return candidates

