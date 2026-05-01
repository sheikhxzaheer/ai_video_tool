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
from footage.qa_store import load_brain


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
    
    winner_dict = winners_signal if isinstance(winners_signal, dict) else {}
    qa_dict = qa_signal if isinstance(qa_signal, dict) else {}

    for c in candidates:
        base_score = float(c.get("similarity", 0.0))
        winner_score = 0.0
        qa_score = 0.0

        clip_tags = c.get("structural_tags", []) + c.get("visual_keywords", [])
        for tag in clip_tags:
            qa_score += qa_dict.get(tag, 0.0)
            winner_score += winner_dict.get(tag, 0.0)

        final_score = base_score + winner_score + qa_score
        c["final_similarity_score"] = final_score
        c["score_explanation"] = f"Score: {final_score:.4f} (Semantic: {base_score:.4f}, Winner: {winner_score:+.4f}, QA: {qa_score:+.4f})"

    candidates.sort(key=lambda x: x.get("final_similarity_score", 0.0), reverse=True)
    return candidates

