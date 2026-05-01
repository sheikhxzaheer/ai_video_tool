"""Pluggable reranking hooks for Winners + QA corrections.

This module is intentionally lightweight:
- It provides a stable interface the matcher can call.
- Default behavior is no-op (doesn't change rankings).

You can later wire real data sources:
- Winners "gold standard" similarity signals
- QA correction logs: [text] + [rejected] + [accepted]
"""





from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def load_brain(filepath="learning_weights.json"):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Bhai JSON read karne mein error aaya: {e}")

    return {"winner_tags": [], "qa_rejected_tags": []}






def save_rejection(bad_video_data, filepath="learning_weights.json"):
    brain = load_brain(filepath)
    bad_tags = bad_video_data.get("structural_tags", []) + bad_video_data.get("visual_keywords", [])
    for tag in bad_tags:
        if tag and tag not in brain["qa_rejected_tags"]:
            brain["qa_rejected_tags"].append(tag)
    with open(filepath, "w") as f:
        json.dump(brain, f, indent=2)

    print(f"Buri video reject ho gayi! Naye tags add ho gaye: {bad_tags}")


def save_winner(good_video_data, filepath="learning_weights.json"):
    brain = load_brain(filepath)
    good_tags = good_video_data.get("structural_tags", []) + good_video_data.get("visual_keywords", [])
    for tag in good_tags:
        if tag and tag not in brain["winner_tags"]:
            brain["winner_tags"].append(tag)
    with open(filepath, "w") as f:
        json.dump(brain, f, indent=2)
    print(f"Badhai ho! Nayi winner video mili. Naye tags add ho gaye: {good_tags}")





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
    
    WINNER_BOOST = 0.15
    QA_PENALTY = 0.20

    winner_tags = winners_signal if isinstance(winners_signal, list) else []
    qa_rejected_tags = qa_signal if isinstance(qa_signal, list) else []

    for c in candidates:
        base_score = float(c.get("similarity", 0.0))
        winner_score = 0.0
        qa_score = 0.0

        clip_tags = c.get("structural_tags", []) + c.get("visual_keywords", [])
        has_qa_tag = any(tag in qa_rejected_tags for tag in clip_tags)
        if has_qa_tag:
            qa_score = -QA_PENALTY

        has_winner_tag = any(tag in winner_tags for tag in clip_tags)
        if has_winner_tag:
            winner_score = WINNER_BOOST

        final_score = base_score + winner_score + qa_score
        c["final_similarity_score"] = final_score
        c["score_explanation"] = f"Score: {final_score:.4f} (Semantic: {base_score:.4f}, Winner: +{winner_score:.4f}, QA: {qa_score:.4f})"

    candidates.sort(key=lambda x: x.get("final_similarity_score", 0.0), reverse=True)
    return candidates

