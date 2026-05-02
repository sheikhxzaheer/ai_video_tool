"""Match segments to footage using ChromaDB vector search and penalty logic."""

import random
from typing import Any, Dict, List, Mapping, Optional

import chromadb

from footage.embeddings import embed_texts_batch
from footage.reranker import RerankContext, rerank_candidates
from footage.qa_store import load_brain


# Default ChromaDB path
DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_COLLECTION = "footage"


def _get_collection(chroma_path: str = DEFAULT_CHROMA_PATH, collection_name: str = DEFAULT_COLLECTION):
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_collection(name=collection_name)


def _tags_to_query(segment: Dict[str, Any]) -> str:
    """Combine segment text and tags into search query string."""
    text = (segment.get("text") or "").strip()
    tags = segment.get("tags") or []
    additional = segment.get("additional_tags") or []
    parts = [text] + [str(p) for p in tags if p] + [str(p) for p in additional if p]
    return " ".join(p for p in parts if p).strip()

def _split_pipe_list(s: Any) -> List[str]:
    if not s:
        return []
    if isinstance(s, list):
        return [str(x).strip() for x in s if str(x).strip()]
    txt = str(s).strip()
    if not txt:
        return []
    if "|" in txt:
        return [p.strip() for p in txt.split("|") if p.strip()]
    return [txt]


def _candidate_from_metadata(m: Mapping[str, Any], distance: Optional[float]) -> Dict[str, Any]:
    d = float(distance) if distance is not None else None
    similarity = 1 - d if d is not None else 0.0
    video_path = m.get("video_path") or m.get("path") or ""
    seg_start = m.get("segment_start")
    seg_end = m.get("segment_end")
    seg_duration = m.get("segment_duration")
    return {
        "video_path": video_path,
        "path": video_path,  # backward compatibility
        "video_duration": float(m.get("video_duration") or m.get("duration") or 0.0),
        "segment_start": float(seg_start) if seg_start is not None else None,
        "segment_end": float(seg_end) if seg_end is not None else None,
        "segment_duration": float(seg_duration) if seg_duration is not None else None,
        "structural_tags": _split_pipe_list(m.get("structural_tags")),
        "visual_keywords": _split_pipe_list(m.get("visual_keywords")),
        "role": (m.get("role") or "").strip().upper(),
        "shot_type": (m.get("shot_type") or "").strip().lower(),
        "environment": _split_pipe_list(m.get("environment")),
        "mood": _split_pipe_list(m.get("mood")),
        "style_keywords": _split_pipe_list(m.get("style_keywords")),
        "people_present": m.get("people_present"),
        "product_present": m.get("product_present"),
        "brand": m.get("brand") or "Unknown",
        "similarity": float(similarity),
        "distance": d,
        "metadata": m,
    }


def _compute_in_out(
    clip_duration: float,
    segment_duration: float,
) -> tuple[float, float]:
    """Compute in/out points for segment within clip."""
    if clip_duration <= 0 or segment_duration <= 0:
        return 0.0, min(segment_duration, 5.0)
    if segment_duration >= clip_duration:
        return 0.0, clip_duration
    max_start = clip_duration - segment_duration
    in_point = random.uniform(0, max(0, max_start))
    out_point = in_point + segment_duration
    return round(in_point, 2), round(out_point, 2)


def _apply_penalties(
    candidates: List[Dict[str, Any]],
    used_paths: set[str],
    recent_paths: List[str],
    penalty_recent: float = 0.3,
    penalty_reused: float = 0.5,
    recent_window: int = 5,
) -> List[Dict[str, Any]]:
    """Apply penalties for repetition; return re-sorted candidates."""
    scored: List[tuple[float, Dict[str, Any]]] = []
    for c in candidates:
        score = c.get("similarity", 0)
        if score <= 0 and "distance" in c:
            score = 1 - c["distance"]
        path = c.get("path", "")
        if path in used_paths:
            score -= penalty_reused
        for i, rp in enumerate(reversed(recent_paths[-recent_window:])):
            if path == rp:
                score -= penalty_recent * (1 - i / recent_window)
                break
        scored.append((max(0, score), c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored]

def _top_sources(candidates: List[Dict[str, Any]], max_sources: int = 5) -> List[str]:
    """
    Pick top source videos by best candidate similarity.
    """
    best_by_video: Dict[str, float] = {}
    for c in candidates:
        vp = c.get("video_path") or c.get("path") or ""
        if not vp:
            continue
        score = float(c.get("similarity") or 0.0)
        if vp not in best_by_video or score > best_by_video[vp]:
            best_by_video[vp] = score
    ordered = sorted(best_by_video.items(), key=lambda kv: -kv[1])
    return [vp for vp, _ in ordered[:max_sources]]


def _contradiction_penalty(anchor: Dict[str, Any], cand: Dict[str, Any], penalty: float = 0.15) -> float:
    """
    Lightweight contradiction heuristic:
    if candidate keywords have low overlap with anchor keywords, apply penalty.
    This is intentionally conservative; you can replace it with Winners/QA-trained weights.
    """
    # Prefer structured attributes when available; fall back to visual keyword overlap.
    a_env = {e.lower() for e in (anchor.get("environment") or []) if e}
    c_env = {e.lower() for e in (cand.get("environment") or []) if e}
    a_mood = {m.lower() for m in (anchor.get("mood") or []) if m}
    c_mood = {m.lower() for m in (cand.get("mood") or []) if m}

    structured_ok = bool(a_env or a_mood) and bool(c_env or c_mood)
    if structured_ok:
        env_overlap = len(a_env & c_env) / max(1, len(a_env)) if a_env and c_env else 1.0
        mood_overlap = len(a_mood & c_mood) / max(1, len(a_mood)) if a_mood and c_mood else 1.0
        return penalty if (env_overlap < 0.2 or mood_overlap < 0.2) else 0.0

    a_kw = {k.lower() for k in (anchor.get("visual_keywords") or []) if k}
    c_kw = {k.lower() for k in (cand.get("visual_keywords") or []) if k}
    if not a_kw or not c_kw:
        return 0.0
    overlap = len(a_kw & c_kw) / max(1, len(a_kw))
    return penalty if overlap < 0.15 else 0.0


def _select_anchor_and_broll(
    candidates: List[Dict[str, Any]],
    *,
    broll_count: int = 2,
    broll_contradiction_penalty: float = 0.15,
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not candidates:
        return None, []
    # Prefer explicit ANCHOR role if present.
    explicit_anchor = next((c for c in candidates if (c.get("role") or "") == "ANCHOR"), None)
    anchor = explicit_anchor or candidates[0]
    remaining = candidates[1:]
    if broll_count <= 0 or not remaining:
        return anchor, []

    scored: List[tuple[float, Dict[str, Any]]] = []
    for c in remaining:
        score = float(c.get("similarity") or 0.0)
        score -= _contradiction_penalty(anchor, c, penalty=broll_contradiction_penalty)
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    broll = [c for _, c in scored[:broll_count]]
    return anchor, broll


def match_segments_to_footage(
    segments: List[Dict[str, Any]],
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 10,
    top_sources_k: int = 5,
    enable_top_down: bool = True,
    enable_anchor_broll: bool = True,
    broll_count: int = 2,
    penalty_recent: float = 0.3,
    penalty_reused: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    For each segment, find best matching footage clip and add matched_footage.
    
    Args:
        segments: Segments with tags and additional_tags
        chroma_path: ChromaDB persistence path
        collection_name: Collection name
        top_k: Candidates to retrieve before applying penalties
        penalty_recent: Penalty for recently used clips
        penalty_reused: Penalty for already used clips
        
    Returns:
        Segments with matched_footage added
    """
    try:
        collection = _get_collection(chroma_path, collection_name)
    except Exception as e:
        print(f"ChromaDB not available: {e}. Run build_footage_index first.")
        return segments
    
    used_paths: set[str] = set()
    recent_paths: List[str] = []

    queries: List[str] = [_tags_to_query(s) for s in segments]
    query_embeddings = embed_texts_batch([q for q in queries if q])

    # Map back embeddings to segment indices (skip empty queries)
    emb_by_idx: Dict[int, List[float]] = {}
    emb_i = 0
    for i, q in enumerate(queries):
        if not q:
            continue
        emb_by_idx[i] = query_embeddings[emb_i]
        emb_i += 1

    result: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        seg_copy = dict(seg)
        query = queries[i]
        if not query:
            result.append(seg_copy)
            continue

        seg_duration = (seg.get("end") or 0) - (seg.get("start") or 0)
        q_emb = emb_by_idx.get(i)
        if not q_emb:
            result.append(seg_copy)
            continue

        try:
            base_results = collection.query(
                query_embeddings=[q_emb],
                n_results=min(max(top_k, 25), max(1, collection.count())),
                include=["metadatas", "distances"],
            )
        except Exception as e:
            print(f"Query error: {e}")
            result.append(seg_copy)
            continue

        metadatas = (base_results.get("metadatas") or [[]])[0] or []
        distances = (base_results.get("distances") or [[]])[0] or []
        base_candidates = [_candidate_from_metadata(m, d) for m, d in zip(metadatas, distances)]
        if not base_candidates:
            result.append(seg_copy)
            continue

        # Top-down: restrict to top source videos then re-query (if possible) or filter locally.
        candidates = base_candidates
        top_sources = _top_sources(base_candidates, max_sources=top_sources_k)
        if enable_top_down and top_sources:
            try:
                # Chroma supports rich filters; $in may not be available on all versions.
                where: Any = {"video_path": {"$in": top_sources}}
                restricted = collection.query(
                    query_embeddings=[q_emb],
                    n_results=min(top_k, max(1, collection.count())),
                    include=["metadatas", "distances"],
                    where=where,
                )
                metadatas2 = (restricted.get("metadatas") or [[]])[0] or []
                distances2 = (restricted.get("distances") or [[]])[0] or []
                candidates = [_candidate_from_metadata(m, d) for m, d in zip(metadatas2, distances2)]
            except Exception:
                candidates = [c for c in base_candidates if (c.get("video_path") or "") in set(top_sources)]

        if not candidates:
            result.append(seg_copy)
            continue

        penalized = _apply_penalties(
            candidates,
            used_paths=used_paths,
            recent_paths=recent_paths,
            penalty_recent=penalty_recent,
            penalty_reused=penalty_reused,
        )

        brain_data = load_brain()

        penalized = rerank_candidates(
            penalized,
            ctx=RerankContext(query_text=query, segment=seg),
            winners_signal=brain_data.get("winner_tags", {}),
            qa_signal=brain_data.get("qa_rejected_tags", {})
        )

        if not penalized:
            result.append(seg_copy)
            continue

        anchor, broll = _select_anchor_and_broll(
            penalized,
            broll_count=broll_count if enable_anchor_broll else 0,
        )
        best = anchor or penalized[0]

        # Prefer semantic-core timestamps from Gemini when available.
        in_point = best.get("segment_start")
        out_point = best.get("segment_end")
        if in_point is None or out_point is None:
            clip_duration = float(best.get("video_duration") or 0.0)
            in_point2, out_point2 = _compute_in_out(clip_duration, seg_duration)
            in_point = in_point if in_point is not None else in_point2
            out_point = out_point if out_point is not None else out_point2

        seg_copy["matched_footage"] = {
            "path": best.get("video_path") or best.get("path") or "",
            "in_point": float(in_point),
            "out_point": float(out_point),
            "similarity_score": round(float(best.get("similarity") or 0.0), 4),
            "final_similarity_score": round(float(best.get("final_similarity_score") or best.get("similarity") or 0.0), 4),
            "score_explanation": best.get("score_explanation", "No explanation available"),
        }

        alternatives = []
        best_path = best.get("video_path") or best.get("path")
        for cand in penalized:
            cand_path = cand.get("video_path") or cand.get("path")
            if cand_path != best_path:
                alt_in = cand.get("segment_start") or 0.0
                alt_out = cand.get("segment_end") or min(float(cand.get("video_duration") or 5.0), seg_duration)

                alternatives.append({
                    "path": cand_path or "",
                    "in_point": float(alt_in),
                    "out_point": float(alt_out),
                    "similarity_score": round(float(cand.get("similarity") or 0.0), 4),
                    "final_similarity_score": round(float(cand.get("final_similarity_score") or cand.get("similarity") or 0.0), 4),
                    "score_explanation": cand.get("score_explanation", "No explanation available")
                })
                if len(alternatives) >= 3:
                    break
                seg_copy["alternatives"] = alternatives

        if enable_anchor_broll and broll:
            seg_copy["matched_broll"] = [
                {
                    "path": b.get("video_path") or b.get("path") or "",
                    "in_point": float(b.get("segment_start") or 0.0),
                    "out_point": float(b.get("segment_end") or 0.0),
                    "similarity_score": round(float(b.get("similarity") or 0.0), 4),
                }
                for b in broll
                if (b.get("video_path") or b.get("path"))
            ]

        used_paths.add(seg_copy["matched_footage"]["path"])
        recent_paths.append(seg_copy["matched_footage"]["path"])
        result.append(seg_copy)

    return result
