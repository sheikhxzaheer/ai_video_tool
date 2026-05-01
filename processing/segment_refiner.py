"""Refine script into timestamped subtitle blocks."""

from __future__ import annotations

import difflib
import re
from typing import List, Dict, Any, Optional, Tuple


def _norm_word(w: str) -> str:
    return "".join(ch for ch in (w or "") if ch.isalnum()).lower()


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _split_marked_text(marked_text: str) -> List[str]:
    parts = [p.strip() for p in (marked_text or "").split("|")]
    return [p for p in parts if p]


def _extract_bracket_blocks(text: str) -> List[str]:
    blocks = [m.strip() for m in re.findall(r"\[([^\]]+)\]", text or "", flags=re.DOTALL)]
    return [b for b in blocks if b]


def extract_blocks(marked_text: str) -> List[str]:
    """Extract bracketed blocks like [text] into plain strings."""
    return _extract_bracket_blocks(marked_text)


def split_pipe_blocks(marked_text: str) -> List[str]:
    """Split pipe-delimited blocks like 'a | b | c' into plain strings."""
    return _split_marked_text(marked_text)


def map_chunks_to_segments(
    chunks: List[str],
    word_timestamps: List[Dict[str, Any]],
    script_alignment: Optional[List[Dict[str, Any]]] = None,
    min_match: float = 0.8,) -> List[Dict[str, Any]]:
    """
    Map text chunks to timestamps using ASR (or script alignment) and return segments.

    Each output segment has: { "text": str, "start": float, "end": float }.
    """
    if not chunks:
        return []

    mapped: List[Tuple[float, float, str]] = []
    if script_alignment:
        mapped = _map_chunks_to_script_alignment(chunks, script_alignment, min_match=min_match)
    if not mapped:
        mapped = _map_chunks_to_timestamps(chunks, word_timestamps, min_match=min_match)

    return [
        {"text": ctext.strip(), "start": round(float(cs), 3), "end": round(float(ce), 3)}
        for cs, ce, ctext in mapped
        if (ctext or "").strip()
    ]

def _map_chunks_to_timestamps(
    chunks: List[str],
    asr_words: List[Dict[str, Any]],
    min_match: float = 0.8,) -> List[Tuple[float, float, str]]:
    """Map chunk text to ASR tokens in sequence."""
    if not chunks or not asr_words:
        return []

    asr_norm = [_norm_word(w.get("word", "")) for w in asr_words]
    out: List[Tuple[float, float, str]] = []

    j = 0
    prev_end: Optional[float] = None

    for chunk in chunks:
        words = [_norm_word(w) for w in chunk.split() if _norm_word(w)]
        if not words:
            continue

        chunk_start: Optional[float] = None
        last_end: Optional[float] = None

        for cw in words:
            found = False
            scan = j
            while scan < len(asr_words):
                aw = asr_norm[scan]
                if aw and (aw == cw or _ratio(aw, cw) >= min_match):
                    if chunk_start is None:
                        chunk_start = asr_words[scan]["start"]
                    last_end = asr_words[scan]["end"]
                    scan += 1
                    j = scan
                    found = True
                    break
                scan += 1
            if not found:
                continue

        if chunk_start is None or last_end is None:
            continue

        if prev_end is not None and chunk_start < prev_end:
            chunk_start = prev_end
        if last_end < chunk_start:
            continue

        out.append((float(chunk_start), float(last_end), chunk))
        prev_end = float(last_end)

    return out


def _map_chunks_to_script_alignment(
    chunks: List[str],
    script_alignment: List[Dict[str, Any]],
    min_match: float = 0.8,) -> List[Tuple[float, float, str]]:
    """Map chunks to script words and use aligned timestamps."""
    if not chunks or not script_alignment:
        return []

    aligned = [
        a for a in script_alignment
        if a.get("matched") and a.get("start") is not None and a.get("end") is not None
    ]
    if not aligned:
        return []

    script_norm = [_norm_word(a.get("script_word", "")) for a in aligned]
    out: List[Tuple[float, float, str]] = []

    j = 0
    prev_end: Optional[float] = None

    min_chunk_match_ratio = 0.6

    for chunk in chunks:
        words = [_norm_word(w) for w in chunk.split() if _norm_word(w)]
        if not words:
            continue

        chunk_start: Optional[float] = None
        last_end: Optional[float] = None
        matched_count = 0
        k = j

        # Only advance k on successful word match. If a chunk word has no aligned
        # token (e.g. script "delete" in diff), do not scan to end of alignment — that
        # would commit j past all later script words and drop remaining sentences.
        for cw in words:
            found = False
            scan = k
            while scan < len(aligned):
                sw = script_norm[scan]
                if sw and (sw == cw or _ratio(sw, cw) >= min_match):
                    if chunk_start is None:
                        chunk_start = float(aligned[scan]["start"])
                    last_end = float(aligned[scan]["end"])
                    scan += 1
                    k = scan
                    matched_count += 1
                    found = True
                    break
                scan += 1
            if not found:
                continue

        if chunk_start is None or last_end is None:
            continue
        if (matched_count / len(words)) < min_chunk_match_ratio:
            continue

        if prev_end is not None and chunk_start < prev_end:
            chunk_start = prev_end
        if last_end < chunk_start:
            continue

        # Commit global cursor only after successful chunk mapping.
        j = k
        out.append((float(chunk_start), float(last_end), chunk))
        prev_end = float(last_end)
    
    with open("Mapped chunks to script alignment.txt", "w", encoding="utf-8") as f:
        f.write(f"Mapped chunks to script alignment: {out}\n")
    return out


async def refine_segments_async(
    segments: List[Dict[str, Any]],
    word_timestamps: List[Dict[str, Any]],
    cut_marker_generator: Any,
    script_alignment: Optional[List[Dict[str, Any]]] = None,
    min_match: float = 0.8,) -> List[Dict[str, Any]]:
    """Refine script into timestamped subtitle blocks using a marker generator."""
    if not segments:
        return []
    full_text = " ".join((seg.get("text") or "").strip() for seg in segments if (seg.get("text") or "").strip()).strip()
    if not full_text:
        return segments

    total_duration = max(0.0, float(segments[-1].get("end", 0.0)) - float(segments[0].get("start", 0.0)))

    try:
        marked = await cut_marker_generator(
            full_text,
            duration_seconds=total_duration,
            partitions=3,
        )
    except Exception:
        return segments

    chunks = _extract_bracket_blocks(marked)
    if not chunks:
        chunks = _split_marked_text(marked)
    if not chunks:
        return segments

    mapped: List[Tuple[float, float, str]] = []
    if script_alignment:
        mapped = _map_chunks_to_script_alignment(
            chunks, script_alignment, min_match=min_match
        )
    if not mapped:
        mapped = _map_chunks_to_timestamps(
            chunks, word_timestamps, min_match=min_match
        )
    if not mapped:
        return segments

    refined: List[Dict[str, Any]] = []
    for cs, ce, ctext in mapped:
        mid = (cs + ce) / 2.0
        source_seg = next(
            (
                s
                for s in segments
                if float(s.get("start", 0.0)) <= mid <= float(s.get("end", 0.0))
            ),
            None,
        )
        preserved = (
            {k: v for k, v in source_seg.items() if k not in ("text", "start", "end")}
            if source_seg
            else {}
        )
        refined.append(
            {
                **preserved,
                "text": ctext.strip(),
                "start": round(cs, 3),
                "end": round(ce, 3),
            }
        )
    return refined