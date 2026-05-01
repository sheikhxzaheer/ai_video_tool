"""
Segment builder module for grouping words into meaningful segments.
"""

from typing import Any, Dict, List, Optional, Tuple

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = False
    return _NLP


def new_segment_builder_config(
    min_duration: float = 1.5,
    max_duration: float = 6.0,
    target_duration: float = 5.0,
    split_line_threshold: float = 6.0,) -> Dict[str, float]:
    
    return {
        "min_duration": min_duration,
        "max_duration": max_duration,
        "target_duration": target_duration,
        "split_line_threshold": split_line_threshold,
    }


def build_segments(word_timestamps: List[Dict[str, Any]], config: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    if not word_timestamps:
        return []
    cfg = config or new_segment_builder_config()
    nlp = _get_nlp()
    if nlp is False:
        return _build_segments_fallback(word_timestamps, cfg)
    return _build_segments_spacy(word_timestamps, nlp, cfg)


def build_segments_from_script(
    script: str,
    script_alignment: List[Dict[str, Any]],
    config: Optional[Dict[str, float]] = None,) -> List[Dict[str, Any]]:

    cfg = config or new_segment_builder_config()
    if not script or not script_alignment:
        return []

    section_markers = {"hook", "lead", "body"}
    current_section: Optional[str] = None
    script_index_to_section: Dict[int, Optional[str]] = {}
    for item in script_alignment:
        word = (item.get("script_word") or "").strip()
        idx = item.get("script_index")
        if idx is None:
            continue
        if word.lower() in section_markers:
            current_section = word.capitalize()
            continue
        script_index_to_section[idx] = current_section

    lines = [ln.strip() for ln in script.strip().split("\n") if ln.strip()]
    if not lines:
        return []

    word_index = 0
    line_ranges: List[Tuple[str, int, int]] = []
    for line in lines:
        n = len(line.split())
        if n > 0:
            line_ranges.append((line, word_index, word_index + n))
            word_index += n

    segments: List[Dict[str, Any]] = []
    for line_idx, (line_text, start_idx, end_idx) in enumerate(line_ranges):
        if line_text.strip().lower() in section_markers:
            continue
        line_sections = [
            script_index_to_section.get(i)
            for i in range(start_idx, end_idx)
            if script_index_to_section.get(i) is not None
        ]
        line_section = line_sections[0] if line_sections else None

        aligned = [
            a for a in script_alignment
            if start_idx <= a["script_index"] < end_idx and a.get("matched")
        ]
        if not aligned:
            continue
        starts = [a["start"] for a in aligned if a.get("start") is not None]
        ends = [a["end"] for a in aligned if a.get("end") is not None]
        if not starts or not ends:
            continue
        seg_start, seg_end = min(starts), max(ends)
        duration = seg_end - seg_start

        if duration <= cfg["split_line_threshold"]:
            segments.append({
                "text": line_text,
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "section": line_section,
                "_line_idx": line_idx,
            })
        else:
            subsegs = _split_line_by_meaning(line_text, aligned, seg_start, seg_end, line_section)
            for s in subsegs:
                s["_line_idx"] = line_idx
            segments.extend(subsegs)

    merged = _merge_very_short_script(segments, cfg)
    for s in merged:
        s.pop("_line_idx", None)
    return merged


def _chunk_duration(aligned: List[Dict[str, Any]], lo: int, hi: int) -> float:
    chunk = aligned[lo:hi]
    starts = [a["start"] for a in chunk if a.get("start") is not None]
    ends = [a["end"] for a in chunk if a.get("end") is not None]
    return (max(ends) - min(starts)) if starts and ends else 0.0


def _split_line_by_meaning(
    line_text: str,
    aligned: List[Dict[str, Any]],
    seg_start: float,
    seg_end: float,
    line_section: Optional[str] = None,) -> List[Dict[str, Any]]:

    n = len(aligned)
    if n == 0:
        return []
    if n == 1:
        return [{"text": line_text, "start": round(seg_start, 3), "end": round(seg_end, 3), "section": line_section}]

    min_chunk_dur = 1.0
    split_strong: List[int] = []
    split_weak: List[int] = []
    nlp = _get_nlp()
    if nlp is not False:
        doc = nlp(line_text)
        for token in doc:
            is_punct = token.text.strip() in (",", ";", ":")
            is_conj = token.pos_ in ("CCONJ", "SCONJ")
            if is_punct or is_conj:
                char_end = token.idx + len(token.text)
                word_count = len(line_text[:char_end].split())
                if 0 < word_count < n:
                    left_dur = _chunk_duration(aligned, 0, word_count)
                    right_dur = _chunk_duration(aligned, word_count, n)
                    if left_dur >= min_chunk_dur and right_dur >= min_chunk_dur:
                        if is_punct:
                            split_strong.append(word_count)
                        else:
                            split_weak.append(word_count)

    split_after = sorted(set(split_strong + split_weak))
    if not split_after:
        mid = max(1, n // 2)
        if _chunk_duration(aligned, 0, mid) >= min_chunk_dur and _chunk_duration(aligned, mid, n) >= min_chunk_dur:
            split_after = [mid]

    boundaries = sorted(set([0] + split_after + [n]))
    result: List[Dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        chunk = aligned[lo:hi]
        starts = [a["start"] for a in chunk if a.get("start") is not None]
        ends = [a["end"] for a in chunk if a.get("end") is not None]
        if not starts or not ends:
            continue
        result.append({
            "text": " ".join(a["script_word"] for a in chunk),
            "start": round(min(starts), 3),
            "end": round(max(ends), 3),
            "section": line_section,
        })
    return result


def _build_char_to_word_map(word_timestamps: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, int]]]:
    words = [w["word"] for w in word_timestamps]
    char_ranges: List[Tuple[int, int]] = []
    pos = 0
    for w in words:
        start = pos
        pos += len(w)
        char_ranges.append((start, pos))
        pos += 1
    return " ".join(words), char_ranges


def _char_pos_to_word_idx(char_pos: int, char_ranges: List[Tuple[int, int]]) -> int:
    for i, (s, e) in enumerate(char_ranges):
        if s <= char_pos < e:
            return i
    for i in range(len(char_ranges) - 1, -1, -1):
        if char_ranges[i][1] <= char_pos:
            return i
    return 0


def _build_segments_spacy(
    word_timestamps: List[Dict[str, Any]],
    nlp: Any,
    config: Dict[str, float],) -> List[Dict[str, Any]]:

    text, char_ranges = _build_char_to_word_map(word_timestamps)
    if not text.strip():
        return []
    doc = nlp(text)
    segments: List[Dict[str, Any]] = []
    for sent in doc.sents:
        start_idx = _char_pos_to_word_idx(sent.start_char, char_ranges)
        end_idx = _char_pos_to_word_idx(sent.end_char - 1, char_ranges)
        end_idx = min(end_idx, len(word_timestamps) - 1)
        start_idx = min(start_idx, end_idx)
        sent_words = word_timestamps[start_idx:end_idx + 1]
        if not sent_words:
            continue
        sent_duration = sent_words[-1]["end"] - sent_words[0]["start"]
        if sent_duration <= config["max_duration"]:
            segments.append({
                "text": " ".join(w["word"] for w in sent_words).strip(),
                "start": round(sent_words[0]["start"], 3),
                "end": round(sent_words[-1]["end"], 3),
            })
        else:
            segments.extend(_split_long_sentence(sent_words, sent.text, sent.start_char, start_idx, nlp, char_ranges, config))
    return _merge_very_short(segments, config)


def _split_long_sentence(
    sent_words: List[Dict[str, Any]],
    sent_text: str,
    sent_start_char: int,
    sent_start_word_idx: int,
    nlp: Any,
    char_ranges: List[Tuple[int, int]],
    config: Dict[str, float],) -> List[Dict[str, Any]]:

    sent_doc = nlp(sent_text)
    split_token_ends = []
    for token in sent_doc:
        if token.text.strip() in (",", ";", ":") or token.pos_ in ("CCONJ", "SCONJ"):
            split_token_ends.append(token.idx + len(token.text))

    valid_splits = set()
    for local_char_end in split_token_ends:
        global_char = sent_start_char + local_char_end
        wi = _char_pos_to_word_idx(global_char, char_ranges)
        local_next = (wi - sent_start_word_idx) + 1
        if 0 < local_next < len(sent_words):
            valid_splits.add(local_next)

    result: List[Dict[str, Any]] = []
    chunk_start = 0
    valid_sorted = sorted(valid_splits)
    while chunk_start < len(sent_words):
        chunk_end = chunk_start
        last_valid_split = chunk_start
        for j in range(chunk_start, len(sent_words)):
            chunk_end = j + 1
            chunk_dur = sent_words[j]["end"] - sent_words[chunk_start]["start"]
            if (j + 1) in valid_sorted:
                last_valid_split = j + 1
            if chunk_dur >= config["max_duration"]:
                break
            if chunk_dur >= config["target_duration"] and last_valid_split > chunk_start:
                chunk_end = last_valid_split
                break
        if chunk_end <= chunk_start:
            chunk_end = chunk_start + 1
        chunk = sent_words[chunk_start:chunk_end]
        result.append({
            "text": " ".join(w["word"] for w in chunk).strip(),
            "start": round(chunk[0]["start"], 3),
            "end": round(chunk[-1]["end"], 3),
        })
        chunk_start = chunk_end
    return result


def _merge_very_short(segments: List[Dict[str, Any]], config: Dict[str, float]) -> List[Dict[str, Any]]:
    if not segments:
        return []
    merged = []
    i = 0
    while i < len(segments):
        cur = segments[i]
        if (cur["end"] - cur["start"]) < config["min_duration"] and i + 1 < len(segments):
            nxt = segments[i + 1]
            merged.append({"text": f"{cur['text']} {nxt['text']}".strip(), "start": cur["start"], "end": nxt["end"]})
            i += 2
        else:
            merged.append(cur)
            i += 1
    return merged


def _merge_very_short_script(segments: List[Dict[str, Any]], config: Dict[str, float]) -> List[Dict[str, Any]]:
    if not segments:
        return []
    merged = []
    i = 0
    while i < len(segments):
        cur = dict(segments[i])
        line_idx = cur.get("_line_idx")
        if (
            (cur["end"] - cur["start"]) < config["min_duration"]
            and i + 1 < len(segments)
            and segments[i + 1].get("_line_idx") == line_idx
        ):
            nxt = segments[i + 1]
            cur["text"] = f"{cur['text']} {nxt['text']}".strip()
            cur["end"] = nxt["end"]
            i += 2
        else:
            i += 1
        merged.append(cur)
    return merged


def _build_segments_fallback(word_timestamps: List[Dict[str, Any]], config: Dict[str, float]) -> List[Dict[str, Any]]:
    segments = []
    current_words = []
    current_start = word_timestamps[0]["start"]
    for i, wd in enumerate(word_timestamps):
        current_words.append(wd["word"])
        current_duration = wd["end"] - current_start
        should_break = False
        word = wd["word"]
        if any(word.endswith(p) for p in (".", "!", "?")):
            should_break = True
        elif current_duration >= config["max_duration"]:
            should_break = True
        elif current_duration >= config["target_duration"]:
            if any(word.endswith(p) for p in (",", ";")):
                should_break = True
            elif i + 1 < len(word_timestamps):
                gap = word_timestamps[i + 1]["start"] - wd["end"]
                if gap > 0.5:
                    should_break = True
        if should_break and current_words:
            segments.append({
                "text": " ".join(current_words).strip(),
                "start": round(current_start, 3),
                "end": round(wd["end"], 3),
            })
            current_words = []
            if i + 1 < len(word_timestamps):
                current_start = word_timestamps[i + 1]["start"]
    if current_words:
        segments.append({
            "text": " ".join(current_words).strip(),
            "start": round(current_start, 3),
            "end": round(word_timestamps[-1]["end"], 3),
        })
    return segments


def merge_short_segments(segments: List[Dict[str, Any]], min_duration: float = 1.5) -> List[Dict[str, Any]]:
    if not segments:
        return []
    merged = []
    i = 0
    while i < len(segments):
        cur = segments[i]
        if (cur["end"] - cur["start"]) < min_duration and i + 1 < len(segments):
            nxt = segments[i + 1]
            merged.append({"text": f"{cur['text']} {nxt['text']}".strip(), "start": cur["start"], "end": nxt["end"]})
            i += 2
        else:
            merged.append(cur)
            i += 1
    return merged
