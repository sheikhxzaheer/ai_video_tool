"""Audio alignment module using stable-ts for word-level timestamps."""

import difflib
from pathlib import Path
from typing import Any, Dict, List

import stable_whisper


def load_model(model_name: str = "base"):
    print(f"Loading Whisper model: {model_name}...")
    model = stable_whisper.load_model(model_name)
    print("Model loaded successfully!")
    return model


def align_audio(audio_path: str, model: Any) -> List[Dict[str, Any]]:
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    result = model.transcribe(audio_path, word_timestamps=True, regroup=False)
    word_timestamps: List[Dict[str, Any]] = []
    for segment in result.segments:
        for word_info in segment.words:
            word_timestamps.append({
                "word": word_info.word.strip(),
                "start": round(word_info.start, 3),
                "end": round(word_info.end, 3),
            })
    return word_timestamps


def _tokenize_script(script: str) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    if not script:
        return tokens
    raw_words = script.split()
    for idx, w in enumerate(raw_words):
        clean = "".join(ch for ch in w if ch.isalnum()).lower()
        if not clean:
            continue
        tokens.append({"index": idx, "word": w, "norm": clean})
    return tokens


def _normalize_asr_words(word_timestamps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for i, item in enumerate(word_timestamps):
        w = item.get("word", "")
        clean = "".join(ch for ch in w if ch.isalnum()).lower()
        normalized.append({
            "index": i,
            "word": w,
            "norm": clean if clean else "",
            "start": item.get("start"),
            "end": item.get("end"),
        })
    return normalized


def align_script_to_transcript(
    word_timestamps: List[Dict[str, Any]],
    script: str,) -> List[Dict[str, Any]]:
    
    script_tokens = _tokenize_script(script)
    if not script_tokens or not word_timestamps:
        return []
    asr_tokens = _normalize_asr_words(word_timestamps)
    script_seq = [t["norm"] for t in script_tokens]
    asr_seq = [t["norm"] for t in asr_tokens]
    matcher = difflib.SequenceMatcher(None, script_seq, asr_seq, autojunk=False)
    alignment: List[Dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            length = min(i2 - i1, j2 - j1)
            for offset in range(length):
                s_tok = script_tokens[i1 + offset]
                a_tok = asr_tokens[j1 + offset]
                alignment.append({
                    "script_index": s_tok["index"],
                    "script_word": s_tok["word"],
                    "asr_word": a_tok["word"],
                    "start": a_tok["start"],
                    "end": a_tok["end"],
                    "matched": True,
                    "match_type": tag,
                })
            for extra in range(length, i2 - i1):
                s_tok = script_tokens[i1 + extra]
                alignment.append({
                    "script_index": s_tok["index"],
                    "script_word": s_tok["word"],
                    "asr_word": None,
                    "start": None,
                    "end": None,
                    "matched": False,
                    "match_type": "delete",
                })
        elif tag == "delete":
            for idx in range(i1, i2):
                s_tok = script_tokens[idx]
                alignment.append({
                    "script_index": s_tok["index"],
                    "script_word": s_tok["word"],
                    "asr_word": None,
                    "start": None,
                    "end": None,
                    "matched": False,
                    "match_type": "delete",
                })
    alignment.sort(key=lambda x: x["script_index"])
    return alignment


def align_audio_with_script(audio_path: str, script: str, model: Any) -> Dict[str, Any]:
    word_timestamps = align_audio(audio_path, model)
    script_alignment = align_script_to_transcript(word_timestamps, script)
    return {"word_timestamps": word_timestamps, "script_alignment": script_alignment}


def get_audio_duration(audio_path: str) -> float:
    import subprocess
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        return 0.0
