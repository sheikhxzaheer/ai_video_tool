"""Build ChromaDB index from footage library using Gemini video analysis (upload + semantic core extraction)."""

import hashlib
import json
import math
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BATCH_SIZE = 1  # Concurrent Gemini requests per batch

import chromadb

from footage.embeddings import embed_texts_batch
from llm.prompt import PREDEFINED_TAGS, SEMANTIC_CORE_EXTRACTION_PROMPT

# Video extensions to index
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
# Skip macOS resource fork files
SKIP_PREFIXES = ("._",)

DEFAULT_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
        print(f"Warning: Could not get duration for {video_path}: {e}")
        return 0.0


def _parse_timecode_to_seconds(t: Any) -> Optional[float]:
    """
    Parse timecodes like "MM:SS.MS" or "HH:MM:SS.MS" into seconds.
    Returns None if parsing fails.
    """
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            mm = int(parts[0])
            ss = float(parts[1])
            return mm * 60.0 + ss
        if len(parts) == 3:
            hh = int(parts[0])
            mm = int(parts[1])
            ss = float(parts[2])
            return hh * 3600.0 + mm * 60.0 + ss
    except Exception:
        return None
    return None


def collect_video_files(roots: List[Path]) -> List[Path]:
    """Recursively collect video files from root folder(s). Deduplicates by resolved path."""
    files: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if p.name.startswith(SKIP_PREFIXES):
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(p)
    return files


def _safe_json_filename(video_path: Path, roots: List[Path]) -> str:
    """Generate a safe unique filename for the JSON output."""
    resolved = video_path.resolve()
    for r in roots:
        try:
            rel = resolved.relative_to(r.resolve())
            safe = "__".join(rel.parts).replace(" ", "_").replace(".", "_")
            return f"{safe}.json"
        except ValueError:
            continue
    h = hashlib.md5(str(resolved).encode()).hexdigest()[:8]
    return f"{video_path.stem}_{h}.json"


def _path_relative_to_brand(path: Path, root: Path) -> Optional[str]:
    """Get brand/folder name from path."""
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 1:
            return parts[0]
    except ValueError:
        pass
    return None


def _extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from Gemini response (handles ```json blocks)."""
    if not text or not text.strip():
        return None
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return None


def analyze_video_with_gemini(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Upload video to Google, run Gemini semantic core extraction, return JSON.
    """
    try:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)

        tags_str = ", ".join(
            tag for category in PREDEFINED_TAGS.values() for tag in category
        )
        prompt = SEMANTIC_CORE_EXTRACTION_PROMPT.format(prerequisite_tags=tags_str)

        video_file = genai.upload_file(path=video_path)
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name != "ACTIVE":
            print(f"Warning: Video {video_path} failed to process (state: {video_file.state.name})")
            return None

        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content([video_file, prompt])
        parsed = _extract_json_from_response(response.text or "")
        return parsed
    except Exception as e:
        print(f"Error analyzing {video_path}: {e}")
        return None


def _build_full_text(analysis: Dict[str, Any]) -> str:
    """Build embeddable text from analysis for vector search."""
    parts: List[str] = []
    desc = analysis.get("detailed_description") or ""
    if desc:
        parts.append(desc)
    segments = analysis.get("segments") or []
    for seg in segments:
        struct = seg.get("structural_tags") or []
        visual = seg.get("visual_keywords") or []
        parts.append(" ".join(struct))
        parts.append(" ".join(visual))
    return " ".join(p for p in parts if p).strip()


def _build_segment_text(
    *,
    detailed_description: str,
    seg: Dict[str, Any],
) -> str:
    parts: List[str] = []
    if detailed_description:
        parts.append(detailed_description)
    struct = seg.get("structural_tags") or []
    visual = seg.get("visual_keywords") or []
    why_kept = (seg.get("why_kept") or "").strip()
    role = (seg.get("role") or "").strip()
    attrs = seg.get("attributes") or {}
    shot_type = (attrs.get("shot_type") or "").strip() if isinstance(attrs, dict) else ""
    environment = attrs.get("environment") if isinstance(attrs, dict) else None
    mood = attrs.get("mood") if isinstance(attrs, dict) else None
    style_keywords = attrs.get("style_keywords") if isinstance(attrs, dict) else None
    if struct:
        parts.append(" ".join(str(s) for s in struct if s))
    if visual:
        parts.append(" ".join(str(v) for v in visual if v))
    if role:
        parts.append(f"ROLE {role}")
    if shot_type:
        parts.append(f"SHOT {shot_type}")
    if environment:
        if isinstance(environment, list):
            parts.append(" ".join(str(e) for e in environment if e))
        else:
            parts.append(str(environment))
    if mood:
        if isinstance(mood, list):
            parts.append(" ".join(str(m) for m in mood if m))
        else:
            parts.append(str(mood))
    if style_keywords:
        if isinstance(style_keywords, list):
            parts.append(" ".join(str(k) for k in style_keywords if k))
        else:
            parts.append(str(style_keywords))
    if why_kept:
        parts.append(why_kept)
    return " ".join(p for p in parts if p).strip()


def _stable_segment_id(video_path: str, seg_start: Optional[float], seg_end: Optional[float], seg_idx: int) -> str:
    base = f"{video_path}::{seg_start if seg_start is not None else 'na'}::{seg_end if seg_end is not None else 'na'}::{seg_idx}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()
    return f"seg_{h}"


def _analysis_to_segment_docs(
    *,
    video_path: str,
    filename: str,
    brand: str,
    video_duration: float,
    analysis: Dict[str, Any],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Convert Gemini analysis into segment-level docs.

    Returns tuples: (id, document_text, metadata_dict)
    """
    detailed_description = (analysis.get("detailed_description") or "").strip()
    segments = analysis.get("segments") or []
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for i, seg in enumerate(segments):
        start_tc = seg.get("start_time")
        end_tc = seg.get("end_time")
        start_s = _parse_timecode_to_seconds(start_tc)
        end_s = _parse_timecode_to_seconds(end_tc)
        seg_duration = None
        if start_s is not None and end_s is not None and end_s >= start_s:
            seg_duration = float(end_s - start_s)

        doc_text = _build_segment_text(detailed_description=detailed_description, seg=seg)
        seg_id = _stable_segment_id(video_path, start_s, end_s, i)

        structural_tags = seg.get("structural_tags") or []
        visual_keywords = seg.get("visual_keywords") or []
        role = (seg.get("role") or "").strip().upper()
        attrs = seg.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        shot_type = (attrs.get("shot_type") or "").strip().lower()
        environment = attrs.get("environment") or []
        mood = attrs.get("mood") or []
        style_keywords = attrs.get("style_keywords") or []
        people_present = attrs.get("people_present")
        product_present = attrs.get("product_present")

        metadata: Dict[str, Any] = {
            # Identity / grouping
            "video_path": video_path,
            "path": video_path,  # backward compatibility for older code that expects m["path"]
            "filename": filename,
            "brand": brand,
            # Video-level info
            "video_duration": float(video_duration or 0.0),
            # Segment-level info
            "segment_index": int(i),
            "segment_start": float(start_s) if start_s is not None else None,
            "segment_end": float(end_s) if end_s is not None else None,
            "segment_duration": float(seg_duration) if seg_duration is not None else None,
            # Tags as queryable strings (Chroma metadata filters prefer primitives)
            "structural_tags": "|".join(str(t) for t in structural_tags if t),
            "visual_keywords": "|".join(str(k) for k in visual_keywords if k),
            # Anchor/B-roll and attributes
            "role": role if role in {"ANCHOR", "BROLL"} else "",
            "shot_type": shot_type,
            "environment": "|".join(str(e) for e in environment if e) if isinstance(environment, list) else str(environment or ""),
            "mood": "|".join(str(m) for m in mood if m) if isinstance(mood, list) else str(mood or ""),
            "style_keywords": "|".join(str(s) for s in style_keywords if s) if isinstance(style_keywords, list) else str(style_keywords or ""),
            "people_present": bool(people_present) if people_present is not None else None,
            "product_present": bool(product_present) if product_present is not None else None,
            # Raw context (small-ish)
            "why_kept": (seg.get("why_kept") or "").strip(),
            "detailed_description": detailed_description,
        }
        clean_metadata: Dict[str, Any] = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, float) and not math.isfinite(v):
                continue
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)
        out.append((seg_id, doc_text, clean_metadata))
    return out


def _load_existing_json(video_path: Path, roots: List[Path], output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load existing JSON if it exists and path matches. Returns analysis dict or None."""
    json_path = output_dir / _safe_json_filename(video_path, roots)
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        path_in_file = data.get("path", "")
        if path_in_file != str(video_path.resolve()):
            return None
        root = roots[0] if roots else video_path.parent
        return data
    except (json.JSONDecodeError, OSError):
        return None


def process_single_video(
    video_path: Path,
    roots: List[Path],
    output_dir: Optional[Path] = None,
    skip_existing: bool = True,
) -> Optional[Dict[str, Any]]:
    """Analyze video with Gemini, return analysis dict. Optionally save Gemini output to JSON.
    If skip_existing=True and JSON already exists, load from file instead of calling Gemini."""
    path_str = str(video_path.resolve())
    root = roots[0] if roots else video_path.parent

    if skip_existing and output_dir:
        out_path = Path(output_dir)
        existing_analysis = _load_existing_json(video_path, roots, out_path)
        if existing_analysis:
            return existing_analysis

    analysis = analyze_video_with_gemini(path_str)
    if not analysis:
        return None

    duration = get_video_duration(path_str)
    brand = _path_relative_to_brand(video_path, root) or "Unknown"
    full_text = _build_full_text(analysis)

    full_metadata = {
        "path": path_str,
        "detailed_description": analysis.get("detailed_description", ""),
        "segments": analysis.get("segments", []),
    }

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        json_path = Path(output_dir) / _safe_json_filename(video_path, roots)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)

    return full_metadata


def build_footage_index(
    footage_root: Optional[str] = None,
    chroma_path: Optional[str] = None,
    collection_name: str = "footage",
    output_dir: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    skip_existing: bool = True,) -> int:
    """
    Build or rebuild ChromaDB index from footage folder(s).
    Uses Gemini video upload + semantic core extraction (no keyframes).
    Processes videos in parallel batches (default 50 concurrent requests).
    Saves each video's Gemini output to a JSON file (with full path).

    Args:
        footage_root: Path(s) to footage folder(s). Comma-separated for multiple (e.g. "final-database,Dataset/Brands")
        chroma_path: Path for ChromaDB persistence (default: ./chroma_db)
        collection_name: ChromaDB collection name
        output_dir: Directory to save Gemini JSON outputs (default: gemini_output). None to skip.
        batch_size: Concurrent Gemini requests per batch (default: 50).
        skip_existing: If True, skip videos that already have JSON in output_dir (resume on rerun).

    Returns:
        Number of clips indexed
    """
    raw = (footage_root or "final-database").strip()
    roots = [Path(p.strip()) for p in raw.split(",") if p.strip()]
    existing = [r for r in roots if r.exists()]
    if not existing:
        raise FileNotFoundError(f"Footage root(s) not found: {roots}")

    chroma_dir = chroma_path or "chroma_db"
    os.makedirs(chroma_dir, exist_ok=True)

    all_files = collect_video_files(existing)
    if not all_files:
        print("No video files found.")
        return 0

    total = len(all_files)
    total_batches = (total + batch_size - 1) // batch_size
    skip_msg = " (resume: skip existing JSON)" if skip_existing else " (force: reprocess all)"
    print(f"\nTotal: {total} videos | Batch size: {batch_size} | Batches: {total_batches}{skip_msg}\n")

    out_path = Path(output_dir) if output_dir else Path("gemini_output")

    analyses: List[Dict[str, Any]] = []
    video_infos: List[Tuple[Path, float, str]] = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for batch_start in range(0, total, batch_size):
            batch = all_files[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            processed_so_far = batch_start
            pending = total - batch_start

            analyses_before_batch = len(analyses)
            print(f"Batch {batch_num}/{total_batches} | Firing {len(batch)} requests | Processed: {processed_so_far} | Pending: {pending}")

            futures = {
                executor.submit(process_single_video, vp, existing, out_path, skip_existing): vp
                for vp in batch
            }
            for future in as_completed(futures):
                vp = futures[future]
                try:
                    analysis = future.result()
                    if analysis:
                        analyses.append(analysis)
                        root = existing[0] if existing else vp.parent
                        duration = get_video_duration(str(vp.resolve()))
                        brand = _path_relative_to_brand(vp, root) or "Unknown"
                        video_infos.append((vp, duration, brand))
                except Exception as e:
                    print(f"  Failed {vp.name}: {e}")

            completed = len(analyses)
            succeeded_this_batch = completed - analyses_before_batch
            failed_this_batch = len(batch) - succeeded_this_batch
            pending = total - batch_start - len(batch)
            print(f"  Done. Completed: {completed} | Pending: {pending}" + (f" | Failed this batch: {failed_this_batch}" if failed_this_batch else "") + "\n")

    if not analyses:
        print("No clips could be processed.")
        return 0

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name, metadata=DEFAULT_COLLECTION_METADATA)

    # Flatten analysis -> segment docs
    seg_ids: List[str] = []
    seg_docs: List[str] = []
    seg_metadatas: List[Dict[str, Any]] = []
    for (vp, duration, brand), analysis in zip(video_infos, analyses):
        video_path = str(vp.resolve())
        filename = vp.name
        seg_tuples = _analysis_to_segment_docs(
            video_path=video_path,
            filename=filename,
            brand=brand,
            video_duration=duration,
            analysis=analysis,
        )
        for sid, doc_text, md in seg_tuples:
            if not doc_text:
                continue
            seg_ids.append(sid)
            seg_docs.append(doc_text)
            seg_metadatas.append(md)

    if not seg_docs:
        print("No segment-level docs to index.")
        return 0

    embeddings = embed_texts_batch(seg_docs)

    # Chroma has an internal max batch size; write in chunks to avoid InternalError.
    # Use a conservative size to work across different Chroma builds.
    CHROMA_WRITE_BATCH_SIZE = 5000
    n_total = len(seg_ids)
    for start in range(0, n_total, CHROMA_WRITE_BATCH_SIZE):
        end = min(start + CHROMA_WRITE_BATCH_SIZE, n_total)
        ids_chunk = seg_ids[start:end]
        emb_chunk = embeddings[start:end]
        docs_chunk = seg_docs[start:end]
        mds_chunk = seg_metadatas[start:end]

        # Prefer upsert for incremental updates; fall back to add (may error on duplicates).
        try:
            collection.upsert(
                ids=ids_chunk,
                embeddings=emb_chunk,
                documents=docs_chunk,
                metadatas=mds_chunk,
            )
        except Exception:
            collection.add(
                ids=ids_chunk,
                embeddings=emb_chunk,
                documents=docs_chunk,
                metadatas=mds_chunk,
            )

    print(f"Indexed {len(seg_docs)} segment docs into ChromaDB.")
    return len(seg_docs)