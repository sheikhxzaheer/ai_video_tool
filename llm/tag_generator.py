"""LLM-based semantic tag generation for segments."""

import asyncio
import json
import os
import re
from typing import List, Dict, Optional

from llm.prompt import PREDEFINED_TAGS, TAGGING_SYSTEM_PROMPT

_GEMINI_MODEL = None
_ANTHROPIC_CLIENT = None

def _get_allowed_tags() -> List[str]:
    allowed_tags: List[str] = []
    try:
        for category, items in PREDEFINED_TAGS.items():
            if category in {"Science / Animation", "Fail/Metaphor"}:
                continue
            allowed_tags.extend(items)
    except Exception as e:
        print(f"Warning: could not load predefined tags: {e}")
        return []
    return allowed_tags


def _get_gemini_model():
    global _GEMINI_MODEL
    if _GEMINI_MODEL is not None:
        return _GEMINI_MODEL
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        _GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
        return _GEMINI_MODEL
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")


def _get_anthropic_client():
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is not None:
        return _ANTHROPIC_CLIENT
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    try:
        from anthropic import AsyncAnthropic
    except ImportError as e:
        raise ImportError("anthropic not installed. Run: pip install anthropic") from e
    _ANTHROPIC_CLIENT = AsyncAnthropic(api_key=api_key)
    return _ANTHROPIC_CLIENT


def _extract_json_array(text: str) -> List[str]:
    if not text:
        return []
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        return []
    return []


def extract_keywords(text: str) -> List[str]:
    """Fallback keyword extraction without LLM."""
    words = text.lower().split()
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "this", "that", "these",
        "those", "it", "its",
    }

    keywords = [
        word.strip(".,!?;:")
        for word in words
        if word.strip(".,!?;:") not in stop_words and len(word) > 2
    ]

    unique_keywords = []
    for kw in keywords:
        if kw not in unique_keywords:
            unique_keywords.append(kw)
        if len(unique_keywords) >= 5:
            break

    return unique_keywords if unique_keywords else ["video", "content"]


async def generate_tags_with_gemini(text: str, allowed_tags: Optional[List[str]] = None) -> List[str]:
    """Generate tags using Gemini, constrained to allowed tags."""
    current_allowed = allowed_tags or _get_allowed_tags()
    if not current_allowed:
        return []

    allowed_str = ", ".join(current_allowed)
    prompt = (
        "You are a video editing assistant.\n"
        "Given the following segment text, choose 3-5 tags that best describe it.\n\n"
        "VERY IMPORTANT:\n"
        f"- You MUST choose tags ONLY from this fixed list (no new tags): {allowed_str}\n"
        "- Return ONLY the chosen tag names as a comma-separated list.\n"
        "- Do NOT invent new tags.\n\n"
        f'Text: "{text}"\n\n'
        "Tags (comma-separated, only from the list above):"
    )

    try:
        model = _get_gemini_model()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
        tags_text = (response.text or "").strip()
        raw_tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
        allowed_lower = {t.lower(): t for t in current_allowed}
        filtered: List[str] = []
        for t in raw_tags:
            key = t.lower()
            if key in allowed_lower and allowed_lower[key] not in filtered:
                filtered.append(allowed_lower[key])
            if len(filtered) >= 5:
                break
        return filtered
    except Exception as e:
        print(f"Error generating tags with Gemini: {e}")
        return []


async def generate_additional_tags_with_claude(
    text: str,
    model: str = "claude-haiku-4-5",) -> List[str]:
    """Generate free-form visual tags (3-6) from custom Claude prompt."""
    try:
        client = _get_anthropic_client()
        response = await client.messages.create(
            model=model,
            max_tokens=400,
            temperature=0.0,
            system=TAGGING_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Subtitle block: {text}",
                }
            ],
        )
        output_text = ""
        for block in response.content:
            if getattr(block, "text", None):
                output_text += block.text
        tags = _extract_json_array(output_text)
        deduped: List[str] = []
        for tag in tags:
            cleaned = " ".join(tag.split()).strip()
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
            if len(deduped) >= 6:
                break
        return deduped
    except Exception as e:
        print(f"Error generating additional tags with Claude: {e}")
        return []


async def generate_tags_for_segment(
    segment: Dict[str, any],
    provider: str = "gemini",
    allowed_tags: Optional[List[str]] = None,) -> Dict[str, any]:
    """Generate predefined tags + additional Claude visual tags for a segment."""
    if provider == "gemini":
        tags = await generate_tags_with_gemini(segment["text"], allowed_tags=allowed_tags)
        if not tags:
            tags = extract_keywords(segment["text"])
    else:
        tags = extract_keywords(segment["text"])
    additional_tags = await generate_additional_tags_with_claude(segment["text"])
    return {
        **segment,
        "tags": tags,
        "additional_tags": additional_tags,
    }


async def generate_tags_async(
    segments: List[Dict[str, any]],
    batch_size: int = 5,
    provider: str = "gemini",) -> List[Dict[str, any]]:
    """Generate tags for all segments asynchronously with batching."""
    results = []
    allowed_tags = _get_allowed_tags()
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        tasks = [
            generate_tags_for_segment(seg, provider=provider, allowed_tags=allowed_tags)
            for seg in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    return results


def generate_tags_sync(
    segments: List[Dict[str, any]],
    batch_size: int = 5,
    provider: str = "gemini",) -> List[Dict[str, any]]:
    """Synchronous wrapper for tag generation."""
    return asyncio.run(generate_tags_async(segments, batch_size=batch_size, provider=provider))