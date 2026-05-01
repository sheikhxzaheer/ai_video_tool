from __future__ import annotations

import os
import traceback
from typing import Literal, Optional

from llm.prompt import SENTENCE_SEGMENT_PROMPT, SUBTITLE_BLOCK_SYSTEM_PROMPT

Provider = Literal["openai", "anthropic"]

_OPENAI_CLIENT = None
_ANTHROPIC_CLIENT = None

def _get_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError("openai not installed. Run: pip install openai") from e

    _OPENAI_CLIENT = AsyncOpenAI(api_key=api_key)
    return _OPENAI_CLIENT


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


def _log_exception(stage: str, exc: Exception) -> None:
    msg = (
        f"[cut_generator][{stage}] {type(exc).__name__}: {exc}\n"
        f"{traceback.format_exc()}\n"
    )
    with open("cut_generator_error.log", "w", encoding="utf-8") as f:
        f.write(msg)
    print(f"[cut_generator] error logged in cut_generator_error.log ({stage})")


def _anthropic_message_text(resp) -> str:
    parts: list[str] = []
    for block in resp.content:
        if getattr(block, "text", None):
            parts.append(block.text)
    return "".join(parts).strip()


async def insert_markers(
    text: str,
    duration_seconds: Optional[float] = None,
    partitions: int = 2,
    model: str = "gpt-4o",
    provider: Optional[Provider] = None,) -> str:
    """Break script text into bracketed subtitle blocks with an LLM."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    resolved_provider: Provider = provider or (
        "anthropic" if model.lower().startswith("claude") else "openai"
    )
    partitions = max(2, min(3, int(partitions or 2)))
    system = SUBTITLE_BLOCK_SYSTEM_PROMPT

    user = f"Break down the provided script below into subtitles following the rules above. Provide ONLY the final bracketed text.:\n\n{cleaned}"
    if resolved_provider == "anthropic":
        print("Using Anthropic")
        try:
            aclient = _get_anthropic_client()
            resp = await aclient.messages.create(
                model=model,
                max_tokens=20000,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            out = _anthropic_message_text(resp)
            with open("anthropic_response.txt", "w", encoding="utf-8") as f:
                f.write(f"Anthropic response: {out}\n")
        except Exception as e:
            _log_exception("anthropic.messages.create", e)
            raise
    else:
        try:
            client = _get_client()
            print("Using OpenAI")
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            out = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            _log_exception("openai.chat.completions.create", e)
            raise

    out = out.strip("`").strip()
    return out


async def segment_sentences(
    text: str,
    model: str = "gpt-4o",
    provider: Optional[Provider] = None,) -> str:
    """Break raw script into bracketed sentence blocks with an LLM."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    resolved_provider: Provider = provider or (
        "anthropic" if model.lower().startswith("claude") else "openai"
    )
    system = SENTENCE_SEGMENT_PROMPT
    user = (
        "Break down the provided script below into complete sentence blocks following the rules above. "
        "Provide ONLY the final bracketed text.\n\n"
        f"{cleaned}"
    )

    if resolved_provider == "anthropic":
        try:
            aclient = _get_anthropic_client()
            resp = await aclient.messages.create(
                model=model,
                max_tokens=20000,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            out = _anthropic_message_text(resp)
            with open("output.txt", "w", encoding="utf-8") as f:
                f.write(f"Segmented sentences: {out}\n")
        except Exception as e:
            _log_exception("anthropic.messages.create", e)
            raise
    else:
        try:
            client = _get_client()
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            out = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            _log_exception("openai.chat.completions.create", e)
            raise

    return out.strip("`").strip()