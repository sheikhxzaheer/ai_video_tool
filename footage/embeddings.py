"""Shared embedding logic for footage indexing and matching."""

import os
from typing import List

from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-large"
_OPENAI_CLIENT = None


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def embed_text(text: str) -> List[float]:
    """Generate embedding for text using OpenAI text-embedding-3-large."""
    client = _get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


EMBED_BATCH_SIZE = 100  # Max texts per API call (keeps under 300K token limit)


def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts. Sends in batches of 100 to stay under API limits."""
    if not texts:
        return []
    client = _get_openai_client()
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        chunk = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunk,
        )
        data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in data])
    return all_embeddings
