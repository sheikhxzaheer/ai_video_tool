"""Milestone-2: Vector search and footage matching."""

from footage.indexer import build_footage_index
from footage.matcher import match_segments_to_footage

__all__ = ["build_footage_index", "match_segments_to_footage"]
