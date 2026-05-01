"""Processing module for segment building."""

from .segment_builder import (
    build_segments,
    build_segments_from_script,
    merge_short_segments,
    new_segment_builder_config,
)

__all__ = [
    "build_segments",
    "build_segments_from_script",
    "merge_short_segments",
    "new_segment_builder_config",
]
