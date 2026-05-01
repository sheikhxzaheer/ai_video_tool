"""CLI script to build footage index. Run from cmd - no UI.

Usage:
    python run_index.py
    python run_index.py --footage "final-database,Database"
    python run_index.py --batch-size 10 --force
"""

import argparse
from pathlib import Path

from footage.indexer import build_footage_index


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB footage index (Gemini video analysis)")
    parser.add_argument(
        "--footage",
        default="final-database,Dataset/Brands",
        help="Comma-separated footage paths (default: final-database,Dataset/Brands)",
    )
    parser.add_argument(
        "--chroma",
        default="chroma_db",
        help="ChromaDB path (default: chroma_db)",
    )
    parser.add_argument(
        "--output",
        default="gemini_output",
        help="Directory for Gemini JSON outputs (default: gemini_output)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Concurrent requests per batch (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all (disable skip/resume)",
    )
    args = parser.parse_args()

    try:
        n = build_footage_index(
            footage_root=args.footage,
            chroma_path=args.chroma,
            output_dir=args.output,
            batch_size=args.batch_size,
            skip_existing=not args.force,
        )
        print(f"\nDone. Indexed {n} clips.")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    exit(main())
