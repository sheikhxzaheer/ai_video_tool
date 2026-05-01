import os
import sys
from pathlib import Path


def count_files(root_folder: Path) -> int:
    if not root_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {root_folder}")
    if not root_folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_folder}")

    total = 0
    for _, _, filenames in os.walk(root_folder):
        total += len(filenames)
    return total


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_files.py <folder_path>")
        sys.exit(1)

    folder = Path(sys.argv[1]).expanduser().resolve()
    try:
        total = count_files(folder)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(2)

    print(f"Total files (recursively) under '{folder}': {total}")

