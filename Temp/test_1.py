import os
import sys
from pathlib import Path

def delete_dot_underscore_files(root_folder: Path) -> None:
    """
    Recursively delete files whose names start with '._' under root_folder.
    Prints what is deleted and what is skipped.
    """
    if not root_folder.exists():
        print(f"[ERROR] Folder does not exist: {root_folder}")
        return
    if not root_folder.is_dir():
        print(f"[ERROR] Not a directory: {root_folder}")
        return

    deleted = []
    skipped = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        dirpath = Path(dirpath)
        for name in filenames:
            if name.startswith("._"):
                file_path = dirpath / name
                try:
                    os.remove(file_path)
                    deleted.append(file_path)
                    print(f"[DELETED] {file_path}")
                except Exception as e:
                    skipped.append((file_path, str(e)))
                    print(f"[SKIPPED] {file_path} (error: {e})")
            else:
                skipped.append((dirpath / name, "name does not start with '._'"))

    print("\nSummary:")
    print(f"  Deleted files: {len(deleted)}")
    print(f"  Non-matching / failed files: {len(skipped)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_dot_underscore.py <folder_path>")
        sys.exit(1)

    folder = Path(sys.argv[1]).expanduser().resolve()
    delete_dot_underscore_files(folder)