"""Clear all cache directories for IDIR-KS"""

import shutil
import os
from pathlib import Path


def clear_all_cache():
    """Clear all cache directories"""

    cache_dirs = [
        "./cache",
        "./__pycache__",
        "./.pytest_cache",
        "./.mypy_cache",
        "./.cache",
    ]

    # Add Python cache recursively
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__" or d == ".pytest_cache" or d.endswith(".egg-info"):
                cache_dirs.append(os.path.join(root, d))

    print("Clearing cache directories...")
    cleared_count = 0

    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"  Cleared: {cache_dir}")
                cleared_count += 1
            except Exception as e:
                print(f"  Failed to clear {cache_dir}: {e}")

    print(f"\nCleared {cleared_count} cache directories")

    # Create fresh cache directory
    os.makedirs("./cache", exist_ok=True)
    print("Created fresh ./cache directory")


def clean_python_files():
    """Remove compiled Python files"""
    print("\nRemoving compiled Python files...")

    for root, dirs, files in os.walk("."):
        # Remove __pycache__ directories
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"  Removed: {pycache_path}")
            except Exception as e:
                print(f"  Failed to remove {pycache_path}: {e}")

        # Remove .pyc files
        for file in files:
            if file.endswith((".pyc", ".pyo")):
                filepath = os.path.join(root, file)
                try:
                    os.remove(filepath)
                    print(f"  Removed: {filepath}")
                except Exception as e:
                    print(f"  Failed to remove {filepath}: {e}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")

    print("=" * 60)
    print("IDIR-KS Cache Cleaner")
    print("=" * 60)
    print()

    clear_all_cache()
    clean_python_files()

    print("\n" + "=" * 60)
    print("Cache clearing complete!")
    print("=" * 60)
