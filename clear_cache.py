import os
import shutil

CACHE_TARGETS = [
    ".ruff_cache",
    "training.log",
    "training_error.log",
]


def remove_path(path):
    if not os.path.exists(path):
        return False
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        os.remove(path)
    return True


def prune_pycache(root_dir):
    removed = []
    for current_path, dirs, _ in os.walk(root_dir):
        if "__pycache__" in dirs:
            cache_path = os.path.join(current_path, "__pycache__")
            shutil.rmtree(cache_path, ignore_errors=True)
            removed.append(cache_path)
    return removed


def main():
    removed_any = False

    for target in CACHE_TARGETS:
        if remove_path(target):
            print(f"Removed cache target: {target}")
            removed_any = True

    erased_pycache = prune_pycache('.')
    for path in erased_pycache:
        print(f"Removed __pycache__: {path}")
        removed_any = True

    if not removed_any:
        print("No caches found to remove.")


if __name__ == "__main__":
    main()
