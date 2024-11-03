from pathlib import Path


def get_base_dir() -> Path:
    path_utils_dir = Path(__file__).resolve()
    base_dir = path_utils_dir.parent.parent
    return base_dir
