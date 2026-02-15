# shared/path_utils.py
from pathlib import Path

# Repo root = parent of "shared"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path

