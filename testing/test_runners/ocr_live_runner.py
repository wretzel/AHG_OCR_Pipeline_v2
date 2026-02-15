# testing/test_runners/ocr_live_runner.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from server_utils.http_server import run_server

if __name__ == "__main__":
    run_server()
