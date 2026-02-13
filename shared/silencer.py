# ocr_modules/shared/silencer.py

import subprocess
import os

def run_silent_initialization():
    """
    Silently runs the initialization pipeline to suppress noisy logs from OCR backends.
    Returns True if successful, False otherwise.
    """
    result = subprocess.run(
        ["python", "testing/test_modules/silent_init.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0

