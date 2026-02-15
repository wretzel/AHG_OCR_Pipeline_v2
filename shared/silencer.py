# ocr_modules/shared/silencer.py

import subprocess
import os

def run_silent_initialization():

    result = subprocess.run(
        ["python", "testing/test_modules/silent_init.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0

