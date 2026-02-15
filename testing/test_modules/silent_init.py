# testing/test_modules/silent_init.py

import sys
import os
from shared.path_utils import PROJECT_ROOT

# Ensure project root is in sys.path
sys.path.insert(0, str(PROJECT_ROOT))

import json
from ocr_modules.base_modules.initialization import initialize_models

def log_callback(name, status, load_time, error):
    try:
        if status:
            msg = f"[OK] {name} initialized in {load_time:.3f}s."
        else:
            msg = f"[FAIL] {name} failed: {error}"

        # Safe print without flush
        print(msg)
        if sys.stdout and sys.stdout.isatty():
            sys.stdout.flush()
    except Exception:
        pass  # Silently ignore any print/flush errors


result = initialize_models(callback=log_callback)

# Write diagnostics to file
diagnostics = result.get("diagnostics", {})
os.makedirs("testing/test_results", exist_ok=True)
with open("testing/test_results/diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
