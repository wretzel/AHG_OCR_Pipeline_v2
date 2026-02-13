# shared/diagnostics.py

import json

def print_module_timings(path="testing/test_results/diagnostics.json"):
    try:
        with open(path, "r") as f:
            diagnostics = json.load(f)
    except Exception:
        print("⚠️ Could not read diagnostic log.")
        return

    print("\n📊 **Module Load Times**")
    for module, info in diagnostics.items():
        status = info.get("status", False)
        time_taken = info.get("load_time")
        error = info.get("error")

        if status:
            print(f"- {module:<15}: {time_taken:.2f} sec")
        else:
            print(f"- {module:<15}: ❌ {error}")
