# testing/test_runners/setup.py

import os
import sys

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import json
import threading
from shared.silencer import run_silent_initialization
from shared.loading_bar import real_loading_bar, show_spinner

def print_diagnostics():
    try:
        with open("testing/test_results/diagnostics.json", "r") as f:
            diagnostics = json.load(f)
    except Exception:
        print("⚠️ Could not read diagnostic log.")
        return False

    all_passed = True
    for module, info in diagnostics.items():
        status = info.get("status", False)
        time_taken = info.get("load_time")
        error = info.get("error")

        if status:
            print(f"✅ {module} initialized in {time_taken:.3f}s.")
        else:
            print(f"❌ {module} failed: {error}")
            all_passed = False

    return all_passed

def main():
    print("🔍 Running library and model checks...\n")

    modules = [
        "pytesseract", "easyocr_en", "easyocr_ru", "easyocr_ar", "easyocr_ch",
        "paddleocr", "cv2_east", "spellchecker", "corpus_freqs"
    ]

    done_flag = {"complete": False}

    def init_task():
        success = run_silent_initialization()
        done_flag["complete"] = True
        done_flag["success"] = success

    # Start real initialization
    init_thread = threading.Thread(target=init_task)
    init_thread.start()

    # Run real loading bar once
    real_loading_bar(modules)

    # Show spinner while waiting for real init to finish
    show_spinner(done_flag)

    # Wait for thread to join
    init_thread.join()

    # Print final diagnostics
    if done_flag.get("success", False):
        all_passed = print_diagnostics()
        if all_passed:
            print("\n🎉 All components initialized successfully.")
        else:
            print("\n⚠️ One or more components failed. Review errors above.")
    else:
        print("\n❌ Silent initialization failed unexpectedly.")

if __name__ == "__main__":
    main()

def initialize_with_loading(modules):
    done_flag = {"complete": False}

    def init_task():
        success = run_silent_initialization()
        done_flag["complete"] = True
        done_flag["success"] = success

    init_thread = threading.Thread(target=init_task)
    init_thread.start()
    real_loading_bar(modules)
    show_spinner(done_flag)
    init_thread.join()

    return done_flag.get("success", False)
