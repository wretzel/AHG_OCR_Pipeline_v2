# testing/test_runners/batch_runner.py

import os
import sys
import warnings
from threading import Thread, Event

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

from ocr_modules.base_modules.ocr_engines import load_ocr_models
from shared.loading_bar import start_spinner
from shared.json_utils import save_json
from shared.runner_core import load_images, run_all_ocr_engines
from shared.text_output import print_ocr_text_outputs
from shared.summary_table import print_ocr_summary

def run_batch(selected_categories=None, folder="testing/test_images/benchmark_images"):
    print(f"\n📁 Running batch OCR on folder: {folder}\n")

    available = ["clear_images", "scene_images", "complex_images", "dummy_images"]

    # Interactive selection if no CLI categories provided
    if selected_categories is None:
        print("📂 Select a category to OCR:")
        print("  1. Clear")
        print("  2. Scene")
        print("  3. Complex")
        print("  4. Dummy")
        print("  5. All")
        choice = input("\nEnter a number (1–5): ").strip()

        options = {
            "1": ["clear_images"],
            "2": ["scene_images"],
            "3": ["complex_images"],
            "4": ["dummy_images"],
            "5": available
        }

        selected = options.get(choice)
        if not selected:
            print("❌ Invalid choice. Aborting.")
            return
    else:
        selected = [f"{cat}_images" for cat in selected_categories if f"{cat}_images" in available]
        if not selected:
            print("⚠️ No valid categories selected. Aborting.")
            return

    # Warm up models
    stop_event = Event()
    models = {}

    def preload_task():
        nonlocal models
        models = load_ocr_models()
        stop_event.set()

    spinner_thread = start_spinner("🧠 Warming up OCR engines", stop_event)
    loader_thread = Thread(target=preload_task)
    loader_thread.start()
    loader_thread.join()
    spinner_thread.join()

    print("✅ Models ready. Starting batch...\n")

    batch_results = {}

    for category in selected:
        category_path = os.path.join(folder, category)
        if not os.path.isdir(category_path):
            continue

        print(f"\n📂 Category: {category}")
        batch_results[category] = {}

        for filename in sorted(os.listdir(category_path)):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(category_path, filename)
            print(f"\n🖼️ Image: {filename}")

            try:
                cv_img, pil_img = load_images(image_path)
                results_log = run_all_ocr_engines(cv_img, pil_img, models)
                print_ocr_text_outputs(results_log)
                print_ocr_summary(results_log)
                batch_results[category][filename] = results_log
            except Exception as e:
                print(f"❌ Failed to process image: {e}")
                batch_results[category][filename] = {"error": str(e)}

    # Save results
    path = "testing/test_results/batch_outputs.json"
    save_json(batch_results, path)
    print(f"\n📁 Batch results saved to:\n- {path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run batch OCR on selected categories.")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["clear", "scene", "complex", "dummy"],
        help="Specify one or more categories to OCR (default: interactive menu)"
    )
    args = parser.parse_args()
    run_batch(selected_categories=args.categories)
