# testing/test_runners/ocr_race_runner.py
import os
import sys
import warnings
from threading import Thread, Event
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

from ocr_modules.pipeline_utils.ocr_race import ocr_race_engines
from ocr_modules.base_modules.initialization import initialize_models
from shared.runner_core import load_images
from shared.loading_bar import start_spinner
from shared.json_utils import save_json
from shared.pipeline_summary import run_summary
from shared.text_output import print_ocr_text_outputs
from shared.summary_table import print_ocr_summary
from shared.master_summary_table import print_master_summary

def run_race_on_folder(folder_path, models):
    results = {}
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        print(f"\n🖼️ Image: {filename}")

        try:
            cv_img, pil_img = load_images(image_path)
            if cv_img is None or pil_img is None:
                raise ValueError("Image loading failed — cv_img or pil_img is None")

            # ⏱️ Start external timer
            start = time.perf_counter()
            # NOTE: east_result is optional; we don’t pass it here so runner behavior is unchanged
            results_log = ocr_race_engines(cv_img, pil_img, models, timeout=5.0)
            elapsed = round(time.perf_counter() - start, 3)
            print(f"⏱️ Total OCR race time: {elapsed}s")

            # ⏱️ Per-engine runtimes
            for engine in ["tesseract", "easyocr", "paddleocr"]:
                entry = results_log["all_outputs"].get(engine, {})
                if "runtime" in entry:
                    print(f"  ⏱️ {engine.capitalize()} runtime: {entry['runtime']}s")

            print_ocr_text_outputs(results_log["all_outputs"])
            print_ocr_summary(results_log["all_outputs"])
            results[filename] = results_log
            time.sleep(1.5) # Delay to prevent crash
        except Exception as e:
            print(f"❌ Failed to process image: {e}")
            results[filename] = {"error": str(e)}

    return results

def run_race_batch(base_folder="testing/test_images/benchmark_images"):
    print(f"\n🏁 Running OCR race on folder: {base_folder}\n")

    stop_event = Event()
    models = {}

    def preload_task():
        nonlocal models
        models = initialize_models()
        stop_event.set()

    spinner_thread = start_spinner("🧠 Warming up OCR engines", stop_event)
    loader_thread = Thread(target=preload_task)
    loader_thread.start()
    loader_thread.join()
    spinner_thread.join()

    print("✅ Models ready. Starting OCR race...\n")

    all_results = {}
    categories_processed = 0

    for category in sorted(os.listdir(base_folder)):
        category_path = os.path.join(base_folder, category)
        if not os.path.isdir(category_path):
            continue

        print(f"\n📂 Category: {category}")
        try:
            category_results = run_race_on_folder(category_path, models)
            all_results[category] = category_results
            categories_processed += 1
        except Exception as e:
            print(f"❌ Failed to process category '{category}': {e}")
            all_results[category] = {"error": str(e)}

    output_path = "testing/test_results/ocr_race_outputs.json"
    save_json(all_results, output_path)
    print(f"\n📦 Results saved to {output_path}")
    print(f"✅ Finished OCR race on {categories_processed} categories.")

    try:
        run_summary(json_path=output_path)
        print_master_summary(all_results)
    except Exception as e:
        print(f"⚠️ Failed to generate summary: {e}")


if __name__ == "__main__":
    run_race_batch()
