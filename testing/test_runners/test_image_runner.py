# testing/test_runners/test_image_runner.py

import os
import sys
import warnings
from threading import Thread, Event

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

from testing.test_runners.setup import initialize_with_loading
from ocr_modules.base_modules.ocr_engines import load_ocr_models
from shared.loading_bar import start_spinner
from shared.diagnostics import print_module_timings
from shared.json_utils import save_json
from shared.summary_table import print_ocr_summary
from shared.text_output import print_ocr_text_outputs
from shared.runner_core import load_images, run_all_ocr_engines

def run_all_engines(image_path):
    print(f"\n🖼️ Testing OCR on: {image_path}\n")

    # Load image
    cv_img, pil_img = load_images(image_path)

    # Initialize modules
    modules = [
        "pytesseract", "easyocr_en", "easyocr_ru", "easyocr_ar", "easyocr_ch",
        "paddleocr", "cv2_east", "spellchecker", "corpus_freqs"
    ]
    if not initialize_with_loading(modules):
        print("❌ Initialization failed. Aborting OCR test.")
        return

    print_module_timings()

    # Warm up models with spinner
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

    print("✅ Setup complete. Running OCR...\n")

    # Run all engines
    results_log = run_all_ocr_engines(cv_img, pil_img, models)

    # Output text and summary
    print_ocr_text_outputs(results_log)
    print_ocr_summary(results_log)

    # Save results
    path = "testing/test_results/ocr_outputs.json"
    save_json(results_log, path)
    print(f"📁 OCR results saved to:\n- {path}")

if __name__ == "__main__":
    sample_path = "testing/test_images/sample_text.png"
    run_all_engines(sample_path)
