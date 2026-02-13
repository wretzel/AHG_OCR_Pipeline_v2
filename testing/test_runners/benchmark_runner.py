# testing/test_runners/benchmark_runner.py

import os
import sys
import cv2
import json
import warnings
from PIL import Image
from threading import Thread, Event

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

from ocr_modules.base_modules.ocr_engines import (
    load_ocr_models,
    run_tesseract,
    run_easyocr,
    run_paddleocr,
    run_east
)
from shared.loading_bar import start_spinner
from shared.json_utils import save_json
from shared.summary_table import print_ocr_summary
from shared.runtime import timed_run
from shared.runner_core import load_images, run_all_ocr_engines
from shared.text_output import print_ocr_text_outputs

def run_benchmark(image_path):
    print(f"\n📈 Benchmarking OCR on: {image_path}\n")

    # Load image
    cv_img, pil_img = load_images(image_path)

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

    print("✅ Models ready. Starting benchmark...\n")

    results_log = run_all_ocr_engines(cv_img, pil_img, models)

    print_ocr_text_outputs(results_log)

    # Summary
    print_ocr_summary(results_log)

    # Save results
    path = "testing/test_results/benchmark_outputs.json"
    save_json(results_log, path)
    print(f"📁 Benchmark results saved to:\n- {path}")

if __name__ == "__main__":
    sample_path = "testing/test_images/sample_text.png"
    run_benchmark(sample_path)
