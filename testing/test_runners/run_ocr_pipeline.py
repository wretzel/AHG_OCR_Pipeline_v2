# testing/test_runners/run_ocr_pipeline.py

import os
import sys
import concurrent.futures
import numpy as np

# Project root setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# Imports
from shared.runner_core import load_images
from ocr_modules.base_modules.initialization import initialize_models
from ocr_modules.base_modules.ocr_engines import run_east
from ocr_modules.pipeline_utils.pipeline import run_pipeline, print_pipeline_log

def load_ocr_models(force_reload=False):
    """Initialize and cache OCR models."""
    global _models
    if "_models" not in globals() or _models is None or force_reload:
        _models = initialize_models()
    return _models

if __name__ == "__main__":
    models = load_ocr_models()
    # Warm-up EAST once
    _ = run_east(np.zeros((320, 320, 3), dtype=np.uint8))

    folders = [
        "clear_images",
        "scene_images",
        "complex_images",
        "dummy_images"
    ]

    # Choose mode: "fast", "steady", or "extended"
    current_mode = "extended"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for folder in folders:
            folder_path = os.path.join("testing/test_images/benchmark_images", folder)
            print(f"\n📂 Processing folder: {folder}")

            for filename in sorted(os.listdir(folder_path)):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                image_path = os.path.join(folder_path, filename)
                print(f"\n🖼️ Image: {filename}")

                try:
                    cv_img, pil_img = load_images(image_path)
                    pipeline_result = run_pipeline(cv_img, pil_img, models, executor, mode=current_mode)
                    print_pipeline_log(pipeline_result)

                except Exception as e:
                    print(f"❌ Failed to process image {filename}: {e}")
