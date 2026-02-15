# testing/test_runners/ocr_image_test.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
import json
import cv2
from PIL import Image
import concurrent.futures
import traceback

from ocr_modules.pipeline_utils.pipeline import run_pipeline, print_pipeline_log
from ocr_modules.base_modules.initialization import initialize_models
from shared.pipeline_summary import run_summary
from shared.loading_bar import real_loading_bar, start_spinner
import threading
import time


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "..", "test_images", "Benchmark_Images")
)

OUTPUT_JSON = os.path.normpath(
    os.path.join(BASE_DIR, "..", "test_results", "pipeline_outputs.json")
)

CATEGORY_MAP = {
    "1": "clear",
    "2": "complex",
    "3": "scene",
    "4": "dummy",
    "5": "all"
}

# Your preferred category order:
CATEGORY_ORDER = ["clear", "scene", "dummy", "complex"]

DIFFICULTY_ORDER = ["easy", "medium", "hard"]


# ------------------------------------------------------------
# IMAGE DISCOVERY
# ------------------------------------------------------------

def collect_images(selection):
    """
    Returns a dict: {category: [list of image paths]}
    respecting your category + difficulty order.
    """

    categories = CATEGORY_ORDER if selection == "all" else [selection]
    results = {}

    for cat in categories:
        folder = os.path.join(BENCHMARK_DIR, f"{cat}_images")
        if not os.path.isdir(folder):
            print(f"⚠️ Warning: folder not found: {folder}")
            continue

        # Collect images in difficulty order
        ordered_files = []
        for diff in DIFFICULTY_ORDER:
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    if f"{cat}_{diff}" in fname:
                        ordered_files.append(os.path.join(folder, fname))

        results[cat] = ordered_files

    return results


# ------------------------------------------------------------
# MAIN TEST RUNNER
# ------------------------------------------------------------

def main():
    print("\n=== OCR IMAGE TEST RUNNER ===\n")

    # --------------------------------------------------------
    # 1. Show loading bar + spinner while initializing models
    # --------------------------------------------------------
    modules = ["cv2_east", "pytesseract", "spellchecker", "corpus_freqs"]
    real_loading_bar(modules)

    print("\n🔧 Initializing OCR models (this may take a moment)...")
    stop_event = threading.Event()
    spinner_thread = start_spinner("Loading OCR models", stop_event)

    # Actual model loading
    models = initialize_models()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    stop_event.set()
    spinner_thread.join()
    print("✅ Models ready.\n")

    # --------------------------------------------------------
    # 2. User selects category
    # --------------------------------------------------------
    print("Select which image set to run:")
    print("  1. clear")
    print("  2. complex")
    print("  3. scene")
    print("  4. dummy")
    print("  5. all")
    choice = input("\nEnter choice: ").strip()

    if choice not in CATEGORY_MAP:
        print("❌ Invalid choice. Exiting.")
        return

    selection = CATEGORY_MAP[choice]
    print(f"\n📂 Selected category: {selection}\n")

    # --------------------------------------------------------
    # 3. Collect images
    # --------------------------------------------------------
    image_dict = collect_images(selection)
    total_images = sum(len(v) for v in image_dict.values())

    if total_images == 0:
        print("❌ No images found for this selection.")
        return

    print(f"Found {total_images} images.\n")

    # --------------------------------------------------------
    # 4. Run OCR on each image
    # --------------------------------------------------------
    results = {}

    for category, paths in image_dict.items():
        print(f"\n=== CATEGORY: {category.upper()} ===")
        results[category] = {}

        for img_path in paths:
            fname = os.path.basename(img_path)
            print(f"\n📸 Processing: {fname}")

            try:
                # Load images (use safe reads for Windows paths)
                data = np.fromfile(img_path, dtype=np.uint8)
                cv_img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
                pil_img = Image.open(open(img_path, "rb"))

                if cv_img is None:
                    raise RuntimeError("cv2.imdecode returned None (failed to read image)")

                # Run pipeline
                pipeline_result = run_pipeline(
                    cv_img, pil_img, models, executor, mode="steady"
                )

            except Exception as e:
                print("❌ Exception during image processing:")
                traceback.print_exc()
                pipeline_result = {
                    "final_result": {"text": "", "confidence": 0.0, "reliable": False, "error": str(e)},
                    "case_triggered": "exception",
                    "total_runtime": 0.0,
                    "mode": "steady"
                }

            # Print per-image summary
            print_pipeline_log(pipeline_result)

            # Store minimal summary for master summary JSON
            final = pipeline_result.get("final_result", {})
            results[category][fname] = {
                "winner": pipeline_result.get("case_triggered"),
                "confidence": final.get("confidence", 0.0),
                "reliable": final.get("reliable", False),
                "runtime": pipeline_result.get("total_runtime", 0.0)
            }

    # --------------------------------------------------------
    # 5. Save JSON results
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n💾 Results saved to: {OUTPUT_JSON}")

    # --------------------------------------------------------
    # 6. Run master summary
    # --------------------------------------------------------
    print("\n📊 Running master summary...")
    run_summary(OUTPUT_JSON)

    # --------------------------------------------------------
    # 7. Done
    # --------------------------------------------------------
    print("\n🎉 OCR Image Test Complete!\n")


if __name__ == "__main__":
    main()
