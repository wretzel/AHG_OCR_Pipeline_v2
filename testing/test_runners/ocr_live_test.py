# testing/test_runners/ocr_live_test.py
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import time
import numpy as np
from PIL import Image
import concurrent.futures
import threading

from ocr_modules.base_modules.initialization import initialize_models
from ocr_modules.pipeline_utils.async_pipeline import AsyncPipeline
from ocr_modules.pipeline_utils.modes import MODES, enforce_mode
from shared.loading_bar import real_loading_bar, start_spinner
from server_utils.camera import init_camera


# ------------------------------------------------------------
# MAIN LIVE OCR TEST
# ------------------------------------------------------------

def main():
    print("\n=== OCR LIVE TEST (Terminal Mode) ===\n")

    # --------------------------------------------------------
    # 1. Loading screen
    # --------------------------------------------------------
    modules = ["cv2_east", "pytesseract", "spellchecker", "corpus_freqs"]
    real_loading_bar(modules)

    print("\n🔧 Initializing OCR models...")
    stop_event = threading.Event()
    spinner_thread = start_spinner("Loading OCR models", stop_event)

    models = initialize_models()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    stop_event.set()
    spinner_thread.join()
    print("✅ Models ready.\n")

    # --------------------------------------------------------
    # 2. Mode selection
    # --------------------------------------------------------
    print("Select OCR mode:")
    print("  1. fast")
    print("  2. steady")
    print("  3. extended")
    choice = input("\nEnter choice: ").strip()

    mode_map = {"1": "fast", "2": "steady", "3": "extended"}
    mode = mode_map.get(choice, "steady")

    print(f"\n⚙️ Running in mode: {mode}\n")

    # --------------------------------------------------------
    # 3. Open camera
    # --------------------------------------------------------
    print("📷 Initializing camera...")

    cap = init_camera(0)
    print("💻 Using webcam")

    async_pipeline = AsyncPipeline(models=models, executor=executor, mode=mode)

    last_ocr_time = 0
    frame_count = 0
    fps_timer = time.time()
    fps_counter = 0

    print("\nPress 'q' to quit.\n")

    # --------------------------------------------------------
    # 4. Main loop
    # --------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(1)
            continue

        frame_count += 1
        fps_counter += 1

        # FPS print every second
        if time.time() - fps_timer >= 1.0:
            print(f"📈 FPS: {fps_counter}")
            fps_counter = 0
            fps_timer = time.time()

        # Convert to PIL
        data = np.asarray(frame)
        pil_img = Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

        # Respect mode timing
        now = time.time()
        if now - last_ocr_time < MODES[mode]["min_interval"]:
            continue

        # Only run OCR if pipeline is ready
        if async_pipeline.is_pipeline_ready():
            last_ocr_time = now

            def callback(result):
                text = result["final_result"].get("text", "")
                conf = result["final_result"].get("confidence", 0.0)
                reliable = result["final_result"].get("reliable", False)
                runtime = result.get("total_runtime", 0.0)

                print(f"\n[OCR] text=\"{text}\"")
                print(f"      conf={conf:.2f} reliable={reliable} runtime={runtime:.2f}s\n")

            async_pipeline.process_frame_async(frame, pil_img, callback=callback)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --------------------------------------------------------
    # 5. Cleanup
    # --------------------------------------------------------
    cap.release()
    executor.shutdown(wait=True)
    print("\n👋 Live OCR test ended.\n")


if __name__ == "__main__":
    main()
