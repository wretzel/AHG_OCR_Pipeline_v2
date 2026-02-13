# testing/test_runners/baby_steps.py
import os
import sys
import time
import cv2
import numpy as np
import concurrent.futures

# Project root setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from ocr_modules.base_modules.initialization import initialize_models
from ocr_modules.base_modules.ocr_engines import run_east
from ocr_modules.pipeline_utils.pipeline import run_pipeline, print_pipeline_log
from ocr_modules.pipeline_utils.modes import get_mode_budget

def load_ocr_models(force_reload=False):
    global _models
    if "_models" not in globals() or _models is None or force_reload:
        _models = initialize_models()
    return _models

def overlay_text_top_center(frame, text, conf):
    """Draw OCR text at top center of the frame."""
    display = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(2.0, 30 / max(len(text), 1)))
    thickness = max(1, int(scale * 2))
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (display.shape[1] - w) // 2
    y = h + 20

    # Background box
    overlay = display.copy()
    cv2.rectangle(overlay, (x - 10, y - h - 10), (x + w + 10, y + 10), (0, 0, 0), -1)
    display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)

    # Text
    cv2.putText(display, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # Confidence in corner
    info = f"Conf: {conf:.2f}"
    cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return display

if __name__ == "__main__":
    models = load_ocr_models()
    _ = run_east(np.zeros((320, 320, 3), dtype=np.uint8))

    current_mode = "steady"
    mode_budget = get_mode_budget(current_mode)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Could not open camera")
        sys.exit(1)

    print(f"🎥 Frame-by-frame OCR (mode={current_mode}, budget={mode_budget}s). Press 'q' to quit.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        while True:
            # Flush buffer to avoid stale frames
            for _ in range(5):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame to correct mirroring
            frame = cv2.flip(frame, 1)

            # Prepare inputs
            cv_img = frame.copy()
            pil_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            # Run pipeline synchronously
            result = run_pipeline(cv_img, pil_img, models, executor, mode=current_mode)

            # Overlay and display the same frame that was analyzed
            text = result["final_result"].get("text", "")
            conf = result["final_result"].get("confidence", 0.0)
            display = overlay_text_top_center(cv_img, text, conf)

            # Show the result in a Python window
            cv2.imshow("OCR Result", display)

            # Print pipeline summary AFTER display
            print_pipeline_log(result)

            # Wait for mode pacing before next frame
            time.sleep(mode_budget)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()
