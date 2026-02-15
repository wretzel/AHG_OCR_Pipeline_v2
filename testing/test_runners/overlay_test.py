# testing/test_runners/overlay_test.py
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from graphics.overlay import OverlayEngine


def main():
    print("\n=== OVERLAY TEST ===\n")

    # --------------------------------------------------------
    # 1. Load a test image or create a blank frame
    # --------------------------------------------------------
    test_img_path = PROJECT_ROOT / "testing" / "test_assets" / "test_frame.jpg"

    if test_img_path.exists():
        frame = cv2.imread(str(test_img_path))
        print(f"Loaded test image: {test_img_path}")
    else:
        print("No test image found — generating blank frame.")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)  # dark gray background

    # --------------------------------------------------------
    # 2. Initialize overlay engine
    # --------------------------------------------------------
    overlay = OverlayEngine()

    # Sample text
    overlay.update_voice("Hello world. This is a subtitle test.")
    overlay.update_ocr("AUTHORIZED PERSONNEL ONLY")

    # --------------------------------------------------------
    # 3. Render overlay
    # --------------------------------------------------------
    out = overlay.render(frame.copy())

    # --------------------------------------------------------
    # 4. Display result
    # --------------------------------------------------------
    cv2.imshow("Overlay Test", out)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
