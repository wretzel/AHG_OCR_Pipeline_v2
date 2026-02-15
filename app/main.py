# app/main.py

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from ocr_modules.ocr_engine import OCREngine


def main():
    print("\n=== APP: SCAN MODE (Single Image OCR) ===\n")

    if len(sys.argv) < 2:
        print("Usage: python -m app.main <image_path>")
        return

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return

    # Load image
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"❌ Failed to read image: {img_path}")
        return

    # Initialize OCR engine
    ocr = OCREngine(mode="steady")

    # Run OCR
    result = ocr.run(frame)

    print(f"\n📸 Image: {img_path}")
    print(f"📝 Text: {result['text']}")
    print(f"🔍 Confidence: {result['confidence']:.2f}")
    print(f"✅ Reliable: {result['reliable']}")
    print(f"⏱  Runtime: {result['runtime']:.2f}s\n")


if __name__ == "__main__":
    main()
