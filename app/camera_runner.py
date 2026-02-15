# app/camera_runner.py

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import time

from app import config
from ocr_modules.camera_source import CameraSource
from ocr_modules.async_ocr_engine import AsyncOCREngine
from graphics.overlay import OverlayEngine

if config.ENABLE_VOICE:
    from voice.async_voice_engine import AsyncVoiceEngine


def main():
    print("\n=== APP: LIVE OCR OVERLAY ===\n")

    # --------------------------------------------------------
    # 1. Initialize camera
    # --------------------------------------------------------
    print("📷 Initializing camera...")
    camera = CameraSource(config.CAMERA_SOURCE)
    print(f"✅ Camera ready: {config.CAMERA_SOURCE}")

    # --------------------------------------------------------
    # 2. Initialize OCR (async) + overlay
    # --------------------------------------------------------
    print("🔧 Initializing OCR engine...")
    ocr = AsyncOCREngine(mode=config.OCR_MODE, max_workers=config.OCR_MAX_WORKERS)
    overlay = OverlayEngine()

    # --------------------------------------------------------
    # 3. Initialize voice (optional)
    # --------------------------------------------------------
    voice = None
    if config.ENABLE_VOICE:
        print("🎤 Initializing voice engine...")
        voice = AsyncVoiceEngine(
            model_path=str(config.VOSK_MODEL_PATH),
            samplerate=config.VOICE_SAMPLERATE,
            silence_threshold=config.VOICE_SILENCE_THRESHOLD,
            min_output_interval=config.VOICE_MIN_OUTPUT_INTERVAL,
        )
        voice.start()
        print("✅ Voice engine started.")

    # --------------------------------------------------------
    # 4. OCR callback → update overlay OCR text
    # --------------------------------------------------------
    def ocr_callback(result):
        final = result.get("final_result", {})
        text = final.get("text", "")
        overlay.update_ocr(text)

    print("\nPress 'q' to quit.\n")

    fps_timer = time.time()
    fps_counter = 0

    # --------------------------------------------------------
    # 5. Main loop
    # --------------------------------------------------------
    while True:
        frame = camera.read()
        if frame is None:
            continue

        raw = frame.copy()

        # Kick off async OCR on raw frame
        ocr.process(raw, callback=ocr_callback)

        # Update voice subtitle
        if voice is not None:
            voice.tick()
            latest_voice = voice.get_latest()
            overlay.update_voice(latest_voice)


        # Draw overlay on a *separate* copy
        display_frame = raw.copy()
        overlay.render(display_frame)

        cv2.imshow(config.WINDOW_TITLE, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------------
    # 6. Cleanup
    # --------------------------------------------------------
    camera.release()
    ocr.shutdown()
    if voice is not None:
        voice.stop()

    cv2.destroyAllWindows()
    print("\n👋 Live OCR overlay ended.\n")


if __name__ == "__main__":
    main()
