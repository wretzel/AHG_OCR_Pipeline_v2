# app/config.py

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# OCR settings
OCR_MODE = "steady"  # "fast", "steady", "extended"
OCR_MAX_WORKERS = 3

# Camera settings
CAMERA_SOURCE = 0  # webcam index or URL string

# Voice settings
ENABLE_VOICE = True
VOSK_MODEL_PATH = PROJECT_ROOT / "resources" / "vosk_model_small"
VOICE_SAMPLERATE = 16000
VOICE_SILENCE_THRESHOLD = 1.0
VOICE_MIN_OUTPUT_INTERVAL = 0.4

# Window settings
WINDOW_TITLE = "Live OCR Overlay"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
