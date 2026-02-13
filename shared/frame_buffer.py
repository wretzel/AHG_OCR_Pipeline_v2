# shared/frame_buffer.py

import threading
from collections import deque

class FrameBuffer:
    """Thread-safe frame buffer with latest frame tracking."""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ocr = None
        self.frame_id = 0

    def push_frame(self, frame):
        """Add new camera frame (replaces old one)."""
        with self.lock:
            self.frame_id += 1
            self.latest_frame = frame.copy()
            return self.frame_id

    def get_latest_frame(self):
        """Get latest frame without removing it."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def set_ocr_result(self, frame_id, ocr_result):
        """Store OCR result for a frame."""
        with self.lock:
            self.latest_ocr = {"frame_id": frame_id, "result": ocr_result}

    def get_latest_ocr(self):
        """Get latest OCR result."""
        with self.lock:
            return self.latest_ocr.copy() if self.latest_ocr else None