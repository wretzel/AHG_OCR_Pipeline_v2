import threading

class AppState:
    """Thread-safe application state."""
    def __init__(self):
        self.ocr_paused = False
        self.server_running = True
        self.latest_text = ""
        self.latest_conf = 0.0
        self.future = None
        self.lock = threading.Lock()

    def toggle_pause(self):
        """Toggle OCR pause state."""
        with self.lock:
            self.ocr_paused = not self.ocr_paused
        return self.ocr_paused

    def set_ocr_result(self, text, conf):
        """Update latest OCR result."""
        with self.lock:
            self.latest_text = text
            self.latest_conf = conf

    def get_ocr_result(self):
        """Get latest OCR result."""
        with self.lock:
            return self.latest_text, self.latest_conf

    def is_paused(self):
        """Check if OCR is paused."""
        with self.lock:
            return self.ocr_paused

    def set_future(self, future):
        """Set active OCR future."""
        with self.lock:
            self.future = future

    def get_future(self):
        """Get active OCR future."""
        with self.lock:
            return self.future

    def stop_server(self):
        """Stop server."""
        with self.lock:
            self.server_running = False

    def is_running(self):
        """Check if server is running."""
        with self.lock:
            return self.server_running