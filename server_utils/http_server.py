# server_utils/http_server.py
import threading
import cv2
import numpy as np
import socket
import concurrent.futures
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from ocr_modules.base_modules.initialization import initialize_models
from ocr_modules.base_modules.ocr_engines import run_east
from ocr_modules.pipeline_utils.modes import get_mode_budget
from ocr_modules.pipeline_utils.async_pipeline import AsyncPipeline

from .ui_templates import control_page
from .camera import init_camera
from .overlay import overlay_combined   # combined overlay
from .state import AppState
from .voice import VoiceRecognizer
from .ocr_tasks import ocr_task
from .stream_loop import run_stream_phased

# -----------------------------
# Global state and initialization
# -----------------------------

app_state = AppState()

# Initialize models once
models = initialize_models()
_ = run_east(np.zeros((320, 320, 3), dtype=np.uint8))

current_mode = "steady"
mode_budget = get_mode_budget(current_mode)

# Try phone MJPEG stream first, fall back to webcam if unavailable
try:
    cap = init_camera("http://149.61.230.251:8080/video")
    print("📱 Using phone MJPEG stream")
except SystemExit:
    cap = init_camera(0)
    print("💻 Falling back to laptop webcam")

# Shared frame buffer (updated by capture thread)
latest_frame_ref = {'frame': None}
frame_lock = threading.Lock()

# Thread pool for OCR (used by AsyncPipeline)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

# Async OCR pipeline instance
async_pipeline = AsyncPipeline(models=models, executor=executor, mode=current_mode)

# Voice recognizer
voice = VoiceRecognizer(model_path="resources/vosk_model_small", samplerate=16000)
voice.start(device=15) # Specify your audio input device index here

# -----------------------------
# Camera capture thread
# -----------------------------

def camera_loop(cap):
    while app_state.is_running():
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(1)
            continue
        with frame_lock:
            latest_frame_ref['frame'] = frame

# Start capture thread
threading.Thread(target=camera_loop, args=(cap,), daemon=True).start()

# -----------------------------
# HTTP handler
# -----------------------------

class OCRHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(control_page(app_state.is_paused()).encode("utf-8"))

        elif self.path == "/toggle":
            paused = app_state.toggle_pause()
            state = "paused" if paused else "running"
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
            print(f"🔁 OCR toggled: {state}")

        elif self.path == "/quit":
            app_state.stop_server()
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Server shutting down...")
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        elif self.path == "/stream":
            run_stream_phased(self, app_state, frame_lock, latest_frame_ref,
                            models, executor, current_mode, voice,
                            capture_duration=5.0, ocr_duration=5.0)

# -----------------------------
# Server runner
# -----------------------------

def run_server():
    # Detect LAN IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "Unavailable"

    localhost_ip = "127.0.0.1"

    print("📡 OCR Live Runner server available at:")
    print(f"   http://{localhost_ip}:8080   (local only)")
    print(f"   http://{lan_ip}:8080   (LAN devices)")

    server = ThreadingHTTPServer(('0.0.0.0', 8080), OCRHandler)
    try:
        server.serve_forever()
    finally:
        app_state.stop_server()
        executor.shutdown(wait=True)
        cap.release()
        voice.stop()
        print("✅ Server closed")
