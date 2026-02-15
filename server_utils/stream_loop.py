# server_utils/stream_loop.py
import cv2
from .overlay import overlay_combined
from .ocr_tasks import ocr_task

import time
import cv2
from .overlay import overlay_combined
from .ocr_tasks import ocr_task

def run_stream_phased(self, app_state, frame_lock, latest_frame_ref,
                      models, executor, current_mode, voice,
                      capture_duration=3.0, ocr_duration=3.0):
    boundary = "--frame"
    self.send_response(200)
    self.send_header('Content-type', f'multipart/x-mixed-replace; boundary={boundary}')
    self.end_headers()

    phase = "capture"
    phase_end = time.time() + capture_duration
    frozen_frame = None
    ocr_ran_this_phase = False

    while app_state.is_running():
        now = time.time()
        if now >= phase_end:
            if phase == "capture":
                with frame_lock:
                    frozen_frame = (latest_frame_ref['frame'].copy()
                                    if latest_frame_ref['frame'] is not None else None)
                ocr_ran_this_phase = False
                phase = "ocr"
                phase_end = now + ocr_duration
            else:
                phase = "capture"
                phase_end = now + capture_duration

        frame = None
        if phase == "capture":
            with frame_lock:
                frame = (latest_frame_ref['frame'].copy()
                         if latest_frame_ref['frame'] is not None else None)
        else:
            frame = frozen_frame.copy() if frozen_frame is not None else None

        if frame is None:
            cv2.waitKey(1)
            continue

        if phase == "ocr" and not app_state.is_paused() and not ocr_ran_this_phase:
            try:
                pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                text, conf = ocr_task(frame, pil_img, models, executor, current_mode)
                app_state.set_ocr_result(text, conf)
            except Exception as e:
                print(f"❌ OCR error: {e}")
            finally:
                ocr_ran_this_phase = True

        text, conf = app_state.get_ocr_result()
        voice_lines = voice.latest_lines(n=1)
        display = overlay_combined(frame, text, conf, voice_lines)

        ret, jpeg = cv2.imencode('.jpg', display)
        if not ret:
            continue

        try:
            self.wfile.write(boundary.encode("utf-8") + b"\r\n")
            header = (
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n"
            ).encode("utf-8")
            self.wfile.write(header)
            self.wfile.write(jpeg.tobytes())
            self.wfile.write(b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            print("🔌 Client disconnected")
            return
