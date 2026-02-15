# ocr_modules/pipeline_utils/async_pipeline.py

import threading
import time
from ocr_modules.pipeline_utils.pipeline import run_pipeline

class AsyncPipeline:
    def __init__(self, models, executor, mode="steady"):
        self.models = models
        self.executor = executor
        self.mode = mode
        self.is_ready = True
        self.processing_thread = None
        self.lock = threading.Lock()

    def process_frame_async(self, cv_img, pil_img, callback=None):
        if not self.is_ready:
            return False  # Pipeline busy

        self.is_ready = False

        def worker():
            try:
                result = run_pipeline(cv_img, pil_img, self.models, executor=self.executor, mode=self.mode)
                if callback:
                    callback(result)
            except Exception as e:
                print(f"❌ Pipeline error: {e}")
            finally:
                with self.lock:
                    self.is_ready = True

        self.processing_thread = threading.Thread(target=worker, daemon=True)
        self.processing_thread.start()
        return True

    def is_pipeline_ready(self):
        with self.lock:
            return self.is_ready