# ocr_modules/async_ocr_engine.py

import time
import cv2
import numpy as np
from PIL import Image
import concurrent.futures

from ocr_modules.base_modules.initialization import initialize_models
from ocr_modules.pipeline_utils.async_pipeline import AsyncPipeline
from ocr_modules.pipeline_utils.modes import MODES


class AsyncOCREngine:
    """
    High-level async OCR engine wrapper.
    Handles:
      - model initialization
      - async pipeline
      - cv2→PIL conversion
      - mode timing (fast/steady/extended)
      - callback dispatch
    """

    def __init__(self, mode="steady", max_workers=3):
        self.mode = mode
        self.models = initialize_models()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        self.pipeline = AsyncPipeline(
            models=self.models,
            executor=self.executor,
            mode=self.mode
        )

        self.last_ocr_time = 0.0

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    @staticmethod
    def _cv2_to_pil(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _can_run_now(self):
        """
        Enforce mode timing (min_interval).
        """
        now = time.time()
        if now - self.last_ocr_time < MODES[self.mode]["min_interval"]:
            return False
        return True

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def process(self, frame, callback):
        """
        Process a frame asynchronously.
        callback(result_dict) is called when OCR completes.
        """

        if frame is None:
            return

        # Enforce mode timing
        if not self._can_run_now():
            return

        # Only run if pipeline is ready
        if not self.pipeline.is_pipeline_ready():
            return

        self.last_ocr_time = time.time()

        pil_img = self._cv2_to_pil(frame)

        # AsyncPipeline handles threading internally
        self.pipeline.process_frame_async(
            frame,
            pil_img,
            callback=callback
        )

    def set_mode(self, mode):
        self.mode = mode
        self.pipeline.mode = mode

    def shutdown(self):
        self.executor.shutdown(wait=False)
