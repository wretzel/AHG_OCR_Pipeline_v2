# ocr_modules/ocr_engine.py

import cv2
import numpy as np
from PIL import Image
import concurrent.futures

from ocr_modules.base_modules.initialization import initialize_models
from ocr_modules.pipeline_utils.pipeline import run_pipeline


class OCREngine:
    """
    High-level OCR engine wrapper.
    Provides a clean interface for:
      - model initialization
      - running OCR on cv2 frames
      - selecting pipeline mode ("fast", "steady", "extended")
      - returning clean results for app runners
    """

    def __init__(self, mode="steady", max_workers=3):
        """
        mode: "fast", "steady", or "extended"
        max_workers: thread pool size for OCR pipeline
        """
        self.mode = mode
        self.models = initialize_models()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    @staticmethod
    def _cv2_to_pil(frame):
        """
        Convert a cv2 BGR frame to a PIL RGB image.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def run(self, frame):
        """
        Run OCR on a single cv2 frame.
        Returns a dict:
            {
                "text": str,
                "confidence": float,
                "reliable": bool,
                "runtime": float,
                "mode": str
            }
        """

        if frame is None:
            return {
                "text": "",
                "confidence": 0.0,
                "reliable": False,
                "runtime": 0.0,
                "mode": self.mode,
                "error": "Frame is None"
            }

        pil_img = self._cv2_to_pil(frame)

        # Run your existing pipeline
        result = run_pipeline(
            frame,
            pil_img,
            self.models,
            self.executor,
            mode=self.mode
        )

        final = result.get("final_result", {})

        return {
            "text": final.get("text", ""),
            "confidence": final.get("confidence", 0.0),
            "reliable": final.get("reliable", False),
            "runtime": result.get("total_runtime", 0.0),
            "mode": result.get("mode", self.mode),
        }

    def set_mode(self, mode):
        """
        Change OCR mode at runtime.
        """
        self.mode = mode

    def shutdown(self):
        """
        Cleanly shut down thread pool.
        """
        self.executor.shutdown(wait=False)
