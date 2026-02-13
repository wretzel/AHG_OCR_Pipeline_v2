# shared/runner_core.py

import cv2
from PIL import Image
from shared.runtime import timed_run
from ocr_modules.base_modules.ocr_engines import (
    run_tesseract, run_easyocr, run_paddleocr, run_east
)
from PIL import Image
import cv2
import numpy as np

def load_images(image_path):
    pil_img = Image.open(image_path).convert("RGB")
    pil_img.load()  # Fully load into memory
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv_img, pil_img

def run_all_ocr_engines(cv_img, pil_img, models):
    results = {}

    try:
        east_result, runtime = timed_run(run_east, cv_img)
        east_result["runtime"] = runtime
        results["east"] = east_result
    except Exception as e:
        results["east"] = {"error": str(e)}

    try:
        tess_result, runtime = timed_run(run_tesseract, pil_img)
        tess_result["runtime"] = runtime
        results["tesseract"] = tess_result
    except Exception as e:
        results["tesseract"] = {"error": str(e)}

    try:
        easy_result, runtime = timed_run(run_easyocr, cv_img, lang="en")
        easy_result["runtime"] = runtime
        results["easyocr"] = easy_result
    except Exception as e:
        results["easyocr"] = {"error": str(e)}

    try:
        paddle_result, runtime = timed_run(run_paddleocr, cv_img, models["paddleocr_reader"])
        paddle_result["runtime"] = runtime
        results["paddleocr"] = paddle_result
    except Exception as e:
        results["paddleocr"] = {"error": str(e)}

    return results
