# ocr_modules/base_modules/initialization.py

import time
import os
import sys
import json
import logging
import warnings
import contextlib
import numpy as np
import pytesseract
import easyocr
import cv2
from spellchecker import SpellChecker
from paddleocr import PaddleOCR
import subprocess
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

# Suppress EasyOCR logs
logging.getLogger("easyocr").addHandler(logging.NullHandler())
logging.getLogger("easyocr").setLevel(logging.CRITICAL)

# Suppress PaddleOCR warnings and logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["FLAGS_log_level"] = "0"
os.environ["GLOG_minloglevel"] = "3"

import os
import sys
import contextlib

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())

        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
import logging

def suppress_paddle_logging():
    logging.getLogger().setLevel(logging.CRITICAL)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def initialize_models(callback=None):
    diagnostics = {}
    models = {}

    def timed_load(name, loader):
        start = time.time()
        try:
            result = loader()
            load_time = round(time.time() - start, 3)
            diagnostics[name] = {"status": True, "load_time": load_time}
            if callback:
                callback(name, True, load_time, None)
            return result
        except Exception as e:
            diagnostics[name] = {"status": False, "load_time": None, "error": str(e)}
            if callback:
                callback(name, False, None, str(e))
            return None

    # Tesseract
    def load_tesseract():
        # Explicit path (adjust if needed)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Users\hgk07\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

        # Check if binary is callable
        try:
            version_output = subprocess.check_output(
                [pytesseract.pytesseract.tesseract_cmd, "--version"],
                stderr=subprocess.STDOUT,
                text=True
            )
        except Exception as e:
            raise RuntimeError(f"Tesseract binary not callable: {e}")

        # Run dummy OCR to confirm functionality
        try:
            img = Image.new("RGB", (100, 30), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((10, 5), "Test", fill=(0, 0, 0))
            text = pytesseract.image_to_string(img)
            if "Test" not in text:
                raise RuntimeError("Tesseract OCR failed to detect dummy text.")
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR failed: {e}")

        return True


    timed_load("pytesseract", load_tesseract)

    # EasyOCR
    with suppress_output():
        def load_easyocr_reader(lang_list, name):
            reader = easyocr.Reader(lang_list, gpu=False)
            _ = reader.readtext(np.zeros((10, 10, 3), dtype=np.uint8))  # force model load
            return reader

        models["easyocr_en"] = timed_load("easyocr_en", lambda: load_easyocr_reader(['en'], "easyocr_en"))
        models["easyocr_ru"] = timed_load("easyocr_ru", lambda: load_easyocr_reader(['ru', 'en'], "easyocr_ru"))
        models["easyocr_ar"] = timed_load("easyocr_ar", lambda: load_easyocr_reader(['ar', 'en'], "easyocr_ar"))
        models["easyocr_ch"] = timed_load("easyocr_ch", lambda: load_easyocr_reader(['ch_sim', 'en'], "easyocr_ch"))

    # PaddleOCR
    with suppress_output():
        suppress_paddle_logging()
        models["paddleocr_reader"] = timed_load("paddleocr", lambda: PaddleOCR(use_textline_orientation=True, lang='en'))

    # EAST
    with suppress_output():
        models["east"] = timed_load("cv2_east", lambda: cv2.dnn.readNet("resources/east_model.pb"))


    # Spellchecker
    models["spellchecker"] = timed_load("spellchecker", lambda: SpellChecker())

    # Corpus freqs
    def load_corpus():
        with open("resources/corpus_freqs.json", "r") as f:
            return json.load(f)

    models["corpus_freqs"] = timed_load("corpus_freqs", load_corpus)

    return {
        "status": "initialized",
        "diagnostics": diagnostics,
        **models
    }
