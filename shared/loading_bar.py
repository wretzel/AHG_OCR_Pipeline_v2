# ocr_modules/shared/loading_bar.py

import time
import random
import itertools
import threading

def start_spinner(label, stop_event):
    def spinner_loop():
        try:
            spinner = ["|", "/", "-", "\\"]
            idx = 0
            print(f"{label} ", end="", flush=True)

            while not stop_event.is_set():
                try:
                    print(f"\r{label} {spinner[idx % len(spinner)]}", end="", flush=True)
                    idx += 1
                except Exception:
                    pass  

            # Overwrite spinner with final ✅
            print(f"\r{label} ✅", end="\n", flush=True)
        except Exception as e:
            print(f"⚠️ Spinner thread crashed: {e}")

    thread = threading.Thread(target=spinner_loop)
    thread.start()
    return thread

def real_loading_bar(modules):
    print("⏳ Initializing modules...\n")
    dot_cycle = itertools.cycle([".", "..", "..."])
    total = len(modules)

    for i, name in enumerate(modules, start=1):
        if "easyocr" in name:
            delay = random.uniform(1.6, 2.6)
        elif "paddleocr" in name:
            delay = random.uniform(3.5, 4.2)
        elif "cv2_east" in name:
            delay = random.uniform(0.3, 0.4)
        elif "spellchecker" in name:
            delay = random.uniform(0.08, 0.1)
        elif "corpus_freqs" in name or "pytesseract" in name:
            delay = random.uniform(0.0, 0.01)
        else:
            delay = random.uniform(0.5, 1.0)

        start_time = time.time()
        while time.time() - start_time < delay:
            dots = next(dot_cycle)
            print(f"\r🔄 [{i}/{total}] Loading {name}{dots}   ", end="", flush=True)
            time.sleep(0.3)
        print(f"\r✅ [{i}/{total}] {name} loaded.        ")

def show_spinner(done_flag):
    message = "📦 Finalizing setup"
    dot_frames = [".  ", ".. ", "..."]
    idx = 0

    while not done_flag.get("complete", False):
        print(f"\r{message}{dot_frames[idx % len(dot_frames)]}", end="", flush=True)
        time.sleep(0.4)
        idx += 1

    print("\r✅ Setup complete. Ready to run OCR.")
    time.sleep(0.3)