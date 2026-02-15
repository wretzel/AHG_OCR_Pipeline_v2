# ocr_modules/pipeline_utils/ocr_race.py
import concurrent.futures
import time
from threading import Event, Thread
import gc

from ocr_modules.base_modules.ocr_engines import (
    run_tesseract,
    run_paddleocr,
    run_easyocr_with_reader,
)

def run_with_abort_check(fn, *args, stop_event=None, max_time=3.5, **kwargs):
    result = [None]
    thread = Thread(
        target=lambda: result.__setitem__(0, fn(*args, **kwargs)),
        name=f"OCR-{fn.__name__}"
    )
    thread.start()

    start = time.perf_counter()
    while thread.is_alive():
        if stop_event and stop_event.is_set():
            return {"skipped": True, "aborted": True}
        if time.perf_counter() - start > max_time:
            print("⏱️ Engine timed out internally.")
            return {"skipped": True, "timed_out": True}
        time.sleep(0.05)

    return result[0]
from ocr_modules.base_modules.preprocess import crop_regions, aggregate_crop_results

def run_easyocr_guided(cv_img, reader, east_result=None, conf_threshold=0.6):
    if east_result and east_result.get("region_count", 0) > 0:
        crops = crop_regions(cv_img, east_result)
        return aggregate_crop_results(crops, run_easyocr_with_reader, reader,
                                      conf_threshold=conf_threshold)
    else:
        return run_easyocr_with_reader(cv_img, reader)

def ocr_race_engines(cv_img, pil_img, models, timeout=5.0, east_result=None):
    stop_event = Event()
    start_time = time.perf_counter()
    results = {}
    winner = None

    def engine_wrapper(name, stop_event):
        if stop_event.is_set():
            print(f"⏭️ {name} aborted early due to winner.")
            return {"engine": name, "skipped": True, "aborted": True}

        try:
            start = time.perf_counter()
            if name == "tesseract":
                result = run_with_abort_check(run_tesseract, pil_img,
                                              stop_event=stop_event, max_time=3.5)
            elif name == "paddleocr":
                reader = models.get("paddleocr_reader")
                if east_result and east_result.get("region_count", 0) > 0:
                    # guided mode
                    result = run_with_abort_check(run_paddleocr, cv_img, reader,
                                                  east_result, stop_event=stop_event, max_time=3.5)
                else:
                    # full-image mode
                    result = run_with_abort_check(run_paddleocr, cv_img, reader,
                                                  stop_event=stop_event, max_time=3.5)
            elif name == "easyocr":
                reader = models.get("easyocr_en")
                if east_result and east_result.get("region_count", 0) > 0:
                    # Guided mode: crop first, then aggregate results
                    result = run_easyocr_guided(cv_img, reader, east_result)
                else:
                    # Full image mode
                    result = run_with_abort_check(run_easyocr_with_reader, cv_img, reader,
                                                stop_event=stop_event, max_time=3.5)
            else:
                return None

            result["engine"] = name
            result["runtime"] = round(time.perf_counter() - start, 3)
            if result.get("skipped"):
                result.setdefault("aborted", False)
                result.setdefault("timed_out", False)
            return result

        except Exception as e:
            print(f"🧪 {name} crashed: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(engine_wrapper, name, stop_event): name
                for name in ["tesseract", "easyocr"]}
        try:
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                result = future.result()
                if not isinstance(result, dict):
                    continue
                name = result.get("engine")
                results[name] = result
                if result.get("reliable"):
                    winner = result
                    print(f"🏁 Reliable result from {name}. Cancelling other threads...")
                    stop_event.set()
                    # cancel remaining futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
        except concurrent.futures.TimeoutError:
            print("⚠️ OCR race timed out.")
            stop_event.set()
            for f in futures:
                if not f.done():
                    f.cancel()

    elapsed = round(time.perf_counter() - start_time, 3)
    for name in ["tesseract", "paddleocr", "easyocr"]:
        if name not in results:
            results[name] = {"engine": name, "skipped": True, "aborted": True}

    gc.collect()

    if not isinstance(winner, dict):
        return {
            "winner": None,
            "final_text": "",
            "confidence": 0.0,
            "corpus_score": 0.0,
            "reliable": False,
            "runtime": elapsed,
            "winner_runtime": None,
            "all_outputs": results,
        }

    return {
        "winner": winner.get("engine"),
        "final_text": winner.get("text"),
        "confidence": winner.get("confidence", 0.0),
        "corpus_score": winner.get("corpus_score", 0.0),
        "reliable": True,
        "runtime": elapsed,
        "winner_runtime": winner.get("runtime", None),
        "all_outputs": results,
    }
