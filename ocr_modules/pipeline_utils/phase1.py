# ocr_modules/pipeline_utils/phase1.py

import time
import concurrent.futures
from shared.runtime import timed_run
from shared.helper import normalize_conf  # safe float caster
from ocr_modules.base_modules.ocr_engines import run_east, run_easyocr_with_reader, run_tesseract
from ocr_modules.base_modules.preprocess import crop_regions, aggregate_crop_results

def run_phase1_parallel(cv_img, pil_img, executor, budget=2.0, models=None):
    phase1_start = time.perf_counter()
    
    # Submit both tasks
    # Pass preloaded models to the EAST worker to avoid reinitialization
    future_east = executor.submit(run_east, cv_img, models)
    east_start = time.perf_counter()

    future_tess = executor.submit(run_tesseract, pil_img)
    tess_start = time.perf_counter()
    
    try:
        east_result = future_east.result(timeout=budget)
        east_time = round(time.perf_counter() - east_start, 3)
    except concurrent.futures.TimeoutError:
        print(f"⚠️ EAST timeout ({budget}s budget exceeded)")
        east_result = {"region_count": 0, "regions": []}
        east_time = round(budget, 3)
    except Exception as e:
        print("❌ Exception while retrieving EAST future:")
        import traceback as _tb
        _tb.print_exc()
        east_result = {"region_count": 0, "regions": [], "error": str(e)}
        east_time = round(time.perf_counter() - east_start, 3)
    
    try:
        tess_result = future_tess.result(timeout=budget)
        tess_time = round(time.perf_counter() - tess_start, 3)
    except concurrent.futures.TimeoutError:
        print(f"⚠️ Tesseract timeout ({budget}s budget exceeded)")
        tess_result = {"text": "", "confidence": 0.0}
        tess_time = round(budget, 3)
    except Exception as e:
        print("❌ Exception while retrieving Tesseract future:")
        import traceback as _tb
        _tb.print_exc()
        tess_result = {"text": "", "confidence": 0.0, "error": str(e)}
        tess_time = round(time.perf_counter() - tess_start, 3)
    
    elapsed = round(time.perf_counter() - phase1_start, 3)
    
    # EAST summary
    regions = east_result.get("regions", [])
    region_count = normalize_conf(east_result.get("region_count"), 0)
    avg_conf = round(
        sum(normalize_conf(r.get("confidence")) for r in regions) / max(region_count, 1),
        2
    )
    
    tess_conf = normalize_conf(tess_result.get("confidence"))
    tess_result["isReliable"] = tess_conf >= 0.6
    
    return {
        "east_result": east_result,
        "east_time": east_time,
        "tess_result": tess_result,
        "tess_time": tess_time,
        "elapsed": elapsed,
        "budget": budget,
        "budget_exceeded": elapsed >= budget
    }

def print_phase1_log(phase1_result):
    east_result = phase1_result["east_result"]
    tess_result = phase1_result["tess_result"]
    east_time = phase1_result["east_time"]
    tess_time = phase1_result["tess_time"]
    elapsed = phase1_result["elapsed"]
    budget = phase1_result["budget"]
    
    regions = east_result.get("regions", [])
    region_count = normalize_conf(east_result.get("region_count"), 0)
    avg_conf = round(
        sum(normalize_conf(r.get("confidence")) for r in regions) / max(region_count, 1),
        2
    )
    tess_conf = normalize_conf(tess_result.get("confidence"))
    
    print(f"📦 EAST regions: {region_count} boxes, avg confidence: {avg_conf} (runtime: {east_time}s)")
    print(f"🧠 Tesseract complete (runtime: {tess_time}s)")
    print(f"   Parallel overhead check: max({east_time}s, {tess_time}s) = {max(east_time, tess_time)}s")
    print(f"⏱️ Phase 1 elapsed: {elapsed}s (budget: {budget}s)")

def run_easyocr_guided(cv_img, reader, east_result=None,
                       conf_threshold=0.6, min_token_conf=0.6, max_crops=5, verbose=False):
    region_count = normalize_conf(east_result.get("region_count"), 0) if east_result else 0
    if east_result and region_count > 0:
        if region_count >= max_crops:
            # Too many boxes → fallback to full image
            return run_easyocr_with_reader(cv_img, reader, min_token_conf=min_token_conf)
        else:
            crops = crop_regions(cv_img, east_result)
            return aggregate_crop_results(
                crops,
                run_easyocr_with_reader,
                reader,
                conf_threshold=conf_threshold,
                min_crop_conf=min_token_conf,
                verbose=verbose
            )
    else:
        # No EAST regions → full image
        return run_easyocr_with_reader(cv_img, reader, min_token_conf=min_token_conf)
