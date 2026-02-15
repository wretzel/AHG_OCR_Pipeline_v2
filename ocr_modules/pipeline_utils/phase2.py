# ocr_modules/pipeline_utils/phase2.py

import time
import concurrent.futures
from shared.helper import normalize_conf  # safe float caster
from ocr_modules.pipeline_utils.phase1 import run_easyocr_guided
from ocr_modules.base_modules.ocr_engines import run_easyocr_with_reader

def run_phase2_conditional(cv_img, pil_img, models, east_result,
                           executor=None, budget=2.0, max_crops=5):
    if executor is None:
        local_exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        exec_ctx = local_exec
        owns_executor = True
    else:
        exec_ctx = executor
        owns_executor = False

    overall_start = time.perf_counter()
    case_log = {
        "case_triggered": None,
        "steps": [],
        "final_result": None,
        "total_runtime": None,
        "best_result": None,
        "best_case": None
    }

    best_confidence = -1.0
    best_result_overall = None

    def elapsed():
        return time.perf_counter() - overall_start

    def remaining_budget():
        return max(0.0, budget - elapsed())

    def record_step(case_idx, result, path, engine_name, step_runtime, status):
        result.update({
            "engine": engine_name,
            "path": path,
            "runtime": round(step_runtime, 3),
            "reliable": result.get("reliable", False)
        })
        case_log["steps"].append({
            "case": case_idx,
            "status": status,
            "result": result
        })

    def update_best(case_idx, result):
        nonlocal best_confidence, best_result_overall
        conf = normalize_conf(result.get("confidence"))
        if conf > best_confidence:
            best_confidence = conf
            best_result_overall = result
            case_log["best_case"] = case_idx

    try:
        # Case 1: Guided EasyOCR
        region_count = normalize_conf(east_result.get("region_count"), 0) if east_result else 0
        if east_result and 0 < region_count < max_crops and remaining_budget() > 0.0:
            start = time.perf_counter()
            fut = exec_ctx.submit(run_easyocr_guided, cv_img, models["easyocr_en"], east_result, max_crops=max_crops)
            try:
                easy_result = fut.result(timeout=remaining_budget())
                step_runtime = time.perf_counter() - start
                status = "success" if easy_result.get("text") else "fail"
                record_step(1, easy_result, "Phase2 Case 1: EasyOCR guided by EAST", "easyocr-guided", step_runtime, status)
                update_best(1, easy_result)

                if easy_result.get("text") and (
                    easy_result.get("reliable", False) or normalize_conf(easy_result.get("confidence")) >= 0.8
                ):
                    case_log["case_triggered"] = 1
                    case_log["final_result"] = easy_result
                    case_log["total_runtime"] = round(elapsed(), 3)
                    return case_log
            except concurrent.futures.TimeoutError:
                step_runtime = time.perf_counter() - start
                placeholder = {"text": "", "confidence": 0.0, "reliable": False}
                record_step(1, placeholder, "Phase2 Case 1: EasyOCR guided by EAST (timeout)", "easyocr-guided", step_runtime, "timeout")
                case_log["case_triggered"] = 1
                case_log["final_result"] = best_result_overall if best_result_overall else placeholder
                case_log["total_runtime"] = round(budget, 3)
                return case_log

        # Case 2: EasyOCR full-image (normal thresholds)
        if remaining_budget() > 0.0:
            start = time.perf_counter()
            fut = exec_ctx.submit(run_easyocr_with_reader, cv_img, models["easyocr_en"], min_token_conf=0.6)
            try:
                easy_result = fut.result(timeout=remaining_budget())
                step_runtime = time.perf_counter() - start
                status = "success" if easy_result.get("text") else "fail"
                record_step(2, easy_result, "Phase2 Case 2: EasyOCR full-image", "easyocr-full", step_runtime, status)
                update_best(2, easy_result)

                if easy_result.get("text") and (
                    easy_result.get("reliable", False) or normalize_conf(easy_result.get("confidence")) >= 0.8
                ):
                    case_log["case_triggered"] = 2
                    case_log["final_result"] = easy_result
                    case_log["total_runtime"] = round(elapsed(), 3)
                    return case_log
            except concurrent.futures.TimeoutError:
                step_runtime = time.perf_counter() - start
                placeholder = {"text": "", "confidence": 0.0, "reliable": False}
                record_step(2, placeholder, "Phase2 Case 2: EasyOCR full-image (timeout)", "easyocr-full", step_runtime, "timeout")
                case_log["case_triggered"] = case_log["best_case"] if case_log["best_case"] else 2
                case_log["final_result"] = best_result_overall if best_result_overall else placeholder
                case_log["total_runtime"] = round(budget, 3)
                return case_log

        # Case 3: EasyOCR full-image (looser thresholds)
        if remaining_budget() > 0.0:
            start = time.perf_counter()
            fut = exec_ctx.submit(run_easyocr_with_reader, cv_img, models["easyocr_en"], min_token_conf=0.3)
            try:
                backup_result = fut.result(timeout=remaining_budget())
                step_runtime = time.perf_counter() - start
                backup_result.update({"backup_triggered": True})
                status = "success" if backup_result.get("text") else "fail"
                record_step(3, backup_result, "Phase2 Case 3: EasyOCR full-image (looser thresholds)", "easyocr-full-loose", step_runtime, status)
                update_best(3, backup_result)
            except concurrent.futures.TimeoutError:
                step_runtime = time.perf_counter() - start
                placeholder = {"text": "", "confidence": 0.0, "reliable": False, "backup_triggered": True}
                record_step(3, placeholder, "Phase2 Case 3: EasyOCR full-image (looser thresholds, timeout)", "easyocr-full-loose", step_runtime, "timeout")
                case_log["case_triggered"] = case_log["best_case"] if case_log["best_case"] else 3
                case_log["final_result"] = best_result_overall if best_result_overall else placeholder
                case_log["total_runtime"] = round(budget, 3)
                return case_log

        # Finalize within budget
        case_log["case_triggered"] = case_log["best_case"]
        case_log["final_result"] = best_result_overall if best_result_overall else (
            case_log["steps"][-1]["result"] if case_log["steps"] else {"text": "", "confidence": 0.0, "reliable": False}
        )
        runtime = elapsed()
        case_log["total_runtime"] = round(runtime if runtime <= budget else budget, 3)
        return case_log

    finally:
        if owns_executor:
            exec_ctx.shutdown(wait=False)

def print_phase2_log(case_log):
    for step in case_log["steps"]:
        case = step["case"]
        status = step["status"]
        result = step["result"]
        print(f"📍 Case {case}: {result['path']} → {status.upper()}")
        if result.get("text"):
            print(f"   Text: '{result['text']}' (conf={normalize_conf(result.get('confidence'))})")

    final = case_log["final_result"]
    print(f"🔎 Final OCR result (Case {case_log['case_triggered']}): "
          f"{final.get('text','')} (conf={normalize_conf(final.get('confidence'))})")
    print(f"⏱️ Phase 2 runtime: {case_log['total_runtime']}s")

