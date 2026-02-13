# ocr_modules/pipeline_utils/pipeline.py

import time
from ocr_modules.pipeline_utils.phase1 import run_phase1_parallel, print_phase1_log
from ocr_modules.pipeline_utils.phase2 import run_phase2_conditional, print_phase2_log
from ocr_modules.pipeline_utils.modes import get_mode_budget, enforce_mode

def run_pipeline(cv_img, pil_img, models, executor, mode="steady"):
    """
    Unified OCR pipeline with mode-aware budgeting:
      - Phase 1: EAST + Tesseract in parallel (respects mode budget)
      - Phase 2: Conditional EasyOCR cascade if Phase 1 is unreliable
      - Mode enforcement: fast / steady / extended

    Returns a dict with final result and logs.
    """
    pipeline_start = time.perf_counter()
    mode_budget = get_mode_budget(mode)

    # Phase 1 with mode-aware budget
    phase1 = run_phase1_parallel(cv_img, pil_img, executor, budget=mode_budget)
    print_phase1_log(phase1)

    # Decide whether to stop or continue
    if phase1["tess_result"]["isReliable"] or phase1["elapsed"] >= mode_budget:
        final_result = phase1["tess_result"]
        case_triggered = "phase1"
    else:
        # Phase 2 gets remaining budget from mode
        remaining_budget = max(0.5, mode_budget - phase1["elapsed"])
        race_log = run_phase2_conditional(cv_img, pil_img, models,
                                          phase1["east_result"], executor,
                                          budget=remaining_budget)
        print_phase2_log(race_log)
        final_result = race_log["final_result"]
        case_triggered = f"phase2_case{race_log['case_triggered']}"

    # Enforce mode timing (pad or cap as needed)
    total_runtime = enforce_mode(mode, pipeline_start)

    return {
        "final_result": final_result,
        "case_triggered": case_triggered,
        "total_runtime": total_runtime,
        "mode": mode
    }

def print_pipeline_log(pipeline_result):
    """
    Pretty-print the unified pipeline result.
    Expects the dict returned by run_pipeline().
    """
    final = pipeline_result["final_result"]
    case_triggered = pipeline_result["case_triggered"]
    total_runtime = pipeline_result["total_runtime"]
    mode = pipeline_result["mode"]

    print("\n📊 Pipeline Summary")
    print(f"🔎 Final OCR result ({case_triggered}): "
          f"{final.get('text','')} "
          f"(conf={final.get('confidence',0.0)})")
    print(f"⏱️ Total pipeline runtime: {total_runtime}s (mode={mode})")