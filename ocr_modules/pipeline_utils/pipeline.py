# ocr_modules/pipeline_utils/pipeline.py

import time
import traceback
from shared.helper import normalize_conf  # safe float caster
from ocr_modules.pipeline_utils.phase1 import run_phase1_parallel, print_phase1_log
from ocr_modules.pipeline_utils.phase2 import run_phase2_conditional, print_phase2_log
from ocr_modules.pipeline_utils.modes import get_mode_budget, enforce_mode

def run_pipeline(cv_img, pil_img, models, executor, mode="steady"):

    pipeline_start = time.perf_counter()
    mode_budget = get_mode_budget(mode)

    try:
        # Phase 1 with mode-aware budget (pass models so workers reuse preloaded models)
        phase1 = run_phase1_parallel(cv_img, pil_img, executor, budget=mode_budget, models=models)
        print_phase1_log(phase1)

        # Defensive reads
        tess = phase1.get("tess_result", {}) or {}
        east = phase1.get("east_result", {}) or {}
        elapsed = phase1.get("elapsed", 0.0) or 0.0

        # Log normalized values to catch None early
        tess_conf = normalize_conf(tess.get("confidence"))
        tess_is_rel = bool(tess.get("isReliable", False))
        east_region_count = normalize_conf(east.get("region_count"), 0)

        print(f"🔧 Phase1 debug: tess_conf={tess_conf}, tess_isReliable={tess_is_rel}, "
              f"east_region_count={east_region_count}, elapsed={elapsed}, budget={mode_budget}")

        # Decide whether to stop or continue
        if tess_is_rel or elapsed >= mode_budget:
            final_result = tess
            case_triggered = "phase1"
        else:
            remaining_budget = max(0.5, mode_budget - elapsed)
            race_log = run_phase2_conditional(
                cv_img, pil_img, models, east, executor, budget=remaining_budget
            )
            print_phase2_log(race_log)
            final_result = race_log.get("final_result") or {"text": "", "confidence": 0.0, "reliable": False}
            case_triggered = f"phase2_case{race_log.get('case_triggered')}"

        # Apply reliability filter
        conf = normalize_conf(final_result.get("confidence"))
        rel = bool(final_result.get("reliable", False))
        if not rel or conf < 0.5:
            final_result = {"text": "", "confidence": conf, "reliable": False}

        # Enforce mode timing
        total_runtime = enforce_mode(mode, pipeline_start)

        return {
            "final_result": final_result,
            "case_triggered": case_triggered,
            "total_runtime": total_runtime,
            "mode": mode
        }


    except Exception as e:
        print(f"❌ Pipeline exception: {e}")
        traceback.print_exc()
        # Return a safe failure payload so the caller can keep going
        return {
            "final_result": {"text": "", "confidence": 0.0, "reliable": False, "error": str(e)},
            "case_triggered": "exception",
            "total_runtime": round(time.perf_counter() - pipeline_start, 3),
            "mode": mode
        }


def print_pipeline_log(pipeline_result):

    final = pipeline_result.get("final_result") or {}
    case_triggered = pipeline_result.get("case_triggered")
    total_runtime = pipeline_result.get("total_runtime")
    mode = pipeline_result.get("mode")

    text = (final.get("text") or "").strip()
    conf = normalize_conf(final.get("confidence"))
    rel = bool(final.get("reliable", False))
    err = final.get("error")

    print("\n📊 Pipeline Summary")
    print(f"🔎 Final OCR result ({case_triggered}): {text} (conf={conf}, reliable={rel})")
    if err:
        print(f"⚠️ Error: {err}")
    print(f"⏱️ Total pipeline runtime: {total_runtime}s (mode={mode})")
