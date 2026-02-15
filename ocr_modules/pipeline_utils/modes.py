# ocr_modules/pipeline_utils/modes.py

import time

MODES = {
    "fast": {"budget": 1.0, "min_interval": 0.0},        # cap at 1s, finish ASAP
    "steady": {"budget": 5.0, "min_interval": 2.0},      # Consistent rhythm, allows up to 5s
    "extended": {"budget": 9999.0, "min_interval": 10.0}    # no max per image, but wait 10s between cycles
}

def get_mode_budget(mode_name):
    mode = MODES.get(mode_name, MODES["steady"])
    return mode["budget"]

def enforce_mode(mode_name, start_time):
    mode = MODES.get(mode_name, MODES["steady"])
    elapsed = time.perf_counter() - start_time

    # Cap runtime if budget is set
    if mode["budget"] is not None and elapsed > mode["budget"]:
        return round(mode["budget"], 3)

    # Pad runtime if min_interval is set
    if mode["min_interval"] > 0 and elapsed < mode["min_interval"]:
        sleep_time = mode["min_interval"] - elapsed
        time.sleep(sleep_time)
        return round(mode["min_interval"], 3)

    return round(elapsed, 3)