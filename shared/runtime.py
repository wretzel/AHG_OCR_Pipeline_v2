# shared/runtime.py

import time

def timed_run(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, round(end - start, 3)
