# shared/json_utils.py

import numpy as np
import json
import os
from datetime import datetime

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

def save_json(data, path, timestamped=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(sanitize_for_json(data), f, indent=2)

    if timestamped:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{os.path.splitext(path)[0]}_{stamp}.json"
        with open(backup, "w") as f:
            json.dump(sanitize_for_json(data), f, indent=2)
        return path, backup

    return path, None
