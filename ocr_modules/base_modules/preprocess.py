# ocr_modules/base_modules/preprocess.py

import cv2
import numpy as np
from PIL import Image
from shared.helper import normalize_conf  # central safe float caster

def normalize_to_rgb(image):

    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # grayscale
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            # Assume BGR (OpenCV default) → convert to RGB
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Fallback: return as-is
    return image


def fast_preprocess_bgr(image, max_side=1280):

    if not isinstance(image, np.ndarray) or image.ndim < 2:
        raise ValueError("fast_preprocess_bgr expects a BGR ndarray")

    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_regions(image, east_result, max_regions=10, base_padding=10):
    if not isinstance(image, np.ndarray):
        raise ValueError("crop_regions expects a BGR ndarray")

    crops = []
    h, w = image.shape[:2]

    for i, region in enumerate(east_result.get("regions", [])[:max_regions]):
        box = region.get("box")
        if not box or len(box) != 4:
            continue

        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except Exception:
            continue

        # Dynamic padding: scale with box size
        pad_x = max(base_padding, int(0.2 * (x2 - x1)))
        pad_y = max(base_padding, int(0.2 * (y2 - y1)))

        # Apply padding safely
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Skip tiny boxes
        if (x2 - x1) < 40 or (y2 - y1) < 20:
            continue

        if isinstance(image, np.ndarray):
            crop = image[y1:y2, x1:x2]
            crops.append(crop)

        # Debug log
        print(f"Region {i}: original={box}, padded=({x1},{y1},{x2},{y2}), size={x2-x1}x{y2-y1}")

    return crops


def aggregate_crop_results(
    crops,
    runner_fn,
    reader=None,
    conf_threshold=0.6,
    min_crop_conf=0.4,
    verbose=False
):

    if not callable(runner_fn):
        raise TypeError(f"runner_fn must be callable, got {type(runner_fn)}")

    texts, confs, details = [], [], []

    for i, crop in enumerate(crops):
        try:
            res = runner_fn(crop, reader) if reader is not None else runner_fn(crop)
            text = (res.get("text") or "").strip()
            conf = normalize_conf(res.get("confidence"))

            details.append({"index": i, "text": text, "confidence": conf})

            if verbose:
                print(f"🧪 Crop {i}: '{text}' (conf={conf})")

            if text and conf >= min_crop_conf:
                texts.append(text)
                confs.append(conf)
            elif verbose and text:
                print(f"⚠️ Crop {i} skipped (conf={conf})")

        except Exception as e:
            details.append({"index": i, "error": str(e)})
            if verbose:
                print(f"❌ OCR failed on crop {i}: {e}")

    merged_text = " ".join(texts).strip()
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    return {
        "text": merged_text,
        "confidence": round(avg_conf, 2),
        "reliable": normalize_conf(avg_conf) >= conf_threshold,
        "details": details
    }
