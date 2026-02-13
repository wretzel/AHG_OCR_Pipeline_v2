# ocr_modules/base_modules/ocr_engines.py

import cv2
import numpy as np
import pytesseract
from PIL import Image

from ocr_modules.base_modules.initialization import initialize_models
from ocr_modules.base_modules.east_boxes import decode_predictions
from ocr_modules.base_modules.preprocess import fast_preprocess_bgr, normalize_to_rgb
from ocr_modules.base_modules.parsers import (
    parse_tesseract_output,
    parse_easyocr_output,
    parse_paddleocr_output,
    parse_east_output,
)
import os
import cv2
from ocr_modules.base_modules.parsers import parse_east_output
from ocr_modules.base_modules.initialization import initialize_models
import cv2
import numpy as np
from ocr_modules.base_modules.preprocess import crop_regions, fast_preprocess_bgr
from ocr_modules.base_modules.parsers import parse_paddleocr_output
from ocr_modules.base_modules.east_boxes import decode_predictions
from ocr_modules.base_modules.east_boxes import merge_horizontal_boxes
from ocr_modules.base_modules.corpus_score import corpus_score
_models = None

def load_ocr_models(force_reload=False):
    """Load OCR models once and reuse them."""
    global _models
    if _models is None or force_reload:
        _models = initialize_models()
    return _models


def run_tesseract(image):
    """Run Tesseract OCR and parse results."""
    raw = pytesseract.image_to_data(
        image, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    return parse_tesseract_output(raw)


def run_easyocr(image, lang="en"):
    """Run EasyOCR with language-specific reader."""
    models = load_ocr_models()
    reader = models.get(f"easyocr_{lang}")
    if not reader:
        raise ValueError(f"EasyOCR model for '{lang}' not initialized.")
    return run_easyocr_with_reader(image, reader)

def run_easyocr_with_reader(image, reader, min_token_conf=0.6):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = fast_preprocess_bgr(image, max_side=1280)
    raw = reader.readtext(image, detail=1, paragraph=False)

    # ✅ Call the parser with min_token_conf
    return parse_easyocr_output(raw, min_token_conf=min_token_conf)

def run_paddleocr(image, reader, east_result=None):
    """Run PaddleOCR with preprocessing, region cropping, and error handling."""
    try:
        # Ensure numpy BGR
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
        else:
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Collect crops if EAST regions exist
        crops = crop_regions(image_bgr, east_result) if east_result else [image_bgr]

        all_results = []
        for crop in crops:
            if crop.size == 0:
                continue
            # optional preprocessing for contrast/size
            crop_proc = fast_preprocess_bgr(crop)
            results = reader.ocr(crop_proc)
            if results:
                all_results.extend(results)

        if not all_results:
            return {
                "text": "",
                "confidence": 0.0,
                "corpus_score": 0.0,
                "reliable": False,
            }

        return parse_paddleocr_output(all_results)

    except Exception as e:
        return {
            "error": f"PaddleOCR failed: {str(e)}",
            "text": "",
            "confidence": 0.0,
            "corpus_score": 0.0,
            "reliable": False,
        }

from ocr_modules.base_modules.east_boxes import merge_horizontal_boxes
# If you place cluster_by_baseline and expand_boxes in east_boxes.py, import them similarly:
from ocr_modules.base_modules.east_boxes import cluster_by_baseline, expand_boxes

from ocr_modules.base_modules.east_boxes import sort_regions_by_reading_order

def run_east(image):
    """Run EAST text detector and return annotated regions (parsed)."""
    models = load_ocr_models()
    east_net = models.get("east")

    target_size = 640
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (target_size, target_size)),
        1.0,
        (target_size, target_size),
        (123.68, 116.78, 103.94),
        True,
        False,
    )
    east_net.setInput(blob)
    scores, geometry = east_net.forward(
        ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    )
    raw_boxes, raw_confidences = decode_predictions(scores, geometry)

    # NMS
    rects = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in raw_boxes]
    indices = cv2.dnn.NMSBoxes(rects, raw_confidences, score_threshold=0.45, nms_threshold=0.2)

    scaled_boxes = []
    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        (x, y, w, h) = rects[i]
        box = [
            int(x * (image.shape[1] / target_size)),
            int(y * (image.shape[0] / target_size)),
            int((x + w) * (image.shape[1] / target_size)),
            int((y + h) * (image.shape[0] / target_size)),
        ]
        conf = raw_confidences[i]
        scaled_boxes.append({"box": box, "confidence": round(conf, 2)})

    # Merge + cluster + expand
    merged_h = merge_horizontal_boxes(scaled_boxes, y_tolerance=25, x_gap=30)
    merged_lines = cluster_by_baseline(merged_h, y_tolerance=0.3)
    expanded = expand_boxes(
        merged_lines,
        pad_frac_x=0.12,
        pad_frac_y=0.25,
        min_pad=10,
        image_shape=image.shape
    )

    # Annotate for debugging
    annotated = image.copy()
    for r in scaled_boxes:
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 1)
    for r in expanded:
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out_dir = os.path.join("testing", "test_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "east_output.png")
    if annotated is not None and annotated.size > 0:
        ok = cv2.imwrite(out_path, annotated)
        if not ok:
            print(f"⚠️ cv2.imwrite failed for {out_path}")
    else:
        print("⚠️ Annotated image empty, skipping save.")

    # ✅ Normalize reading order before returning
    ordered = sort_regions_by_reading_order(expanded)

    return parse_east_output({"regions": ordered, "region_count": len(ordered)})
