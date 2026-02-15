# ocr_modules/base_modules/east_boxes.py

import cv2
import numpy as np

from shared.helper import normalize_conf

def decode_predictions(scores, geometry, conf_threshold=0.5):
    num_rows, num_cols = scores.shape[2:4]
    boxes = []
    confidences = []

    for y in range(num_rows):
        for x in range(num_cols):
            score = scores[0, 0, y, x]
            if score < conf_threshold:
                continue

            # Geometry data
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos, sin = np.cos(angle), np.sin(angle)

            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]

            end_x = int(offset_x + cos * geometry[0, 1, y, x] + sin * geometry[0, 2, y, x])
            end_y = int(offset_y - sin * geometry[0, 1, y, x] + cos * geometry[0, 2, y, x])
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            boxes.append((start_x, start_y, end_x, end_y))
            confidences.append(float(score))

    return boxes, confidences

def detect_text_east(image, model_path="resources/east_model.pb", net=None, conf_threshold=0.5, nms_threshold=0.4):
    if net is None:
        net = cv2.dnn.readNet(model_path)

    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (320, 320))
    rW, rH = orig_w / 320.0, orig_h / 320.0

    blob = cv2.dnn.blobFromImage(resized, 1.0, (320, 320),
                                 (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                    "feature_fusion/concat_3"])

    raw_boxes, confidences = decode_predictions(scores, geometry, conf_threshold)

    # Convert to OpenCV-style rectangles for NMS
    rects = []
    for (start_x, start_y, end_x, end_y) in raw_boxes:
        rects.append([start_x, start_y, end_x - start_x, end_y - start_y])

    indices = cv2.dnn.NMSBoxes(rects, confidences, conf_threshold, nms_threshold)

    final_boxes = []
    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        (x, y, w, h) = rects[i]
        final_boxes.append([
            int(x * rW),
            int(y * rH),
            int((x + w) * rW),
            int((y + h) * rH)
        ])

    return final_boxes

def merge_horizontal_boxes(regions, y_tolerance=15, x_gap=60):
    merged = []
    for region in sorted(regions, key=lambda r: r["box"][0]):  # sort by x1
        x1, y1, x2, y2 = region["box"]
        merged_flag = False
        for m in merged:
            mx1, my1, mx2, my2 = m["box"]
            if abs(y1 - my1) < y_tolerance and x1 <= mx2 + x_gap:
                m["box"] = [min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2)]
                m["confidence"] = max(m["confidence"], region["confidence"])
                merged_flag = True
                break
        if not merged_flag:
            merged.append(region)
    return merged

def cluster_by_baseline(regions, y_tolerance=0.3):
    if not regions:
        return []

    # Compute median height to scale tolerance
    heights = [(r["box"][3] - r["box"][1]) for r in regions]
    med_h = max(1, int(np.median(heights)))
    tol = int(med_h * y_tolerance)

    # Sort by x1 for stable left-to-right grouping
    regions_sorted = sorted(regions, key=lambda r: r["box"][0])

    lines = []
    for r in regions_sorted:
        x1, y1, x2, y2 = r["box"]
        yc = (y1 + y2) // 2
        placed = False
        for line in lines:
            ly1, ly2 = line["box"][1], line["box"][3]
            lyc = (ly1 + ly2) // 2
            if abs(yc - lyc) <= tol:
                # Merge into the line span
                lx1, ly1, lx2, ly2 = line["box"]
                line["box"] = [min(lx1, x1), min(ly1, y1), max(lx2, x2), max(ly2, y2)]
                line["confidence"] = max(line["confidence"], r["confidence"])
                placed = True
                break
        if not placed:
            lines.append({"box": [x1, y1, x2, y2], "confidence": r["confidence"]})
    return lines

def expand_boxes(boxes, pad_frac_x=0.12, pad_frac_y=0.18, min_pad=10, image_shape=None):
    expanded = []
    H, W = image_shape[:2]
    for r in boxes:
        x1, y1, x2, y2 = r["box"]
        w, h = x2 - x1, y2 - y1
        pad_x = max(int(w * pad_frac_x), min_pad)
        pad_y = max(int(h * pad_frac_y), min_pad)
        new_box = [
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(W, x2 + pad_x),
            min(H, y2 + pad_y),
        ]
        expanded.append({"box": new_box, "confidence": r.get("confidence", 1.0)})
    return expanded

# ocr_modules/base_modules/east_boxes.py

def sort_regions_by_reading_order(regions):
    return sorted(regions, key=lambda r: (r["box"][1], r["box"][0]))

import cv2
import numpy as np
from PIL import Image

def visualize_region_ocr_debug(cv_img, regions, ocr_fn, engine="easyocr", min_conf=0.0):
    debug_img = cv_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r["box"]
        crop = cv_img[y1:y2, x1:x2]

        # Run OCR depending on engine type
        try:
            if engine == "easyocr":
                results = ocr_fn(crop)  # EasyOCR expects np.ndarray
            elif engine == "tesseract":
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                results = ocr_fn(pil_crop)  # Tesseract expects PIL.Image
            else:
                raise ValueError(f"Unsupported engine: {engine}")
        except Exception as e:
            results = []
            print(f"⚠️ OCR failed on region {i}: {e}")

        # Normalize output
        texts = [t[1] for t in results if normalize_conf(t[2]) >= min_conf]
        confs = [normalize_conf(t[2]) for t in results if normalize_conf(t[2]) >= min_conf]
        avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
        label = f"R{i}: {' '.join(texts)} ({avg_conf})"

        # Draw box and label
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(debug_img, label, (x1, y1 - 10), font, 0.5, (255, 255, 0), 1)

    return debug_img
