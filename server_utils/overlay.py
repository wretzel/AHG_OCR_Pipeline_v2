import cv2
import numpy as np

def draw_rounded_box(img, pt1, pt2, radius=10, color=(0,0,0), alpha=0.6):
    """Draw a rounded rectangle with transparency."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2

    # Filled rounded rectangle
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
    cv2.circle(overlay, (x1+radius, y1+radius), radius, color, -1)
    cv2.circle(overlay, (x2-radius, y1+radius), radius, color, -1)
    cv2.circle(overlay, (x1+radius, y2-radius), radius, color, -1)
    cv2.circle(overlay, (x2-radius, y2-radius), radius, color, -1)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def put_text_shadow(img, text, org, font, scale, color, thickness):
    """Draw text with a soft shadow for readability."""
    x, y = org
    shadow_color = (0, 0, 0)

    # Shadow offsets
    for dx, dy in [(1,1), (2,2), (1,2)]:
        cv2.putText(img, text, (x+dx, y+dy), font, scale, shadow_color, thickness+1, cv2.LINE_AA)

    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def overlay_text_top_center(frame, text, conf):
    display = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = max(0.6, min(1.8, 30 / max(len(text), 1)))
    thickness = max(1, int(scale * 2))

    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (display.shape[1] - w) // 2
    y = h + 30

    # Rounded background box
    display = draw_rounded_box(
        display,
        (x - 20, y - h - 20),
        (x + w + 20, y + 20),
        radius=12,
        color=(0, 0, 0),
        alpha=0.55
    )

    # Text with shadow
    put_text_shadow(display, text, (x, y), font, scale, (0, 255, 0), thickness)

    # Confidence (top-left)
    conf_text = f"Conf: {conf:.2f}"
    put_text_shadow(display, conf_text, (20, 40), font, 0.7, (255, 255, 255), 2)

    return display


def overlay_combined(frame, ocr_text, ocr_conf, voice_lines):
    display = overlay_text_top_center(frame, ocr_text, ocr_conf)
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = display.shape[0] - 40

    for line in reversed(voice_lines):
        scale = 0.9
        thickness = 2
        (w, h), _ = cv2.getTextSize(line, font, scale, thickness)
        x = (display.shape[1] - w) // 2

        # Rounded background
        display = draw_rounded_box(
            display,
            (x - 20, y - h - 20),
            (x + w + 20, y + 20),
            radius=12,
            color=(0, 0, 0),
            alpha=0.55
        )

        # Text with shadow
        put_text_shadow(display, line, (x, y), font, scale, (0, 255, 0), thickness)

        y -= (h + 35)

    return display
