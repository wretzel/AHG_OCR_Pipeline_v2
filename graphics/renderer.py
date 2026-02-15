# graphics/renderer.py

import cv2
import numpy as np
from . import theme


def _put_text_with_box(
    frame,
    text_lines,
    anchor,
    font_scale,
    thickness,
    text_color,
    box_color,
    alpha,
    padding,
    line_spacing,
):
    """
    Draw multi-line text with a semi-transparent box.
    anchor: (x_center, y_bottom) for subtitles,
            or (x_left, y_top) depending on usage.
    """
    if not text_lines:
        return frame

    h, w, _ = frame.shape
    font = theme.FONT_FACE

    # Measure text block
    line_sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines
    ]
    text_width = max(size[0] for size in line_sizes)
    text_height = sum(size[1] for size in line_sizes) + line_spacing * (len(text_lines) - 1)

    # Compute box coordinates (anchor is bottom-center)
    x_center, y_bottom = anchor
    x1 = int(x_center - text_width / 2 - padding)
    x2 = int(x_center + text_width / 2 + padding)
    y2 = int(y_bottom)
    y1 = int(y_bottom - text_height - 2 * padding)

    # Clamp to frame
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    # Draw box with alpha
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw text lines
    y_text = y1 + padding
    for i, line in enumerate(text_lines):
        size = line_sizes[i]
        x_text = int(x_center - size[0] / 2)
        y_text_baseline = y_text + size[1]
        cv2.putText(
            frame,
            line,
            (x_text, y_text_baseline),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        y_text += size[1] + line_spacing

    return frame


def draw_subtitle_block(frame, text, position="bottom"):
    """
    Draw a subtitle-style block (for voice).
    position: 'bottom' or 'top'
    """
    if not text:
        return frame

    h, w, _ = frame.shape
    max_width = int(w * theme.MAX_WIDTH_RATIO)

    # Simple word-wrapping
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        size, _ = cv2.getTextSize(
            test, theme.FONT_FACE, theme.FONT_SCALE_SUBTITLE, theme.THICKNESS_SUBTITLE
        )
        if size[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    if position == "bottom":
        anchor = (w // 2, h - theme.MARGIN_BOTTOM)
    else:
        anchor = (w // 2, theme.MARGIN_TOP + 80)

    return _put_text_with_box(
        frame,
        lines,
        anchor,
        font_scale=theme.FONT_SCALE_SUBTITLE,
        thickness=theme.THICKNESS_SUBTITLE,
        text_color=theme.COLOR_TEXT,
        box_color=theme.COLOR_BOX,
        alpha=theme.BOX_ALPHA,
        padding=theme.PADDING,
        line_spacing=theme.LINE_SPACING,
    )


def draw_ocr_block(frame, text):
    """
    Draw OCR text block near the top.
    """
    if not text:
        return frame

    h, w, _ = frame.shape
    max_width = int(w * theme.MAX_WIDTH_RATIO)

    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        size, _ = cv2.getTextSize(
            test, theme.FONT_FACE, theme.FONT_SCALE_OCR, theme.THICKNESS_OCR
        )
        if size[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    anchor = (w // 2, theme.MARGIN_TOP + 20)

    return _put_text_with_box(
        frame,
        lines,
        anchor,
        font_scale=theme.FONT_SCALE_OCR,
        thickness=theme.THICKNESS_OCR,
        text_color=theme.COLOR_OCR_TEXT,
        box_color=theme.COLOR_OCR_BOX,
        alpha=theme.BOX_ALPHA,
        padding=theme.PADDING,
        line_spacing=theme.LINE_SPACING,
    )
