# graphics/overlay.py

from .renderer import draw_subtitle_block, draw_ocr_block


class OverlayEngine:
    """
    Simple overlay engine that composes:
      - voice subtitles (bottom)
      - OCR text (top)
    """

    def __init__(self):
        self.latest_voice = ""
        self.latest_ocr = ""

    def update_voice(self, text: str):
        self.latest_voice = text or ""

    def update_ocr(self, text: str):
        self.latest_ocr = text or ""

    def render(self, frame):
        """
        Draw overlay onto the given frame (in-place).
        """
        if self.latest_ocr:
            frame = draw_ocr_block(frame, self.latest_ocr)

        if self.latest_voice:
            frame = draw_subtitle_block(frame, self.latest_voice, position="bottom")

        return frame
