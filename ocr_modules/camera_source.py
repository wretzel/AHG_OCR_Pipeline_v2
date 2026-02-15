# ocr_modules/camera_source.py

import cv2


class CameraSource:
    """
    Unified camera abstraction for:
      - webcam index (0, 1, ...)
      - MJPEG stream URL
    """

    def __init__(self, source):
        """
        source: int (webcam index) or str (URL)
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {source}")

    def read(self):
        """
        Returns a cv2 frame or None.
        """
        if not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
