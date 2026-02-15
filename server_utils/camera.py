import cv2
import sys

def init_camera(source=0, width=1280, height=720):

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Could not open camera source: {source}")
        sys.exit(1)

    # Only set width/height if source is a local device index
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap
