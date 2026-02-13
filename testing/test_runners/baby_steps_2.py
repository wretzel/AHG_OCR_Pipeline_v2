# testing/test_runners/baby_steps_2.py
import cv2
import time
from datetime import datetime

def add_timestamp(frame, label="Captured"):
    """Overlay current timestamp on the frame."""
    display = frame.copy()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # millisecond precision
    cv2.putText(display, f"{label}: {ts}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return display

def get_fresh_frame(cap):
    """
    Force retrieval of the freshest frame by flushing buffer.
    """
    # Try to set buffer size to 1 (not supported everywhere)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Grab and discard buffered frames
    for _ in range(3):
        cap.grab()

    # Retrieve the most recent frame
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    print("📸 Capturing one fresh frame every 3 seconds. Press 'q' to quit.")

    while True:
        frame = get_fresh_frame(cap)
        if frame is None:
            print("❌ Failed to read frame")
            break

        # Flip frame to correct mirroring
        frame = cv2.flip(frame, 1)

        # Add timestamp overlay
        display = add_timestamp(frame, label="Captured")

        # Show snapshot
        cv2.imshow("Snapshot Viewer", display)

        # Wait for 3 seconds, but keep window responsive
        start = time.time()
        while time.time() - start < 3.0:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
