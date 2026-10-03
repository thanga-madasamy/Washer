"""Step 0 - Probe camera indices 0..5 and report which ones deliver frames."""
import cv2

for index in range(6):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"index {index}: not available")
        continue
    ok, frame = cap.read()
    if ok:
        h, w = frame.shape[:2]
        print(f"index {index}: OK  {w}x{h}")
    else:
        print(f"index {index}: opened but no frame")
    cap.release()
