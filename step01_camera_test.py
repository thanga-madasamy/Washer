"""Step 1 - Verify the USB camera works with OpenCV.

Keys:  s = save current frame to captures/   q = quit
"""
import os
import time
import cv2

CAMERA_INDEX = 1          # 0 = laptop webcam, 1 = USB camera (from step00_find_cameras.py)
REQ_WIDTH, REQ_HEIGHT = 1920, 1080

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit(f"Cannot open camera index {CAMERA_INDEX}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_HEIGHT)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera {CAMERA_INDEX} opened, actual resolution {w}x{h}")

CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed")
        break

    cv2.imshow("Camera test  (s=save, q=quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        path = os.path.join(CAPTURE_DIR, time.strftime("frame_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, frame)
        print("Saved", path)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
