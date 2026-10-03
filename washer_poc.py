"""Washer Presence Inspection - live POC application.

Usage:
  python washer_poc.py                          # live USB camera (index from config.json)
  python washer_poc.py --camera 0               # override camera index
  python washer_poc.py --image some_photo.jpg   # run on a still image instead

Keys:
  R      select hole ROIs  (drag a box around each hole, ENTER after each box, ESC when done)
  T      teach hole size   (at least one hole WITHOUT washer must be in view)
  D      debug overlay on/off (sampling rings + ratio numbers)
  SPACE  freeze / unfreeze
  S      save raw + annotated image to captures/
  F      fullscreen on/off
  Q/ESC  quit
"""
import argparse
import collections
import json
import os
import time

import cv2
import numpy as np

import washer_detector as wd

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
CAPTURE_DIR = os.path.join(HERE, "captures")
WINDOW = "Washer Presence Inspection"

DISPLAY_MAX_W, DISPLAY_MAX_H = 1280, 720   # camera image area on screen
PANEL_W = 420
SMOOTH_FRAMES = 5                           # majority vote over last N frames (anti-flicker)

# BGR colours
COLORS = {wd.PRESENT: (0, 200, 0), wd.MISSING: (0, 0, 230),
          wd.CHECK: (0, 200, 255), wd.NO_HOLE: (0, 200, 255)}
WHITE, GREY, DARK = (255, 255, 255), (170, 170, 170), (35, 35, 35)
FONT = cv2.FONT_HERSHEY_SIMPLEX

DEFAULT_CONFIG = {"camera_index": 1, "width": 1920, "height": 1080, "mode": "auto",
                  "rois": [], "hole_radius": None}


# --- config -----------------------------------------------------------------------
def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# --- camera -----------------------------------------------------------------------
def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {index}. Run step00_find_cameras.py")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # needed for 1080p at speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(f"Camera {index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


# --- helpers ----------------------------------------------------------------------
def fit_scale(frame):
    h, w = frame.shape[:2]
    return min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)


def select_rois(frame):
    """Let the user drag boxes around holes; returns ROIs in full-resolution pixels."""
    s = fit_scale(frame)
    small = cv2.resize(frame, None, fx=s, fy=s)
    win = "Select holes: drag box, ENTER = next, ESC = done"
    boxes = cv2.selectROIs(win, small, showCrosshair=True)
    cv2.destroyWindow(win)
    return [[int(v / s) for v in b] for b in boxes if b[2] > 5 and b[3] > 5]


def text(img, s, org, scale, color, thick=2):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def majority(history):
    return collections.Counter(history).most_common(1)[0][0]


# --- drawing ----------------------------------------------------------------------
def draw_overlay(frame, rois, results, statuses, hole_r, debug):
    vis = frame.copy()
    t = max(2, frame.shape[1] // 600)
    fs = frame.shape[1] / 1600
    for i, (roi, res, status) in enumerate(zip(rois, results, statuses), 1):
        color = COLORS[status]
        x, y, w, h = roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, t)
        label = f"H{i}"                    # full status is in the side panel
        (tw, th), _ = cv2.getTextSize(label, FONT, fs, t)
        cv2.rectangle(vis, (x, y - th - 12), (x + tw + 10, y), color, -1)
        text(vis, label, (x + 5, y - 6), fs, WHITE, t)
        if res.center is None or not hole_r:
            continue
        c = (int(res.center[0]), int(res.center[1]))
        if debug:
            for k, col in ((wd.RING_IN, color), (wd.RING_OUT, color),
                           (wd.REF_IN, GREY), (wd.REF_OUT, GREY)):
                cv2.circle(vis, c, int(k * hole_r), col, 1, cv2.LINE_AA)
            cv2.circle(vis, c, 3, color, -1)
            text(vis, f"ratio {res.ratio:.2f}", (x + 5, y + h + int(30 * fs)), fs, color, t)
        else:
            cv2.circle(vis, c, int(1.55 * hole_r), color, t, cv2.LINE_AA)
    return vis


def draw_panel(height, statuses, cfg, info):
    p = np.full((height, PANEL_W, 3), DARK, np.uint8)
    text(p, "WASHER PRESENCE", (20, 45), 0.95, WHITE)
    text(p, "INSPECTION", (20, 85), 0.95, WHITE)
    cv2.line(p, (20, 105), (PANEL_W - 20, 105), GREY, 1)

    y = 150
    if not cfg["rois"]:
        text(p, "SETUP: press R to", (20, y), 0.7, COLORS[wd.CHECK])
        text(p, "select hole(s)", (20, y + 32), 0.7, COLORS[wd.CHECK])
    elif not cfg["hole_radius"]:
        text(p, "SETUP: press T to teach", (20, y), 0.7, COLORS[wd.CHECK])
        text(p, "(show 1 hole w/o washer)", (20, y + 32), 0.6, COLORS[wd.CHECK])
    else:
        for i, s in enumerate(statuses, 1):
            cv2.circle(p, (35, y - 9), 12, COLORS[s], -1)
            text(p, f"Hole {i}: {s}", (60, y), 0.8, WHITE)
            y += 45
            if y > height - 200:
                text(p, "...", (60, y), 0.8, WHITE)
                break

    # overall result box
    if cfg["rois"] and cfg["hole_radius"]:
        ok = all(s == wd.PRESENT for s in statuses)
        color, word = (COLORS[wd.PRESENT], "OK") if ok else (COLORS[wd.MISSING], "NG")
        n_ok = sum(s == wd.PRESENT for s in statuses)
        top = height - 200
        cv2.rectangle(p, (20, top), (PANEL_W - 20, top + 110), color, -1)
        text(p, "Overall Result", (35, top + 30), 0.65, WHITE)
        text(p, word, (35, top + 95), 2.0, WHITE, 5)
        text(p, f"{n_ok}/{len(statuses)} present", (200, top + 90), 0.65, WHITE)

    footer = (info, "R roi   T teach   D debug   F fullscreen", "SPACE freeze   S save   Q quit")
    for j, line in enumerate(footer):
        text(p, line, (20, height - 62 + j * 22), 0.45, GREY, 1)
    return p


# --- main loop --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camera", type=int, help="camera index (overrides config.json)")
    ap.add_argument("--image", help="inspect a still image instead of the camera")
    args = ap.parse_args()

    cfg = load_config()
    if args.camera is not None:
        cfg["camera_index"] = args.camera

    cap = None
    still = None
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            raise SystemExit(f"Cannot read image {args.image}")
    else:
        cap = open_camera(cfg["camera_index"], cfg["width"], cfg["height"])

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    history = []
    frozen, debug, fullscreen = False, False, False
    frame, message, msg_until = None, "", 0.0
    t_prev, fps = time.time(), 0.0

    while True:
        if not frozen or frame is None:
            if still is not None:
                frame = still.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame grab failed")
                    break

        gray = wd.to_gray(frame)
        rois, hole_r = cfg["rois"], cfg["hole_radius"]
        if len(history) != len(rois):
            history = [collections.deque(maxlen=SMOOTH_FRAMES) for _ in rois]

        results = [wd.inspect_hole(gray, roi, hole_r) if hole_r else wd.HoleResult(wd.CHECK)
                   for roi in rois]
        for h, r in zip(history, results):
            h.append(r.status)
        statuses = [majority(h) for h in history]

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        vis = draw_overlay(frame, rois, results, statuses, hole_r, debug)
        s = fit_scale(vis)
        vis = cv2.resize(vis, None, fx=s, fy=s)
        if frozen:
            text(vis, "FROZEN", (15, 40), 1.1, COLORS[wd.CHECK], 3)
        if message and now < msg_until:
            cv2.rectangle(vis, (0, vis.shape[0] - 45), (vis.shape[1], vis.shape[0]), DARK, -1)
            text(vis, message, (15, vis.shape[0] - 15), 0.7, WHITE)
        panel = draw_panel(vis.shape[0], statuses, cfg, f"{fps:4.1f} fps")
        cv2.imshow(WINDOW, np.hstack([vis, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            cfg["rois"] = select_rois(frame)
            save_config(cfg)
            message, msg_until = f"{len(cfg['rois'])} ROI(s) saved - now press T", now + 4
        elif key == ord("t"):
            if not cfg["rois"]:
                message, msg_until = "Select ROIs first (R)", now + 3
            else:
                r, message = wd.teach_hole_radius(gray, cfg["rois"])
                msg_until = now + 5
                if r:
                    cfg["hole_radius"] = round(float(r), 2)
                    save_config(cfg)
                print(message)
        elif key == ord("d"):
            debug = not debug
        elif key == ord(" "):
            frozen = not frozen
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(CAPTURE_DIR, f"{stamp}_raw.png"), frame)
            cv2.imwrite(os.path.join(CAPTURE_DIR, f"{stamp}_result.png"), np.hstack([vis, panel]))
            message, msg_until = f"Saved captures/{stamp}_*.png", now + 3

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break                                           # window closed with X

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
