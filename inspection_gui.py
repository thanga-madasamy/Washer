"""Washer Presence Inspection - desktop front end (Tkinter).

Two inspection modes:
  AUTO  (default) - every flange hole in view is found and checked automatically,
                    no setup. Needs >= 2 neighbouring holes in view (hole spacing = scale).
  TEACH (backup)  - draw a box around specific holes, teach the bare-hole size once.

  - Start / stop the USB camera, or open a still image
  - Live PRESENT / MISSING per hole, overall OK / NG, event log (also saved to logs/*.csv)
"""
import collections
import csv
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import washer_detector as wd
import washer_poc as core

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
VIEW_W, VIEW_H = 1100, 680

BG, PANEL, FG, MUTED = "#1e1e1e", "#2a2a2a", "#f0f0f0", "#9a9a9a"
HEX = {wd.PRESENT: "#1faa3a", wd.MISSING: "#e02020", wd.CHECK: "#f0a800", wd.NO_HOLE: "#f0a800"}


class CameraGrabber:
    """Reads frames - and runs AUTO inspection on them - in a background thread,
    so the UI never waits for the camera or the image analysis."""

    def __init__(self, cap, analyse):
        self.cap, self.analyse, self.item, self.running = cap, analyse, None, True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                results = self.analyse(frame)
                with self.lock:
                    self.item = (frame, results)

    def latest(self):
        """(frame, auto_results or None) - each frame is handed out once, else None."""
        with self.lock:
            item, self.item = self.item, None
        return item

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)        # let the last read() finish before release
        self.cap.release()


class InspectionApp:
    def __init__(self, root):
        self.root = root
        self.cfg = core.load_config()
        self.cap = None
        self.pending_cap = None
        self.still = None
        self.frame = None
        self.scale = 1.0
        self.history = []
        self.prev_statuses = []
        self.chips = []
        self.tracks = []                 # AUTO mode: holes followed from frame to frame
        self.prev_overall = None
        self.drag_start = None
        self.drag_item = None
        self.photo = None
        self.t_prev, self.fps = time.time(), 0.0

        self.debug = tk.BooleanVar(value=False)
        self.frozen = tk.BooleanVar(value=False)
        self.editing = tk.BooleanVar(value=False)
        self.cam_index = tk.StringVar(value=str(self.cfg["camera_index"]))
        self.mode = tk.StringVar(value=self.cfg.get("mode", "auto"))

        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(core.CAPTURE_DIR, exist_ok=True)
        self.log_path = os.path.join(LOG_DIR, time.strftime("inspection_%Y%m%d.csv"))

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.start_camera()
        self.tick()

    # --- UI layout ------------------------------------------------------------------
    def _build_ui(self):
        r = self.root
        global VIEW_W, VIEW_H                      # fit the image area to this screen
        VIEW_W = min(VIEW_W, r.winfo_screenwidth() - 460)
        VIEW_H = min(VIEW_H, r.winfo_screenheight() - 170)
        r.title("Washer Presence Inspection")
        r.configure(bg=BG)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TCheckbutton", background=PANEL, foreground=FG, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("TRadiobutton", background=PANEL, foreground=FG, font=("Segoe UI", 10))
        style.map("TRadiobutton", background=[("active", PANEL)])

        # header
        head = tk.Frame(r, bg=BG)
        head.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(head, text="WASHER PRESENCE INSPECTION", bg=BG, fg=FG,
                 font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(head, text="Phase 1 POC · USB camera · OpenCV", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left", padx=14, pady=(6, 0))

        body = tk.Frame(r, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=4)

        # left: image
        self.canvas = tk.Canvas(body, width=VIEW_W, height=VIEW_H, bg="black",
                                highlightthickness=0, cursor="arrow")
        self.canvas.pack(side="left")
        self.img_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)

        # right: controls + results
        side = tk.Frame(body, bg=PANEL, width=390)
        side.pack(side="left", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        self.overall = tk.Label(side, text="—", bg="#444", fg="white",
                                font=("Segoe UI", 34, "bold"), height=1)
        self.overall.pack(fill="x", padx=12, pady=(10, 0))
        self.overall_sub = tk.Label(side, text="Overall result", bg="#444", fg="white",
                                    font=("Segoe UI", 11))
        self.overall_sub.pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(side, text="HOLES", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold"),
                 anchor="w").pack(fill="x", padx=14)
        self.summary = tk.Label(side, text="", bg=PANEL, fg=FG, font=("Segoe UI", 11), anchor="w")
        self.summary.pack(fill="x", padx=14)
        self.holes_frame = tk.Frame(side, bg=PANEL)
        self.holes_frame.pack(fill="x", padx=12, pady=(4, 6))

        sec = lambda t: tk.Label(side, text=t, bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold"),
                                 anchor="w").pack(fill="x", padx=14, pady=(6, 2))
        sec("SOURCE")
        src = tk.Frame(side, bg=PANEL)
        src.pack(fill="x", padx=12)
        tk.Label(src, text="Camera", bg=PANEL, fg=FG, font=("Segoe UI", 10)).pack(side="left")
        ttk.Combobox(src, textvariable=self.cam_index, values=["0", "1", "2", "3"], width=3,
                     state="readonly").pack(side="left", padx=6)
        self.cam_btn = ttk.Button(src, text="Stop", width=7, command=self.toggle_camera)
        self.cam_btn.pack(side="left", padx=2)
        ttk.Button(src, text="Image…", command=self.open_image).pack(side="left", padx=2)

        sec("MODE")
        modes = tk.Frame(side, bg=PANEL)
        modes.pack(fill="x", padx=12)
        ttk.Radiobutton(modes, text="Auto – all holes in view", value="auto",
                        variable=self.mode, command=self.on_mode).pack(anchor="w")
        ttk.Radiobutton(modes, text="Teach – selected holes only", value="teach",
                        variable=self.mode, command=self.on_mode).pack(anchor="w")
        setup = tk.Frame(side, bg=PANEL)
        setup.pack(fill="x", padx=12, pady=(2, 0))
        self.edit_btn = ttk.Button(setup, text="✏ Edit holes", command=self.toggle_edit)
        self.edit_btn.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        self.clear_btn = ttk.Button(setup, text="Clear holes", command=self.clear_holes)
        self.clear_btn.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        self.teach_btn = ttk.Button(setup, text="Teach hole size", command=self.teach)
        self.teach_btn.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        setup.columnconfigure((0, 1), weight=1)
        self.teach_widgets = (self.edit_btn, self.clear_btn, self.teach_btn)

        sec("VIEW")
        view = tk.Frame(side, bg=PANEL)
        view.pack(fill="x", padx=12)
        ttk.Checkbutton(view, text="Debug rings", variable=self.debug).pack(side="left")
        ttk.Checkbutton(view, text="Freeze", variable=self.frozen).pack(side="left", padx=8)
        ttk.Button(view, text="Snapshot", command=self.save_snapshot).pack(side="right")

        sec("EVENT LOG")
        self.log = tk.Listbox(side, bg="#1b1b1b", fg=FG, font=("Consolas", 9), height=5,
                              borderwidth=0, highlightthickness=0)
        self.log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.status = tk.Label(r, text="", bg=BG, fg=MUTED, font=("Segoe UI", 9), anchor="w")
        self.status.pack(fill="x", padx=12, pady=(0, 8))
        self.on_mode(initial=True)

    # --- helpers ----------------------------------------------------------------------
    def say(self, msg):
        self.status.config(text=msg)

    def add_log(self, text):
        stamp = time.strftime("%H:%M:%S")
        self.log.insert(0, f"{stamp}  {text}")
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), text])

    def to_full(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def on_mode(self, initial=False):
        auto = self.mode.get() == "auto"
        for w in self.teach_widgets:
            w.state(["disabled"] if auto else ["!disabled"])
        if auto and self.editing.get():
            self.toggle_edit()
        self.cfg["mode"] = self.mode.get()
        self.auto = auto                          # read by the camera thread
        self.tracks, self.history, self.prev_statuses, self.prev_overall = [], [], [], None
        if not initial:
            core.save_config(self.cfg)
            self.add_log(f"MODE  {self.mode.get().upper()}")
        self.say("AUTO: point the camera at the flange - every hole in view is checked "
                 "(keep >= 2 neighbouring holes visible)" if auto else
                 "TEACH: ✏ Edit holes -> draw boxes -> Teach hole size (with a bare hole in view)")

    def _analyse_in_thread(self, frame):
        return wd.inspect_auto(wd.to_gray(frame)) if self.auto else None

    # --- source ---------------------------------------------------------------------
    def start_camera(self):
        """Opening a USB camera at 1080p can take ~15 s, so do it off the UI thread."""
        idx = int(self.cam_index.get())
        self.say(f"Opening camera {idx}…  (can take up to ~15 s)")
        self.cam_btn.config(text="…", state="disabled")
        threading.Thread(target=self._open_camera, args=(idx,), daemon=True).start()

    def _open_camera(self, idx):                 # runs in a worker thread: no Tk calls here
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["height"])
        self.pending_cap = (idx, cap)

    def _camera_opened(self, idx, cap):          # called from tick() on the UI thread
        self.cam_btn.config(state="normal")
        if self.still is not None:               # user opened an image while we waited
            cap.release()
            self.cam_btn.config(text="Start")
            return
        if not cap.isOpened():
            self.say(f"Camera {idx} not available - pick another index or open an image")
            self.cam_btn.config(text="Start")
            return
        self.cap, self.still = CameraGrabber(cap, self._analyse_in_thread), None
        self.cfg["camera_index"] = idx
        core.save_config(self.cfg)
        self.cam_btn.config(text="Stop")
        self.say(f"Camera {idx} running at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                 f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cam_btn.config(text="Start")

    def toggle_camera(self):
        if self.cap is not None:
            self.stop_camera()
            self.say("Camera stopped")
        else:
            self.start_camera()

    def open_image(self):
        path = filedialog.askopenfilename(
            initialdir=HERE, filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.say(f"Cannot read {path}")
            return
        self.stop_camera()
        self.still = img
        self.tracks = []
        self.say(f"Image: {os.path.basename(path)}")

    # --- setup actions --------------------------------------------------------------
    def toggle_edit(self):
        self.editing.set(not self.editing.get())
        on = self.editing.get()
        self.edit_btn.config(text="✔ Done editing" if on else "✏ Edit holes")
        self.canvas.config(cursor="crosshair" if on else "arrow")
        if on:
            self.say("Drag a box around each hole (≈4-5× hole diameter). Right-click a box to delete.")
        else:
            core.save_config(self.cfg)
            self.say(f"{len(self.cfg['rois'])} hole(s) saved"
                     + ("" if self.cfg["hole_radius"] else " - now press Teach hole size"))

    def clear_holes(self):
        self.cfg["rois"] = []
        core.save_config(self.cfg)
        self.say("All holes cleared")

    def teach(self):
        if self.frame is None or not self.cfg["rois"]:
            self.say("Define holes first (✏ Edit holes)")
            return
        r, msg = wd.teach_hole_radius(wd.to_gray(self.frame), self.cfg["rois"])
        if r:
            self.cfg["hole_radius"] = round(float(r), 2)
            core.save_config(self.cfg)
            self.add_log(f"TEACH  r = {r:.1f} px")
        self.say(msg)

    def save_snapshot(self):
        if self.frame is None:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(core.CAPTURE_DIR, f"{stamp}_raw.png"), self.frame)
        cv2.imwrite(os.path.join(core.CAPTURE_DIR, f"{stamp}_result.png"), self.last_vis)
        self.say(f"Saved captures/{stamp}_raw.png and _result.png")

    # --- mouse: draw / delete hole boxes ----------------------------------------------
    def on_press(self, e):
        if self.editing.get():
            self.drag_start = (e.x, e.y)
            self.drag_item = self.canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                          outline="#00d0ff", width=2, dash=(4, 2))

    def on_drag(self, e):
        if self.drag_item:
            x0, y0 = self.drag_start
            self.canvas.coords(self.drag_item, x0, y0, e.x, e.y)

    def on_release(self, e):
        if not self.drag_item:
            return
        self.canvas.delete(self.drag_item)
        self.drag_item = None
        (x0, y0), (x1, y1) = self.to_full(*self.drag_start), self.to_full(e.x, e.y)
        x, y, w, h = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
        if w > 10 and h > 10:
            self.cfg["rois"].append([x, y, w, h])
            core.save_config(self.cfg)
            self.say(f"Hole {len(self.cfg['rois'])} added")

    def on_right_click(self, e):
        if not self.editing.get():
            return
        fx, fy = self.to_full(e.x, e.y)
        for i, (x, y, w, h) in enumerate(self.cfg["rois"]):
            if x <= fx <= x + w and y <= fy <= y + h:
                del self.cfg["rois"][i]
                core.save_config(self.cfg)
                self.say(f"Hole {i + 1} deleted")
                return

    # --- results panel --------------------------------------------------------------
    def update_panel(self, statuses, setup_msg=None):
        if len(self.chips) != len(statuses):          # rebuild the hole chip grid
            for w in self.holes_frame.winfo_children():
                w.destroy()
            cols = 6 if len(statuses) > 12 else 4
            self.chips = []
            for i in range(len(statuses)):
                chip = tk.Label(self.holes_frame, text=f"H{i + 1}", fg="white", width=5,
                                font=("Segoe UI", 10, "bold"), pady=3)
                chip.grid(row=i // cols, column=i % cols, padx=2, pady=2, sticky="ew")
                self.chips.append(chip)
            for c in range(cols):
                self.holes_frame.columnconfigure(c, weight=1)
        for chip, s in zip(self.chips, statuses):
            chip.config(bg=HEX[s])

        n = collections.Counter(statuses)
        n_check = n[wd.CHECK] + n[wd.NO_HOLE]
        self.summary.config(text=f"{len(statuses)} inspected  ·  {n[wd.PRESENT]} present  ·  "
                                 f"{n[wd.MISSING]} missing" + (f"  ·  {n_check} check" if n_check else ""))

        if setup_msg:
            text, sub, color = setup_msg[0], setup_msg[1], "#555"
        else:
            ok = n[wd.PRESENT] == len(statuses)
            text, color = ("OK", HEX[wd.PRESENT]) if ok else ("NG", HEX[wd.MISSING])
            sub = f"{n[wd.PRESENT]}/{len(statuses)} washers present"
        self.overall.config(text=text, bg=color)
        self.overall_sub.config(text=sub, bg=color)

        # event log: per hole in TEACH mode (fixed holes), overall result in AUTO mode
        if setup_msg is None and self.mode.get() == "teach":
            if len(statuses) == len(self.prev_statuses):
                for i, (a, b) in enumerate(zip(self.prev_statuses, statuses), 1):
                    if a != b:
                        self.add_log(f"Hole {i}: {a} -> {b}")
            self.prev_statuses = statuses
        elif setup_msg is None and text != self.prev_overall:
            self.add_log(f"Overall {text}  ({sub})")
        if setup_msg is None:
            self.prev_overall = text

    # --- inspection -----------------------------------------------------------------
    def run_teach_mode(self, gray):
        rois, hole_r = self.cfg["rois"], self.cfg["hole_radius"]
        if len(self.history) != len(rois):
            self.history = [collections.deque(maxlen=core.SMOOTH_FRAMES) for _ in rois]
        results = [wd.inspect_hole(gray, roi, hole_r) if hole_r else wd.HoleResult(wd.CHECK)
                   for roi in rois]
        for h, res in zip(self.history, results):
            h.append(res.status)
        statuses = [core.majority(h) for h in self.history]
        vis = core.draw_overlay(self.frame, rois, results, statuses, hole_r, self.debug.get())
        setup = None
        if not rois:
            setup = ("SETUP", "Click ✏ Edit holes and draw boxes")
        elif not hole_r:
            setup = ("TEACH", "Show a hole without washer → Teach")
        return vis, statuses, setup

    def run_auto_mode(self, results):
        # follow holes between frames so each one gets a stable (majority-vote) result
        for t in self.tracks:
            t["seen"] = False
        for res in results:
            dist = lambda t: np.hypot(t["c"][0] - res.center[0], t["c"][1] - res.center[1])
            free = [t for t in self.tracks if not t["seen"]]
            best = min(free, key=dist, default=None)
            if best is None or dist(best) > res.blob_radius:
                best = {"hist": collections.deque(maxlen=core.SMOOTH_FRAMES)}
                self.tracks.append(best)
            best.update(c=res.center, seen=True, miss=0)
            best["hist"].append(res.status)
            res.smoothed = core.majority(best["hist"])
        for t in self.tracks:
            if not t["seen"]:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= core.SMOOTH_FRAMES]

        statuses = [r.smoothed for r in results]
        vis = self.frame.copy()
        th = max(2, vis.shape[1] // 600)
        fs = vis.shape[1] / 1900
        for i, (res, st) in enumerate(zip(results, statuses), 1):
            col = core.COLORS[st]
            c = (int(res.center[0]), int(res.center[1]))
            r0 = res.hole_radius
            if self.debug.get():
                for k, kc in ((wd.RING_IN, col), (wd.RING_OUT, col),
                              (wd.REF_IN, core.GREY), (wd.REF_OUT, core.GREY)):
                    cv2.circle(vis, c, int(k * r0), kc, 1, cv2.LINE_AA)
                core.text(vis, f"{res.ratio:.2f}", (c[0] - int(r0), c[1] + int(2.6 * r0)), fs, col, th)
            else:
                cv2.circle(vis, c, int(1.6 * r0), col, th, cv2.LINE_AA)
            core.text(vis, f"H{i}", (c[0] - int(r0), c[1] - int(1.8 * r0)), fs, col, th)
        setup = None if results else ("NO HOLES", "Point camera at ≥ 2 neighbouring holes")
        return vis, statuses, setup

    # --- main loop ------------------------------------------------------------------
    def tick(self):
        if self.pending_cap is not None:
            idx, cap = self.pending_cap
            self.pending_cap = None
            self._camera_opened(idx, cap)

        auto_results = None
        if not self.frozen.get() or self.frame is None:
            if self.cap is not None:
                item = self.cap.latest()
                if item is None:                 # no new camera frame yet
                    self.root.after(5, self.tick)
                    return
                self.frame, auto_results = item
            elif self.still is not None:
                self.frame = self.still.copy()

        if self.frame is not None:
            if self.mode.get() == "auto":
                if auto_results is None:         # still image / frozen / just switched mode
                    auto_results = wd.inspect_auto(wd.to_gray(self.frame))
                vis, statuses, setup = self.run_auto_mode(auto_results)
            else:
                vis, statuses, setup = self.run_teach_mode(wd.to_gray(self.frame))
            self.last_vis = vis
            fh, fw = vis.shape[:2]
            self.scale = min(VIEW_W / fw, VIEW_H / fh)
            small = cv2.resize(vis, (int(fw * self.scale), int(fh * self.scale)))
            self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            self.canvas.itemconfig(self.img_item, image=self.photo)
            self.update_panel(statuses, setup)

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_prev, 1e-6)
            self.t_prev = now
            self.root.title(f"Washer Presence Inspection  ·  {self.mode.get().upper()}"
                            f"  ·  {self.fps:4.1f} fps")

        self.root.after(5, self.tick)

    def on_close(self):
        self.stop_camera()
        core.save_config(self.cfg)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    InspectionApp(root)
    root.mainloop()
