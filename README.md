# Washer Presence Inspection – POC

## Files
| File | Purpose |
|---|---|
| `run_gui.bat` | **Double-click to start the front end (use this for the demo)** |
| `inspection_gui.py` | Front end: AUTO / TEACH modes, OK/NG panel, hole grid, event log |
| `washer_detector.py` | The algorithm (no GUI) – all tunable thresholds at the top |
| `run_poc.bat` / `washer_poc.py` | Plain OpenCV window, TEACH mode only, keyboard-driven (≈22 fps) |
| `test_reference_images.py` | Offline check: AUTO mode on `ref_images/` + `captures/` → `results/` |
| `step00_find_cameras.py` | Lists camera indices (0 = laptop webcam, 1 = USB camera) |
| `step01_camera_test.py` | Plain camera viewer / frame saver |
| `config.json` | Created automatically: camera index, mode, TEACH boxes + hole size |
| `logs/inspection_YYYYMMDD.csv` | Result changes, mode changes, teach events |
| `captures/` | Snapshots (raw + annotated) |

## Two modes

### AUTO (default) – no setup
Point the camera at the flange. **Every hole in view** is found and checked automatically.
1. Find all dark, round blobs in the frame.
2. Measure the **hole spacing (pitch)** to the neighbouring holes. The spacing is the same
   with or without a washer, so it gives the scale:
   bare hole radius ≈ 0.265 × pitch, washer outer radius ≈ 0.39 × pitch.
3. Hole width and spacing are both measured along the row of holes, so camera tilt cancels out;
   the rings follow the hole's ellipse shape.
4. Washer ring check (below) + size check must agree, otherwise the hole shows **CHECK**.

Needs **≥ 2 neighbouring holes in view**. Holes that are tiny (< 10 px radius), seen almost
edge-on (squashed more than 2 : 1), cut off by the image edge, or not part of the regular hole
row (e.g. lifting-lug holes) are ignored rather than guessed.

### TEACH (backup) – fixed holes
For a fixed camera and specific positions: **✏ Edit holes** → drag a box around each hole
(≈ 4–5 × hole diameter) → **✔ Done editing** → with a hole WITHOUT washer in view,
**Teach hole size**. Re-do when the camera moves.

### Washer ring check (both modes)
Median grey level of the ring **1.15–1.40 × bare-hole radius** (where the washer sits)
÷ median of a bare-steel reference ring **1.85–2.15 ×**.
ratio < 0.45 → **PRESENT** (black washer) · > 0.65 → **MISSING** (bright steel) · between → **CHECK**.
Result = majority of the last 5 frames (no flicker).

Measured on the reference photos and live captures: PRESENT 0.13–0.34, MISSING 0.79–1.16.
AUTO mode on all 14 photos + 2 live captures: every washer found at normal viewing distance,
no false PRESENT; 2–3 CHECK only on extreme close-ups.

## Demo procedure
1. Double-click `run_gui.bat`. The camera takes ~15 s to start (status bar shows progress).
   Camera 1 = USB camera, 0 = laptop webcam (dropdown + Start/Stop).
2. AUTO is selected: point the camera at the flange (a few holes in view, roughly from above).
3. Remove / insert a washer → hole circle turns red/green, overall **NG/OK**, event log entry.
4. **Debug rings** shows what is measured; **Snapshot** saves evidence to `captures/`;
   **Image…** runs the same inspection on a saved photo.

## Known limitations (later phases)
- Front end ≈ 8–10 fps (Tkinter image display is the limit); results settle in < 1 s.
- AUTO hole numbers (H1, H2 …) are left-to-right in the current view, not fixed flange positions.
  Fixed position numbering needs a fixture / reference mark (Phase 3).
- Strong glare on the steel or a shadow across a hole can give CHECK.
  Phase 4 (diffuse ring light / dome) removes this.
