"""Washer presence detection - core algorithm (no camera, no GUI, testable on images).

Method (per hole ROI):
  1. Find the dark circular blob nearest the ROI centre -> hole centre.
     (With a washer, hole + washer merge into one bigger dark blob; the centre is
     the same because they are concentric.)
  2. Sample the ANNULUS just outside the bare-hole radius (1.15r .. 1.40r).
     Washer present -> annulus is black washer.  Missing -> annulus is bright steel.
  3. Compare annulus brightness with a REFERENCE ring of steel further out
     (1.85r .. 2.15r). Using a ratio instead of a fixed grey level makes the
     decision tolerant to lighting / exposure changes and reflections.

Measured on the reference photos: hole Ø ≈ 75 px, washer OD ≈ 117 px (OD/hole ≈ 1.55),
hole grey ≈ 6, washer grey ≈ 10-30, machined steel grey ≈ 90-180.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# --- Tunable parameters (radii are multiples of the bare-hole radius) --------------
RING_IN, RING_OUT = 1.15, 1.40   # where the washer sits (washer OD ≈ 1.55 r)
REF_IN, REF_OUT = 1.85, 2.15     # bare steel just outside the washer
PRESENT_BELOW = 0.45             # annulus/reference ratio < 0.45 -> PRESENT
MISSING_ABOVE = 0.65             # ratio > 0.65 -> MISSING, in between -> CHECK
DARK_LEVEL = 0.30                # dark threshold = p2 + DARK_LEVEL * (p90 - p2)
MIN_CIRCULARITY = 0.60           # 4*pi*area/perimeter^2, 1.0 = perfect circle

# --- AUTO mode (no ROIs, no teaching) ------------------------------------------------
# Hole centre-to-centre spacing (pitch) is the same with or without washer, so it gives
# the scale. Measured on reference photos + live captures:
#   bare hole radius / pitch   = 0.24 .. 0.30  (median 0.265)
#   washer outer radius / pitch ≈ 0.39
AUTO_DARK_LEVELS = (0.20, 0.30)  # whole-frame search: strict + normal dark threshold
BARE_R_PER_PITCH = 0.265
WASHER_SIZE_SPLIT = 0.33         # blob radius / pitch above this = hole + washer
AUTO_MIN_CIRCULARITY = 0.70
AUTO_MIN_RADIUS_PX = 10         # smaller holes = too few pixels to judge -> ignored
AUTO_MAX_ASPECT = 2.0            # hole seen more squashed than 2:1 = too oblique -> ignored

PRESENT, MISSING, CHECK, NO_HOLE = "PRESENT", "MISSING", "CHECK", "NO HOLE"


@dataclass
class HoleResult:
    status: str
    center: tuple | None = None      # (x, y) in full-frame pixels
    blob_radius: float = 0.0         # radius of dark region (hole, or hole+washer)
    ratio: float = float("nan")      # annulus median / reference median
    ring_med: float = float("nan")
    ref_med: float = float("nan")
    hole_radius: float = 0.0         # bare-hole radius used for the rings (px)


def to_gray(frame):
    """BGR -> grayscale + light blur to suppress sensor noise and machining marks."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.GaussianBlur(gray, (5, 5), 0)


def percentiles(gray, qs):
    """Fast percentiles of an 8-bit image via its histogram (np.percentile is ~20x slower)."""
    cdf = np.cumsum(cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel())
    return [int(np.searchsorted(cdf, q / 100 * cdf[-1])) for q in qs]


def dark_mask(roi_gray, level=DARK_LEVEL):
    """Binary mask of 'dark' pixels, threshold adapted to this image's own contrast."""
    p2, p90 = percentiles(roi_gray, (2, 90))
    thr = p2 + level * (p90 - p2)
    mask = (roi_gray < thr).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small gaps
    return mask


def find_hole(gray, roi):
    """Return (cx, cy, radius) of the dark circular blob nearest the ROI centre, or None."""
    x, y, w, h = roi
    sub = gray[y:y + h, x:x + w]
    mask = dark_mask(sub)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best, best_dist = None, float("inf")
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.005 * w * h:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bx <= 0 or by <= 0 or bx + bw >= w or by + bh >= h:
            continue                         # touches ROI border -> not a whole hole
        perim = cv2.arcLength(c, True)
        if 4 * np.pi * area / (perim * perim) < MIN_CIRCULARITY:
            continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        dist = np.hypot(cx - w / 2, cy - h / 2)
        if dist < best_dist:
            best, best_dist = (x + cx, y + cy, np.sqrt(area / np.pi)), dist
    return best


def ring_median(gray, cx, cy, r_in, r_out):
    """Median grey level of pixels between radius r_in and r_out around (cx, cy)."""
    x0, y0 = int(max(cx - r_out - 1, 0)), int(max(cy - r_out - 1, 0))
    x1 = int(min(cx + r_out + 2, gray.shape[1]))
    y1 = int(min(cy + r_out + 2, gray.shape[0]))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)
    vals = gray[y0:y1, x0:x1][(d >= r_in) & (d <= r_out)]
    return float(np.median(vals)) if vals.size else float("nan")


def classify_at(gray, cx, cy, blob_r, hole_radius):
    """Washer decision for a hole centred at (cx, cy) with known bare-hole radius."""
    ring = ring_median(gray, cx, cy, RING_IN * hole_radius, RING_OUT * hole_radius)
    ref = ring_median(gray, cx, cy, REF_IN * hole_radius, REF_OUT * hole_radius)
    ratio = ring / max(ref, 1.0)

    if ratio < PRESENT_BELOW:
        status = PRESENT
    elif ratio > MISSING_ABOVE:
        status = MISSING
    else:
        status = CHECK
    return HoleResult(status, (cx, cy), blob_r, ratio, ring, ref, hole_radius)


def inspect_hole(gray, roi, hole_radius):
    """TEACH mode: classify one hole ROI. hole_radius = taught bare-hole radius in pixels."""
    found = find_hole(gray, roi)
    if found is None:
        return HoleResult(NO_HOLE)
    return classify_at(gray, *found, hole_radius)


def find_blobs(gray):
    """All dark, roughly circular blobs in the whole frame.

    Searched at two darkness levels (strict + normal) and merged, so a washer that is
    lit slightly grey is still found, and holes in a dark reflection band are not
    merged with the steel. Returns list of dicts: centre, radius, ellipse, contour.
    """
    # search on a half-size copy (4x fewer pixels); geometry is scaled back to full size
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    blobs = []
    for level in AUTO_DARK_LEVELS:
        mask = dark_mask(small, level)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < max(20, 2e-5 * h * w) or len(c) < 5:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1:
                continue                                   # cut off by the image edge
            perim = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / (perim * perim)
            if circ < AUTO_MIN_CIRCULARITY:
                continue
            m = cv2.moments(c)
            cx, cy, r = 2 * m["m10"] / m["m00"], 2 * m["m01"] / m["m00"], 2 * np.sqrt(area / np.pi)
            (ex, ey), (d1, d2), ang = cv2.fitEllipse(c)
            dup = next((b for b in blobs if np.hypot(b["cx"] - cx, b["cy"] - cy) < 0.5 * r), None)
            if dup is not None:
                if circ <= dup["circ"]:
                    continue
                blobs.remove(dup)                          # keep the more circular version
            blobs.append(dict(cx=cx, cy=cy, r=r, circ=circ, contour=2.0 * c[:, 0, :],
                              ellipse=((2 * ex, 2 * ey), (2 * d1, 2 * d2), ang)))
    return blobs


def ellipse_ring_medians(gray, ellipse, rings):
    """Median grey level in each (rho_in, rho_out) ring of an ellipse (rho = 1 is the
    ellipse outline), so the rings follow the perspective of the hole."""
    (cx, cy), (d1, d2), ang = ellipse
    a, b = d1 / 2, d2 / 2
    t = np.deg2rad(ang)
    theta = _RING_ANGLES[None, :]
    h, w = gray.shape
    out = []
    for rho_in, rho_out in rings:
        rho = np.linspace(rho_in, rho_out, 4)[:, None]      # 4 radii x 72 angles = 288 samples
        u, v = rho * a * np.cos(theta), rho * b * np.sin(theta)
        x = np.rint(cx + u * np.cos(t) - v * np.sin(t)).astype(int).ravel()
        y = np.rint(cy + u * np.sin(t) + v * np.cos(t)).astype(int).ravel()
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        out.append(float(np.median(gray[y[ok], x[ok]])) if ok.any() else float("nan"))
    return out


_RING_ANGLES = np.linspace(0, 2 * np.pi, 72, endpoint=False)


def inspect_auto(gray):
    """AUTO mode: find every flange hole in the frame and classify it (no setup needed).

    Scale comes from the hole spacing (pitch), measured ALONG the row of holes, and the
    blob width is measured along the same direction - so camera tilt shrinks both
    equally and cancels out. Needs at least 2 neighbouring holes in view.
    """
    blobs = find_blobs(gray)
    if len(blobs) < 2:
        return []
    pts = np.array([(b["cx"], b["cy"]) for b in blobs])
    rs = np.array([b["r"] for b in blobs])
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    similar = (rs[None, :] > 0.5 * rs[:, None]) & (rs[None, :] < 2.0 * rs[:, None])
    d = np.where(similar, d, np.inf)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)

    results = []
    for i, b in enumerate(blobs):
        j = int(np.argmin(d[i]))
        if not np.isfinite(d[i, j]) or b["r"] < AUTO_MIN_RADIUS_PX:
            continue                                       # no neighbour / too small
        if nn[i] > 1.5 * nn[j]:
            continue                                       # isolated blob, not in the row
        axes = b["ellipse"][1]
        if max(axes) > AUTO_MAX_ASPECT * min(axes):
            continue                                       # seen too edge-on to judge
        near = [k for k in np.argsort(d[i])[:3] if np.isfinite(d[i, k])]
        pitch = float(np.median([nn[i]] + [nn[k] for k in near]))
        u = (pts[j] - pts[i]) / d[i, j]                     # row direction
        proj = b["contour"] @ u
        k_row = (proj.max() - proj.min()) / 2 / pitch       # half-width along row / pitch
        if not 0.18 <= k_row <= 0.50:
            continue                                       # not part of the hole pattern
        r0n = BARE_R_PER_PITCH / k_row                      # bare-hole radius, blob units
        ring, ref = ellipse_ring_medians(gray, b["ellipse"], [(RING_IN * r0n, RING_OUT * r0n),
                                                              (REF_IN * r0n, REF_OUT * r0n)])
        ratio = ring / max(ref, 1.0)
        if ratio < PRESENT_BELOW:
            status = PRESENT
        elif ratio > MISSING_ABOVE:
            status = MISSING
        else:
            status = CHECK
        if status != CHECK and (status == PRESENT) != (k_row > WASHER_SIZE_SPLIT):
            status = CHECK                                 # ring and size disagree
        results.append(HoleResult(status, (b["cx"], b["cy"]), b["r"], ratio, ring, ref,
                                  b["r"] * r0n))
    results.sort(key=lambda r: (r.center[0], r.center[1]))
    return results


def teach_hole_radius(gray, rois):
    """Bare-hole radius = smallest dark blob among the ROIs.

    Requires at least one hole WITHOUT washer to be visible when teaching.
    Returns (radius or None, message).
    """
    radii = [f[2] for f in (find_hole(gray, r) for r in rois) if f is not None]
    if not radii:
        return None, "Teach failed: no hole found in any ROI"
    r_min, r_max = min(radii), max(radii)
    msg = f"Taught hole radius = {r_min:.1f} px"
    if len(radii) > 1 and r_max / r_min < 1.3:
        msg += "  (WARNING: all blobs similar size - was a bare hole visible?)"
    return r_min, msg
