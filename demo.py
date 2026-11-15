"""
project iris: webcam gaze tracking demo (Gaussian binocular S² model).

Tracks the pupils with MediaPipe Face Mesh, lifts the pupil displacement onto
the unit sphere, filters it there, and projects the result onto the screen.

    python3 demo.py                              # live preview
    python3 demo.py --export gaze.json           # also save screen coordinates
    python3 demo.py --export gaze.json --screen-size 1920x1080

Output
------
Screen coordinates use a top-left origin, x to the right and y down, from the
user's point of view (looking at the left of the screen gives a small x).

Every tracked frame is also classified into one of nine regions on a 3x3 grid:

    top left     top center     top right
    middle left  middle center  middle right
    bottom left  bottom center  bottom right

The current region is drawn on the preview and, with --export, recorded for
every sample in the JSON file. Nothing is logged to the terminal per frame.

Calibration
-----------
On start a target is shown in the centre of the preview. Look at it and hold
still: after a short settle time the pupil position is captured and becomes
"looking at the centre" for this person, head position and camera. Press C to
recalibrate at any time, or pass --no-calibrate to skip it and fall back to
the slowly adapting baseline. Keep the preview window in the middle of the
screen (or maximised) so the target really is at the screen centre.

After calibration the head may still move: the centre point shifts with the
eye centroid relative to where it was during calibration, and the gaze range
on each side of it is stretched to its screen edge. Without calibration the
neutral gaze is anchored to where the eye centroid sits in the camera frame.

With --export, every tracked frame is written to a JSON file on exit:

    {
      "format": "project-iris-gaze",
      "version": 1,
      "created": "2026-01-01T12:00:00+00:00",
      "screen": {"width": 1920, "height": 1080},
      "coordinates": "origin top-left, x right, y down, user's point of view",
      "regions": ["top left", "top center", ...],
      "calibration": {"type": "single-point", "target": "center", "t": 2.113},
      "samples": [
        {"frame": 12, "t": 0.412, "x": 963, "y": 547,
         "nx": 0.5016, "ny": 0.5065, "region": "middle center"},
        ...
      ]
    }

"x"/"y" are pixels on the screen given by --screen-size (default: the camera
frame size), "nx"/"ny" are the same point normalised to 0..1, "t" is seconds
since start and "frame" is the camera frame index (frames with no face are
skipped, so gaps in "frame" mean tracking was lost). Frames shown while the
calibration target is up are not exported either. "calibration" describes the
most recent calibration ("t" = when it completed) or is null if there was none.

Keys: ESC quit, C recalibrate, [ and ] vertical gain looking down,
- and = looking up.
"""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# CONFIG
# ============================================================
SMOOTH_ALPHA = 0.35
BASELINE_ALPHA = 0.01
ORIGIN_ALPHA = 0.18

SCREEN_WIDTH_EYE_UNITS = 8.0
SCREEN_Z_EYE_UNITS = 19.0
CLAMP_FRAC = 0.99

# Width of the Gaussian weighting applied to the normalised pupil displacement.
GAUSS_WIDTH = 11.7

# Vertical gain, split by direction. Looking down, the lids and eye corners
# travel with the eye, so the measured pupil shift is much smaller than for
# an equal upward look. Tune live with [ and ] (down) or - and = (up).
GAIN_Y_UP = 1.0
GAIN_Y_DOWN = 2.0

# Iris + pupil
IRIS_L = [469, 470, 471, 472]
IRIS_R = [474, 475, 476, 477]
PUPIL_L = 468
PUPIL_R = 473

# Eye corners
L_INNER, L_OUTER = 133, 33
R_INNER, R_OUTER = 362, 263

# Eyelids (socket stabilization)
UPPER_LID_L = 159
LOWER_LID_L = 145
UPPER_LID_R = 386
LOWER_LID_R = 374

MIRROR = -1.0

# Head-position compensation. The neutral (straight ahead) gaze lands where
# the eye centroid sits in the camera frame instead of always at the screen
# centre, and the gaze range on each side of that anchor is stretched to reach
# its own screen edge. 0 = off (anchor fixed at the centre), 1 = anchor follows
# the centroid fully.
HEAD_COMP_X = 1.0
HEAD_COMP_Y = 1.0
# The camera image is unmirrored, so a head on the image's left is on the
# user's right. Set to False if your camera delivers a mirrored (selfie) feed.
HEAD_MIRROR_X = True
# Keep the anchor this far (fraction of the screen) from every edge so neither
# side of the split collapses to nothing.
ANCHOR_MARGIN = 0.1

# Single-point centre calibration. The target is shown for CALIB_SETTLE_S so
# the eyes can land on it, then pupil positions are collected for
# CALIB_CAPTURE_S and their median becomes the "looking at the centre" pose.
# If fewer than CALIB_MIN_SAMPLES tracked frames arrive (face lost, long
# blink) the capture starts over.
CALIB_SETTLE_S = 1.0
CALIB_CAPTURE_S = 1.2
CALIB_MIN_SAMPLES = 12

# 3x3 gaze regions. A new region is only reported once the gaze is this far
# (as a fraction of the screen) past the boundary, so it does not flicker
# when the gaze sits on a grid line.
REGION_ROWS = ("top", "middle", "bottom")
REGION_COLS = ("left", "center", "right")
REGION_NAMES = [f"{r} {c}" for r in REGION_ROWS for c in REGION_COLS]
REGION_HYSTERESIS = 0.03

EXPORT_FORMAT = "project-iris-gaze"
EXPORT_VERSION = 1

# ============================================================
# UTILS
# ============================================================
def median_radius(iris_pts, pupil):
    return np.median(np.linalg.norm(iris_pts - pupil, axis=1))

def smooth(prev, new, alpha):
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha

def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-8 else v / n

# ============================================================
# S² FILTER
# ============================================================
def s2_log(p, q):
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    theta = math.acos(dot)
    if theta < 1e-6:
        return np.zeros(3)
    v = q - dot * p
    nv = np.linalg.norm(v)
    return np.zeros(3) if nv < 1e-8 else v * (theta / nv)

def s2_exp(p, v):
    theta = np.linalg.norm(v)
    if theta < 1e-6:
        return p
    return normalize(
        math.cos(theta) * p +
        math.sin(theta) * (v / theta)
    )

class GazeS2Filter:
    def __init__(self, alpha=0.25, max_step=0.25):
        self.alpha = alpha
        self.max_step = max_step
        self.g = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def update(self, g_raw):
        g_raw = normalize(g_raw)
        e = s2_log(self.g, g_raw)
        mag = np.linalg.norm(e)
        if mag > self.max_step:
            e *= self.max_step / mag
        self.g = s2_exp(self.g, self.alpha * e)
        return self.g

# ============================================================
# SCREEN GEOMETRY
# ============================================================
def tan_theta_max_from_screen(aspect):
    half_w = 0.5 * SCREEN_WIDTH_EYE_UNITS
    half_h = 0.5 * (SCREEN_WIDTH_EYE_UNITS / max(1e-6, aspect))
    return (
        half_w / SCREEN_Z_EYE_UNITS,
        half_h / SCREEN_Z_EYE_UNITS,
    )

def norm_to_pixels(nx, ny, width, height):
    """Normalised screen position (0..1) -> integer pixel inside the screen."""
    return (
        min(width - 1, max(0, int(nx * width))),
        min(height - 1, max(0, int(ny * height))),
    )

def origin_norm(origin, width, height):
    """Eye centroid in camera pixels -> 0..1 in the user's point of view."""
    ox = float(origin[0]) / max(1, width)
    oy = float(origin[1]) / max(1, height)
    if HEAD_MIRROR_X:
        ox = 1.0 - ox
    return ox, oy

def head_anchor(origin, width, height, ref=(0.5, 0.5)):
    """Eye centroid in camera pixels -> neutral gaze point on screen (0..1).

    `ref` is the normalised eye centroid for which the neutral gaze lands on
    the screen centre: the frame centre by default, or the head position
    captured by the centre calibration.
    """
    ox, oy = origin_norm(origin, width, height)
    ax = 0.5 + HEAD_COMP_X * (ox - ref[0])
    ay = 0.5 + HEAD_COMP_Y * (oy - ref[1])
    lo, hi = ANCHOR_MARGIN, 1.0 - ANCHOR_MARGIN
    return min(hi, max(lo, ax)), min(hi, max(lo, ay))

def anchored_norm(i, anchor):
    """Gaze offset i (-1..1) -> screen position (0..1) split at `anchor`.

    i = 0 lands on the anchor, i = -1 on the near edge (0) and i = +1 on the
    far edge (1), so the whole screen stays reachable wherever the anchor is.
    With anchor = 0.5 this is the plain even split (i + 1) / 2.
    """
    i = float(np.clip(i, -1, 1))
    return anchor + i * ((1.0 - anchor) if i >= 0 else anchor)

# ============================================================
# CENTRE CALIBRATION
# ============================================================
class CenterCalibration:
    """Single-point calibration: the user looks at a target in the centre.

    While `active`, feed it one observation per tracked frame. When it
    finishes it holds the pupil offsets (in iris radii, so they survive the
    user moving closer or further away) and the head position that together
    mean "looking at the screen centre".
    """

    def __init__(self, active=True):
        self.active = False
        self.done = False
        self.base_L = self.base_R = None   # pupil offset / iris radius
        self.origin = None                 # normalised eye centroid
        self._t0 = None
        self._samples = []
        if active:
            self.start()

    def start(self):
        self.active = True
        self._t0 = None
        self._samples = []

    def _elapsed(self, now):
        return 0.0 if self._t0 is None else now - self._t0

    def capturing(self, now):
        return self.active and self._elapsed(now) >= CALIB_SETTLE_S

    def progress(self, now):
        """Capture progress, 0..1 (0 while the eyes are still settling)."""
        done = (self._elapsed(now) - CALIB_SETTLE_S) / CALIB_CAPTURE_S
        return min(1.0, max(0.0, done))

    def tick(self, now):
        """Advance the clock without an observation (no face this frame)."""
        if self.active and self._t0 is not None \
                and self._elapsed(now) >= CALIB_SETTLE_S + CALIB_CAPTURE_S:
            self._finish()

    def feed(self, now, pupil_L, pupil_R, origin):
        """Add one tracked frame. Returns True when calibration completes."""
        if not self.active:
            return False
        if self._t0 is None:
            self._t0 = now
        elapsed = self._elapsed(now)
        if elapsed < CALIB_SETTLE_S:
            return False
        self._samples.append((*pupil_L, *pupil_R, *origin))
        if elapsed < CALIB_SETTLE_S + CALIB_CAPTURE_S:
            return False
        return self._finish()

    def _finish(self):
        if len(self._samples) < CALIB_MIN_SAMPLES:
            self.start()            # not enough data: try again
            return False
        # Median, so blinks and stray saccades during the capture are ignored.
        med = np.median(np.array(self._samples, dtype=np.float32), axis=0)
        self.base_L, self.base_R = med[0:2], med[2:4]
        self.origin = (float(med[4]), float(med[5]))
        self.active = False
        self.done = True
        return True

def draw_calibration(frame, calib, now):
    """Dim the preview and draw the centre target with capture progress."""
    h, w, _ = frame.shape
    frame = cv2.convertScaleAbs(frame, alpha=0.3)
    c = (w // 2, h // 2)
    cv2.circle(frame, c, 22, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, c, 6, (0, 0, 255), -1, cv2.LINE_AA)
    if calib.capturing(now):
        cv2.ellipse(frame, c, (34, 34), -90, 0, 360 * calib.progress(now),
                    (0, 255, 0), 4, cv2.LINE_AA)
        text = "Hold still..."
    else:
        text = "Look at the dot"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, text, ((w - tw) // 2, h // 2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

# ============================================================
# GAZE REGIONS
# ============================================================
def region_cell(v, prev=None, margin=REGION_HYSTERESIS):
    """Normalised coordinate (0..1) -> third of the screen it falls in (0, 1, 2).

    If `prev` is given, the cell only changes once `v` is more than `margin`
    past the edge of the previous cell.
    """
    cell = min(2, max(0, int(v * 3)))
    if prev is not None and cell != prev:
        if prev / 3 - margin <= v <= (prev + 1) / 3 + margin:
            return prev
    return cell

def region_name(col, row):
    """Grid cell (col, row), each 0..2 -> e.g. 'top left', 'middle center'."""
    return f"{REGION_ROWS[row]} {REGION_COLS[col]}"

def gaze_region(nx, ny):
    """Normalised screen position (0..1) -> region name, without hysteresis."""
    return region_name(region_cell(nx), region_cell(ny))

# ============================================================
# JSON EXPORT
# ============================================================
class GazeExporter:
    """Collects per-frame gaze samples and writes them to one JSON file."""

    def __init__(self, path, screen_size=None):
        self.path = path
        self.screen_size = screen_size  # (width, height); None = camera frame size
        self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.samples = []
        self.calibration = None

    def set_calibration(self, t):
        self.calibration = {"type": "single-point", "target": "center",
                            "t": round(t, 4)}

    def add(self, frame, t, nx, ny, region, frame_size):
        if self.screen_size is None:
            self.screen_size = frame_size
        x, y = norm_to_pixels(nx, ny, *self.screen_size)
        self.samples.append({
            "frame": frame,
            "t": round(t, 4),
            "x": x,
            "y": y,
            "nx": round(float(nx), 4),
            "ny": round(float(ny), 4),
            "region": region,
        })

    def save(self):
        width, height = self.screen_size or (None, None)
        doc = {
            "format": EXPORT_FORMAT,
            "version": EXPORT_VERSION,
            "created": self.created,
            "screen": {"width": width, "height": height},
            "coordinates": "origin top-left, x right, y down, user's point of view",
            "regions": REGION_NAMES,
            "calibration": self.calibration,
            "samples": self.samples,
        }
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        os.replace(tmp, self.path)

# ============================================================
# CLI
# ============================================================
def parse_screen_size(text):
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("screen size must be positive")
    return width, height

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Project Iris: webcam gaze tracking demo.")
    p.add_argument("--export", metavar="PATH",
                   help="write per-frame screen coordinates to this JSON file on exit")
    p.add_argument("--screen-size", metavar="WxH", type=parse_screen_size,
                   help="screen size in pixels for exported x/y, e.g. 1920x1080 "
                        "(default: camera frame size)")
    p.add_argument("--camera", type=int, default=0,
                   help="camera index (default: 0)")
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip the centre calibration on start (press C to run it later)")
    return p.parse_args(argv)

# ============================================================
# MAIN
# ============================================================
def run(args):
    gain_y_up = GAIN_Y_UP
    gain_y_down = GAIN_Y_DOWN

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    exporter = GazeExporter(args.export, args.screen_size) if args.export else None

    base_L = base_R = None
    smooth_L = smooth_R = None
    origin_smooth = None
    gaze_filter = GazeS2Filter()
    calib = CenterCalibration(active=not args.no_calibrate)

    col = row = None
    region = None
    frame_idx = -1
    t_start = time.monotonic()

    print("ESC to quit, C to recalibrate\n")

    def poll_keys():
        """Handle one key press. Returns False when the user quits."""
        nonlocal gain_y_up, gain_y_down
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False
        elif key in (ord('c'), ord('C')):
            calib.start()
        elif key == ord(']'):
            gain_y_down += 0.1
        elif key == ord('['):
            gain_y_down = max(0.1, gain_y_down - 0.1)
        elif key == ord('='):
            gain_y_up += 0.1
        elif key == ord('-'):
            gain_y_up = max(0.1, gain_y_up - 0.1)
        return True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            now = time.monotonic()

            h, w, _ = frame.shape
            aspect = w / max(1, h)
            tan_h, tan_v = tan_theta_max_from_screen(aspect)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_face.process(img_rgb)

            if not res.multi_face_landmarks:
                if HEAD_MIRROR_X:
                    frame = cv2.flip(frame, 1)
                if calib.active:
                    calib.tick(now)
                    frame = draw_calibration(frame, calib, now)
                cv2.imshow("Project Iris", frame)
                if not poll_keys():
                    break
                continue

            pts = res.multi_face_landmarks[0].landmark

            def px(i):
                return np.array([pts[i].x * w, pts[i].y * h], dtype=np.float32)

            # ------------------------------------------------
            # Pupils / irises
            # ------------------------------------------------
            pL = px(PUPIL_L)
            pR = px(PUPIL_R)

            irisL = np.array([px(i) for i in IRIS_L])
            irisR = np.array([px(i) for i in IRIS_R])

            rL = median_radius(irisL, pL)
            rR = median_radius(irisR, pR)
            r_iris = 0.5 * (rL + rR)

            # ------------------------------------------------
            # Eye socket centroids
            # ------------------------------------------------
            def eyeball_socket(outer, inner, upper, lower):
                return np.mean(
                    [px(outer), px(inner), px(upper), px(lower)],
                    axis=0
                )

            socket_L = eyeball_socket(L_OUTER, L_INNER, UPPER_LID_L, LOWER_LID_L)
            socket_R = eyeball_socket(R_OUTER, R_INNER, UPPER_LID_R, LOWER_LID_R)

            origin_raw = 0.5 * (socket_L + socket_R)
            origin_smooth = smooth(origin_smooth, origin_raw, ORIGIN_ALPHA)
            origin = origin_smooth

            # ------------------------------------------------
            # Eye-relative pupil displacement
            # ------------------------------------------------
            cL = 0.5 * (px(L_INNER) + px(L_OUTER))
            cR = 0.5 * (px(R_INNER) + px(R_OUTER))

            pL_rel = pL - cL
            pR_rel = pR - cR

            # ------------------------------------------------
            # Centre calibration
            # ------------------------------------------------
            if calib.active:
                finished = calib.feed(now,
                                      pL_rel / (r_iris + 1e-6),
                                      pR_rel / (r_iris + 1e-6),
                                      origin_norm(origin_raw, w, h))
                if finished:
                    # Start the filters fresh from the calibrated centre.
                    smooth_L = smooth_R = None
                    gaze_filter = GazeS2Filter()
                    col = row = None
                    if exporter is not None:
                        exporter.set_calibration(now - t_start)
                else:
                    if HEAD_MIRROR_X:
                        frame = cv2.flip(frame, 1)
                    cv2.imshow("Project Iris", draw_calibration(frame, calib, now))
                    if not poll_keys():
                        break
                    continue

            if calib.done:
                # Fixed baseline: the pupil offsets captured while the user
                # looked at the centre target, rescaled to the current iris size.
                base_L = calib.base_L * r_iris
                base_R = calib.base_R * r_iris
            else:
                # Uncalibrated: slowly adapting baseline.
                base_L = smooth(base_L, pL_rel, BASELINE_ALPHA)
                base_R = smooth(base_R, pR_rel, BASELINE_ALPHA)

            smooth_L = smooth(smooth_L, pL_rel - base_L, SMOOTH_ALPHA)
            smooth_R = smooth(smooth_R, pR_rel - base_R, SMOOTH_ALPHA)

            # ------------------------------------------------
            # Gaussian binocular fusion
            # ------------------------------------------------
            xL = smooth_L[0] / (r_iris + 1e-6)
            yL = smooth_L[1] / (r_iris + 1e-6)

            xR = smooth_R[0] / (r_iris + 1e-6)
            yR = smooth_R[1] / (r_iris + 1e-6)

            r2_L = xL * xL + yL * yL
            r2_R = xR * xR + yR * yR

            wL = math.exp(-(r2_L) / GAUSS_WIDTH)
            wR = math.exp(-(r2_R) / GAUSS_WIDTH)

            # Directional Gaussian fields
            gx = xL * wL
            gy = yR * wR
            gy *= gain_y_down if gy > 0 else gain_y_up   # image y grows downward

            ux = MIRROR * gx
            uy = MIRROR * gy

            # ------------------------------------------------
            # Unit sphere lift
            # ------------------------------------------------
            r2 = ux * ux + uy * uy
            if r2 > CLAMP_FRAC * CLAMP_FRAC:
                s = CLAMP_FRAC / math.sqrt(r2)
                ux *= s
                uy *= s
                r2 = ux * ux + uy * uy

            uz = math.sqrt(max(0.0, 1.0 - r2))
            g = gaze_filter.update(normalize(np.array([ux, -uy, uz])))

            # ------------------------------------------------
            # Screen projection
            # ------------------------------------------------
            ix = (g[0] / max(1e-6, g[2])) / tan_h
            iy = (g[1] / max(1e-6, g[2])) / tan_v

            # Split the screen at the head anchor rather than at its centre,
            # so an off-centre head still reaches every row and column.
            # Once calibrated, the anchor is the centre for the calibrated
            # head position and follows the head from there.
            anchor_ref = calib.origin if calib.done else (0.5, 0.5)
            anchor_x, anchor_y = head_anchor(origin, w, h, anchor_ref)
            nx = anchored_norm(ix, anchor_x)   # 0 = left edge, 1 = right
            ny = anchored_norm(iy, anchor_y)   # 0 = top edge,  1 = bottom

            gaze_x, gaze_y = norm_to_pixels(nx, ny, w, h)

            # ------------------------------------------------
            # Gaze region + export
            # ------------------------------------------------
            col = region_cell(nx, col)
            row = region_cell(ny, row)
            region = region_name(col, row)

            if exporter is not None:
                exporter.add(frame_idx, now - t_start,
                             nx, ny, region, (w, h))

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------
            # Show a selfie view so the overlays, which are in the user's
            # point of view, line up with the face. Tracking above used the
            # unflipped frame, so only the picture changes.
            if HEAD_MIRROR_X:
                frame = cv2.flip(frame, 1)

            # 3x3 region grid, active cell highlighted
            for k in (1, 2):
                cv2.line(frame, (k * w // 3, 0), (k * w // 3, h), (90, 90, 90), 1)
                cv2.line(frame, (0, k * h // 3), (w, k * h // 3), (90, 90, 90), 1)
            cv2.rectangle(frame,
                          (col * w // 3, row * h // 3),
                          ((col + 1) * w // 3 - 1, (row + 1) * h // 3 - 1),
                          (0, 255, 0), 2)

            # Head anchor: where a straight-ahead gaze currently lands
            ax_px, ay_px = norm_to_pixels(anchor_x, anchor_y, w, h)
            cv2.drawMarker(frame, (ax_px, ay_px), (0, 255, 255),
                           cv2.MARKER_CROSS, 24, 2)

            cv2.circle(frame, (gaze_x, gaze_y), 18, (0, 255, 0), -1, cv2.LINE_AA)  # lime green (BGR)

            cv2.putText(frame, "project iris - webcam gaze tracking demo",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            cv2.putText(frame, f"gain up {gain_y_up:.1f} (-/=)   down {gain_y_down:.1f} ([/])   "
                               f"{'calibrated' if calib.done else 'uncalibrated'} (C)",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            cv2.putText(frame, f"looking {region}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            cv2.imshow("Project Iris", frame)

            if not poll_keys():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mp_face.close()
        if exporter is not None:
            exporter.save()
            print(f"\nSaved {len(exporter.samples)} gaze samples to {exporter.path}")


if __name__ == "__main__":
    run(parse_args())
