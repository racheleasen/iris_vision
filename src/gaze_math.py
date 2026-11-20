"""
Gaze math for project iris (angular binocular S² model).

Pure functions and one small filter: eye and head features -> gaze direction
on the unit sphere ->
screen position -> 3x2 region. No camera, no MediaPipe and no drawing in here,
only numpy, so every step can be used and tested on its own. The per-frame
state machine that strings these together is in gaze_engine.py.

Screen positions are normalised to 0..1 with a top-left origin, x right and
y down, from the user's point of view.
"""

import math

import numpy as np

# ---- Config ----------------------------------------------------------------
SCREEN_WIDTH_EYE_UNITS = 8.0
SCREEN_Z_EYE_UNITS = 19.0
CLAMP_FRAC = 0.99
MIRROR = -1.0           # unmirrored camera: image-left is the user's right

# Angular gaze model. Each feature is measured relative to the calibrated
# "looking at the centre" pose and mapped to true gaze angles:
#   sin(yaw)   = GAIN_X * pupil x offset (eye widths)
#   sin(pitch) = GAIN_Y * pupil y offset (eye widths) + GAIN_LID * lid opening
#   then head rotation since calibration is added back on, because the pupil
#   only measures the eye's rotation inside the head.
# Fitted to minimise mean angular error on MPIIFaceGaze subjects p00-p07 and
# checked on held-out p08-p14 (benchmark/README.md).
GAIN_X = 3.37
GAIN_Y = 3.23
GAIN_LID = -1.52        # eyes open wider when looking up
HEAD_GAIN_YAW = 0.75
HEAD_GAIN_PITCH = 0.19

# Vertical gain multipliers on top of the fitted model, split by direction
# (1.0 = as fitted). main.py tunes them live with [ ] and - =.
GAIN_Y_UP = 1.0
GAIN_Y_DOWN = 1.0

# Head-position compensation. The neutral (straight ahead) gaze lands where
# the eye centroid sits in the camera frame instead of always at the screen
# centre, and the gaze range on each side of that anchor is stretched to reach
# its own screen edge. 0 = off (anchor fixed at the centre), 1 = full.
HEAD_COMP_X = 1.0
HEAD_COMP_Y = 1.0
# The camera image is unmirrored, so a head on the image's left is on the
# user's right. Set to False if your camera delivers a mirrored (selfie) feed.
HEAD_MIRROR_X = True
# Keep the anchor this far (fraction of the screen) from every edge so neither
# side of the split collapses to nothing.
ANCHOR_MARGIN = 0.1

# Six gaze regions: 3 columns x 2 rows. A new region is only reported once the
# gaze is this far (as a fraction of the screen) past the boundary, so it does
# not flicker when the gaze sits on a grid line.
REGION_ROWS = ("top", "bottom")
REGION_COLS = ("left", "center", "right")
REGION_NAMES = [f"{r} {c}" for r in REGION_ROWS for c in REGION_COLS]
REGION_HYSTERESIS = 0.03

# ---- Utils -----------------------------------------------------------------
def smooth(prev, new, alpha):
    return new if prev is None else prev * (1 - alpha) + new * alpha

def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-8 else v / n

# ---- S² filter -------------------------------------------------------------
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
    return normalize(math.cos(theta) * p + math.sin(theta) * (v / theta))

class GazeS2Filter:
    """Low-pass filter for a gaze direction on the unit sphere."""

    def __init__(self, alpha=0.25, max_step=0.25):
        self.alpha = alpha
        self.max_step = max_step
        self.g = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def update(self, g_raw):
        e = s2_log(self.g, normalize(g_raw))
        mag = np.linalg.norm(e)
        if mag > self.max_step:
            e *= self.max_step / mag
        self.g = s2_exp(self.g, self.alpha * e)
        return self.g

# ---- Eye and head features -> gaze direction --------------------------------
FEATURES = ("eye_x", "eye_y", "lid", "head_yaw", "head_pitch")

def eye_features(pupil, inner, outer, lid_upper, lid_lower):
    """One eye, in camera pixels -> (x offset, y offset, lid opening).

    The pupil offset is from the midpoint of the eye corners, and it and the
    lid opening are in eye widths, so they survive moving closer or further.
    """
    width = np.linalg.norm(inner - outer) + 1e-6
    off = (pupil - 0.5 * (inner + outer)) / width
    return off[0], off[1], np.linalg.norm(lid_upper - lid_lower) / width

def head_angles(l_outer, r_outer, forehead, chin):
    """3D landmarks (camera pixels, z on the x scale) -> head (yaw, pitch), radians.

    Builds the face frame from the outer eye corners (x axis) and the
    forehead-to-chin line (y axis); yaw and pitch are of its normal.
    """
    xa = normalize(r_outer - l_outer)
    ya = chin - forehead
    ya = normalize(ya - np.dot(ya, xa) * xa)
    za = np.cross(xa, ya)
    return math.atan2(za[0], za[2]), math.asin(float(np.clip(-za[1], -1.0, 1.0)))

def combine_features(eye_L, eye_R, head):
    """Both eyes' features and head angles -> one feature vector (FEATURES)."""
    eyes = 0.5 * (np.array(eye_L, dtype=float) + np.array(eye_R, dtype=float))
    return np.array([*eyes, *head], dtype=float)

def gaze_direction(d, gain_y_up=GAIN_Y_UP, gain_y_down=GAIN_Y_DOWN):
    """Features relative to the centre pose -> unit gaze vector on S².

    The vector has x towards screen-right, y towards screen-down and z
    towards the screen, from the user's point of view.
    """
    eye_x, eye_y, lid, head_yaw, head_pitch = d
    sx = MIRROR * GAIN_X * eye_x
    sy = GAIN_Y * eye_y + GAIN_LID * lid
    sy *= gain_y_down if sy > 0 else gain_y_up   # image y grows downward
    yaw = math.asin(float(np.clip(sx, -CLAMP_FRAC, CLAMP_FRAC))) - MIRROR * HEAD_GAIN_YAW * head_yaw
    pitch = math.asin(float(np.clip(sy, -CLAMP_FRAC, CLAMP_FRAC))) + HEAD_GAIN_PITCH * head_pitch
    return np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch),
                     math.cos(pitch) * math.cos(yaw)])

# ---- Screen geometry -------------------------------------------------------
def tan_theta_max_from_screen(aspect):
    half_w = 0.5 * SCREEN_WIDTH_EYE_UNITS
    half_h = 0.5 * (SCREEN_WIDTH_EYE_UNITS / max(1e-6, aspect))
    return half_w / SCREEN_Z_EYE_UNITS, half_h / SCREEN_Z_EYE_UNITS

def project_to_screen(g, aspect):
    """Unit gaze vector -> offset from the neutral gaze, -1..1 at the screen edges."""
    tan_h, tan_v = tan_theta_max_from_screen(aspect)
    ix = (g[0] / max(1e-6, g[2])) / tan_h
    iy = (g[1] / max(1e-6, g[2])) / tan_v
    return ix, iy

def norm_to_pixels(nx, ny, width, height):
    """Normalised screen position (0..1) -> integer pixel inside the screen."""
    return (min(width - 1, max(0, int(nx * width))),
            min(height - 1, max(0, int(ny * height))))

def origin_norm(origin, width, height):
    """Eye centroid in camera pixels -> 0..1 in the user's point of view."""
    ox = float(origin[0]) / max(1, width)
    oy = float(origin[1]) / max(1, height)
    return (1.0 - ox if HEAD_MIRROR_X else ox), oy

def head_anchor(origin, width, height, ref=(0.5, 0.5)):
    """Eye centroid in camera pixels -> neutral gaze point on screen (0..1).

    `ref` is the eye centroid whose neutral gaze lands on the screen centre:
    the frame centre by default, or the calibrated head position.
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
    """
    i = float(np.clip(i, -1, 1))
    return anchor + i * ((1.0 - anchor) if i >= 0 else anchor)

# ---- Gaze regions ----------------------------------------------------------
def region_cell(v, n, prev=None, margin=REGION_HYSTERESIS):
    """Normalised coordinate (0..1) -> which of `n` equal bands it falls in.

    If `prev` is given, the cell only changes once `v` is more than `margin`
    past the edge of the previous cell.
    """
    cell = min(n - 1, max(0, int(v * n)))
    if prev is not None and cell != prev:
        if prev / n - margin <= v <= (prev + 1) / n + margin:
            return prev
    return cell

def region_name(col, row):
    """Grid cell (col 0..2, row 0..1) -> e.g. 'top left', 'bottom center'."""
    return f"{REGION_ROWS[row]} {REGION_COLS[col]}"
