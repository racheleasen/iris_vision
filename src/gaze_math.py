"""
Gaze math for project iris (Gaussian binocular S² model).

Pure functions and one small filter: pupil displacement -> unit sphere ->
screen position -> 3x3 region. No camera, no MediaPipe and no drawing in here,
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
GAUSS_WIDTH = 11.7      # width of the Gaussian weighting on pupil displacement
MIRROR = -1.0

# Vertical gain, split by direction. Looking down, the lids and eye corners
# travel with the eye, so the measured pupil shift is much smaller than for
# an equal upward look. These are the defaults; main.py tunes them live.
GAIN_Y_UP = 1.0
GAIN_Y_DOWN = 2.0

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

# 3x3 gaze regions. A new region is only reported once the gaze is this far
# (as a fraction of the screen) past the boundary, so it does not flicker
# when the gaze sits on a grid line.
REGION_ROWS = ("top", "middle", "bottom")
REGION_COLS = ("left", "center", "right")
REGION_NAMES = [f"{r} {c}" for r in REGION_ROWS for c in REGION_COLS]
REGION_HYSTERESIS = 0.03

# ---- Utils -----------------------------------------------------------------
def median_radius(iris_pts, pupil):
    return np.median(np.linalg.norm(iris_pts - pupil, axis=1))

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

# ---- Pupil displacement -> gaze direction ----------------------------------
def fuse_binocular(disp_L, disp_R, gain_y_up=GAIN_Y_UP, gain_y_down=GAIN_Y_DOWN):
    """Gaussian binocular fusion of the two pupil displacements.

    `disp_L` / `disp_R` are (x, y) displacements from the baseline in iris
    radii. Each eye is weighted down the further it is from its baseline.
    Returns the fused (ux, uy) in the user's point of view.
    """
    xL, yL = disp_L
    xR, yR = disp_R
    wL = math.exp(-(xL * xL + yL * yL) / GAUSS_WIDTH)
    wR = math.exp(-(xR * xR + yR * yR) / GAUSS_WIDTH)
    gx = xL * wL
    gy = yR * wR
    gy *= gain_y_down if gy > 0 else gain_y_up   # image y grows downward
    return MIRROR * gx, MIRROR * gy

def sphere_lift(ux, uy):
    """Planar gaze offset (ux, uy) -> unit gaze vector on S² (z towards screen)."""
    r2 = ux * ux + uy * uy
    if r2 > CLAMP_FRAC * CLAMP_FRAC:
        s = CLAMP_FRAC / math.sqrt(r2)
        ux, uy = ux * s, uy * s
        r2 = ux * ux + uy * uy
    uz = math.sqrt(max(0.0, 1.0 - r2))
    return normalize(np.array([ux, -uy, uz]))

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
