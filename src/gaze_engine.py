"""
Gaze engine for project iris.

Turns camera frames into gaze results. The engine owns everything that carries
over from one frame to the next (face tracker, smoothing, centre calibration,
region hysteresis, dwell timer) and leans on gaze_math.py for the geometry.
It does no drawing and opens no camera or window, so any interface can drive it:

    engine = GazeEngine(dwell_s=1.0)
    result = engine.process(frame)          # BGR camera frame -> GazeResult
    if result.region is not None:
        print(result.nx, result.ny, result.region)
    if result.dwell and result.dwell.selected:
        print("selected", result.dwell.region)

`GazeResult.to_dict()` gives the same thing as plain JSON-ready data.
"""

import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

from . import gaze_math as gm

# ---- Config ----------------------------------------------------------------
SMOOTH_ALPHA = 0.35     # feature smoothing: lower is steadier but slower
BASELINE_ALPHA = 0.01   # drift rate of the uncalibrated baseline
ORIGIN_ALPHA = 0.18     # smoothing of the eye centroid (head position)

# MediaPipe Face Mesh landmark indices
PUPIL_L, PUPIL_R = 468, 473                 # iris centres
L_INNER, L_OUTER = 133, 33                  # eye corners
R_INNER, R_OUTER = 362, 263
LID_L, LID_R = (159, 145), (386, 374)       # upper and lower lid
SOCKET_L = (L_OUTER, L_INNER, *LID_L)       # corners + lids
SOCKET_R = (R_OUTER, R_INNER, *LID_R)
FOREHEAD, CHIN = 10, 152

# Single-point centre calibration. The target is shown for CALIB_SETTLE_S so
# the eyes can land on it, then pupil positions are collected for
# CALIB_CAPTURE_S and their median becomes the "looking at the centre" pose.
# With fewer than CALIB_MIN_SAMPLES tracked frames the capture starts over.
CALIB_SETTLE_S = 1.0
CALIB_CAPTURE_S = 1.2
CALIB_MIN_SAMPLES = 12

# Dwell: the face may be lost for this long (a blink) without the dwell on the
# current region starting over.
DWELL_GRACE_S = 0.3

# ---- Per-frame geometry ----------------------------------------------------
# Pure functions shared by GazeEngine.update() and the benchmark
# (benchmark/run_engine_py.py), so the benchmark scores the real engine.

def measure(landmarks, width, height):
    """Face Mesh landmarks -> the per-frame quantities the engine works from.

    Returns a dict with
      `features`    the gaze features (gaze_math.FEATURES), before the
                    "looking at the centre" baseline is subtracted
      `origin_raw`  midpoint of the two eye-socket centroids, camera pixels
    """
    w, h = width, height

    def px(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h], dtype=np.float32)

    def px3(i):   # Face Mesh z uses roughly the same scale as x
        return np.array([landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z * w])

    eye_L = gm.eye_features(px(PUPIL_L), px(L_INNER), px(L_OUTER), px(LID_L[0]), px(LID_L[1]))
    eye_R = gm.eye_features(px(PUPIL_R), px(R_INNER), px(R_OUTER), px(LID_R[0]), px(LID_R[1]))
    head = gm.head_angles(px3(L_OUTER), px3(R_OUTER), px3(FOREHEAD), px3(CHIN))

    # Head position: midpoint of the two eye socket centroids
    socket_L = np.mean([px(i) for i in SOCKET_L], axis=0)
    socket_R = np.mean([px(i) for i in SOCKET_R], axis=0)

    return {
        "features": gm.combine_features(eye_L, eye_R, head),
        "origin_raw": 0.5 * (socket_L + socket_R),
    }

def calibration_baseline(measurements):
    """Median features over frames of looking at the centre (the calibration)."""
    return np.median(np.array([m["features"] for m in measurements]), axis=0)

def settled_direction(m, base, gain_y_up=gm.GAIN_Y_UP, gain_y_down=gm.GAIN_Y_DOWN):
    """The direction GazeEngine.update() settles on when frame `m` is held.

    `base` is the baseline from calibration_baseline().
    """
    return gm.gaze_direction(m["features"] - base, gain_y_up, gain_y_down)

# ---- Structured result -----------------------------------------------------
@dataclass(frozen=True)
class CalibrationState:
    """Progress of a running centre calibration."""
    capturing: bool     # False while the eyes settle, True while sampling
    progress: float     # capture progress, 0..1

@dataclass(frozen=True)
class DwellState:
    """How long the gaze has rested on one region."""
    region: str         # the region being dwelt on
    elapsed: float      # seconds the gaze has stayed in it
    progress: float     # elapsed / dwell time, capped at 1
    selected: bool      # True on the single frame the dwell time is reached

@dataclass(frozen=True)
class GazeResult:
    """Everything the engine knows about one camera frame.

    `status` is "calibrating" while the centre target is up, "no_face" when no
    face was found, otherwise "tracking". The gaze fields (`nx` .. `anchor`)
    and `head` are only set when status is "tracking".
    """
    frame: int                      # frame index, counting every frame fed in
    t: float                        # seconds since the first frame
    status: str                     # "tracking" | "calibrating" | "no_face"
    face: bool                      # a face was found in this frame
    calibrated: bool                # a centre calibration has completed
    calibration: Optional[CalibrationState] = None   # set while calibrating
    nx: Optional[float] = None      # gaze on screen, 0..1 from the left edge
    ny: Optional[float] = None      # gaze on screen, 0..1 from the top edge
    x: Optional[int] = None         # gaze in pixels on the engine's screen_size
    y: Optional[int] = None
    region: Optional[str] = None    # e.g. "top left"
    col: Optional[int] = None       # region column, 0..2 from the left
    row: Optional[int] = None       # region row, 0..1 from the top
    anchor: Optional[Tuple[float, float]] = None     # neutral gaze point, 0..1
    head: Optional[Tuple[float, float]] = None       # midpoint between the eyes, 0..1
    dwell: Optional[DwellState] = None               # set when dwell is enabled

    def to_dict(self):
        """Plain dict (nested dataclasses included), ready for json.dumps."""
        return asdict(self)

# ---- Centre calibration ----------------------------------------------------
class CenterCalibration:
    """Single-point calibration: the user looks at a target in the centre.

    While `active`, feed it one observation per tracked frame. When it
    finishes it holds the gaze features (pupil offsets in eye widths, so they
    survive the user moving closer or further away, lid opening and head
    rotation) and the head position that together mean "looking at the
    screen centre".
    """

    def __init__(self, active=True):
        self.done = False
        self.base = None                   # gaze features (gaze_math.FEATURES)
        self.origin = None                 # normalised eye centroid
        self.start()
        self.active = active

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
        """Advance the clock without an observation (no face this frame).
        Returns True when calibration completes."""
        if self.active and self._t0 is not None \
                and self._elapsed(now) >= CALIB_SETTLE_S + CALIB_CAPTURE_S:
            return self._finish()
        return False

    def feed(self, now, features, origin):
        """Add one tracked frame. Returns True when calibration completes."""
        if not self.active:
            return False
        if self._t0 is None:
            self._t0 = now
        elapsed = self._elapsed(now)
        if elapsed < CALIB_SETTLE_S:
            return False
        self._samples.append((*features, *origin))
        if elapsed < CALIB_SETTLE_S + CALIB_CAPTURE_S:
            return False
        return self._finish()

    def _finish(self):
        if len(self._samples) < CALIB_MIN_SAMPLES:
            self.start()            # not enough data: try again
            return False
        # Median, so blinks and stray saccades during the capture are ignored.
        med = np.median(np.array(self._samples), axis=0)
        self.base = med[:-2]
        self.origin = (float(med[-2]), float(med[-1]))
        self.active = False
        self.done = True
        return True

# ---- Dwell -----------------------------------------------------------------
class DwellTimer:
    """Fires once when the gaze has rested on one region for `seconds`.

    The timer restarts whenever the region changes, and a region that has
    fired does not fire again until the gaze has left it and come back.
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self.reset()

    def reset(self):
        self._region = None
        self._t0 = None
        self._fired = False
        self._last_seen = None

    def update(self, region, now):
        if region != self._region:
            self._region, self._t0, self._fired = region, now, False
        self._last_seen = now
        elapsed = now - self._t0
        selected = not self._fired and elapsed >= self.seconds
        self._fired = self._fired or selected
        return DwellState(region, elapsed, min(1.0, elapsed / self.seconds), selected)

    def lost(self, now):
        """No gaze this frame: start over once the face has been gone too long."""
        if self._last_seen is not None and now - self._last_seen > DWELL_GRACE_S:
            self.reset()

# ---- Engine ----------------------------------------------------------------
class GazeEngine:
    """Per-frame gaze tracker: feed it frames, get a GazeResult for each.

    calibrate    run the centre calibration on the first frames
    dwell_s      seconds the gaze must rest on a region to select it
                 (None = dwell off)
    screen_size  (width, height) in pixels for GazeResult.x / .y
                 (None = the camera frame size)

    `gain_y_up` and `gain_y_down` may be changed at any time.
    """

    def __init__(self, calibrate=True, dwell_s=None, screen_size=None):
        if dwell_s is not None and dwell_s <= 0:
            raise ValueError("dwell_s must be positive")
        self.screen_size = screen_size
        self.gain_y_up = gm.GAIN_Y_UP
        self.gain_y_down = gm.GAIN_Y_DOWN
        self.dwell_s = dwell_s
        self.calibrated_at = None   # GazeResult.t of the last completed calibration

        self._calib = CenterCalibration(active=calibrate)
        self._dwell = DwellTimer(dwell_s) if dwell_s is not None else None
        self._filter = gm.GazeS2Filter()
        self._base = None
        self._smooth = None
        self._origin = None
        self._col = self._row = None
        self._frame = -1
        self._t_start = None
        self._face_mesh = None

    # -- public API ----------------------------------------------------------
    def process(self, frame_bgr, now=None):
        """Track one BGR camera frame (as read by OpenCV) -> GazeResult."""
        if self._face_mesh is None:
            import mediapipe as mp   # deferred: update() works without it
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        h, w = frame_bgr.shape[:2]
        res = self._face_mesh.process(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        faces = res.multi_face_landmarks
        return self.update(faces[0].landmark if faces else None, w, h, now)

    def update(self, landmarks, width, height, now=None):
        """Advance one frame from Face Mesh landmarks -> GazeResult.

        `landmarks` is the refined 478-point list (normalised .x / .y), or
        None when no face was found. `now` is a monotonic time in seconds.
        """
        now = time.monotonic() if now is None else now
        if self._t_start is None:
            self._t_start = now
        self._frame += 1
        w, h = width, height
        if self.screen_size is None:
            self.screen_size = (w, h)

        if landmarks is None:
            if self._calib.tick(now):
                self._on_calibrated(now)
            if self._dwell is not None:
                self._dwell.lost(now)
            return self._result(now, face=False)

        m = measure(landmarks, w, h)
        origin_raw = m["origin_raw"]
        origin = self._origin = gm.smooth(self._origin, origin_raw, ORIGIN_ALPHA)

        # Centre calibration
        if self._calib.active:
            if not self._calib.feed(now, m["features"],
                                    gm.origin_norm(origin_raw, w, h)):
                return self._result(now, face=True)
            self._on_calibrated(now)

        if self._calib.done:
            # Fixed baseline: the features captured while the user looked at
            # the centre target.
            self._base = self._calib.base
        else:
            # Uncalibrated: slowly adapting baseline.
            self._base = gm.smooth(self._base, m["features"], BASELINE_ALPHA)

        self._smooth = gm.smooth(self._smooth, m["features"] - self._base, SMOOTH_ALPHA)

        # Features relative to the centre pose -> gaze direction on S² -> screen offset
        g = self._filter.update(gm.gaze_direction(self._smooth, self.gain_y_up, self.gain_y_down))
        ix, iy = gm.project_to_screen(g, w / max(1, h))

        # Split the screen at the head anchor rather than at its centre,
        # so an off-centre head still reaches every row and column.
        anchor_ref = self._calib.origin if self._calib.done else (0.5, 0.5)
        anchor = gm.head_anchor(origin, w, h, anchor_ref)
        nx = gm.anchored_norm(ix, anchor[0])   # 0 = left edge, 1 = right
        ny = gm.anchored_norm(iy, anchor[1])   # 0 = top edge,  1 = bottom

        self._col = gm.region_cell(nx, len(gm.REGION_COLS), self._col)
        self._row = gm.region_cell(ny, len(gm.REGION_ROWS), self._row)
        region = gm.region_name(self._col, self._row)
        dwell = self._dwell.update(region, now) if self._dwell is not None else None

        x, y = gm.norm_to_pixels(nx, ny, *self.screen_size)
        return self._result(now, face=True, nx=nx, ny=ny, x=x, y=y, region=region,
                            col=self._col, row=self._row, anchor=anchor,
                            head=gm.origin_norm(origin_raw, w, h), dwell=dwell)

    @property
    def calibrating(self):
        """True while the centre calibration is running."""
        return self._calib.active

    def recalibrate(self):
        """Show the centre target again and recapture the centre pose."""
        self._calib.start()
        if self._dwell is not None:
            self._dwell.reset()

    def close(self):
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- internals -----------------------------------------------------------
    def _on_calibrated(self, now):
        """Start the filters fresh from the calibrated centre."""
        self._smooth = None
        self._filter = gm.GazeS2Filter()
        self._col = self._row = None
        self.calibrated_at = now - self._t_start
        if self._dwell is not None:
            self._dwell.reset()

    def _result(self, now, face, **gaze):
        calibration = None
        if self._calib.active:
            calibration = CalibrationState(self._calib.capturing(now),
                                           self._calib.progress(now))
        status = "calibrating" if calibration else "tracking" if face else "no_face"
        return GazeResult(frame=self._frame, t=now - self._t_start, status=status,
                          face=face, calibrated=self._calib.done,
                          calibration=calibration, **gaze)
