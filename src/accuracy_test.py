"""
Gaze accuracy test for project iris.

Shows the user a sequence of targets, records where the engine puts the gaze
while they look at each one, and turns that into the usual eye-tracking
statistics:

  accuracy    mean angular error between the gaze and the target, in degrees
  precision   how steady the gaze is while fixating: RMS sample-to-sample
              distance and standard deviation around the mean, in degrees
  bias        the average offset, horizontal and vertical, in degrees
  region hits how often the reported region was the target's region
  data loss   frames with no face while a target was being recorded

Angles are visual angles from an eye centred in front of the screen. By
default the screen geometry is the one the gaze model assumes (a screen
SCREEN_WIDTH_EYE_UNITS wide at SCREEN_Z_EYE_UNITS, about 24 degrees across);
pass the real screen width and viewing distance for angles in your own setup.

No drawing, camera or window code: feed it a GazeResult per frame and read
`target` / `progress` to draw, then `report()` when `done`.
"""

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import gaze_math as gm

# ---- Config ----------------------------------------------------------------
# One target at the centre, then the centre of each of the six regions, so
# every target has an unambiguous region.
TARGETS = ((0.5, 0.5),
           (1 / 6, 0.25), (0.5, 0.25), (5 / 6, 0.25),
           (5 / 6, 0.75), (0.5, 0.75), (1 / 6, 0.75))
SETTLE_S = 1.0          # time for the eyes to land on a new target (not recorded)
CAPTURE_S = 1.0         # time recorded per target
MIN_SAMPLES = 5         # fewer tracked frames than this and the target is skipped

# ---- Geometry --------------------------------------------------------------
def screen_direction(nx, ny, aspect, width, distance):
    """Screen point (0..1) -> unit view direction from an eye centred in front
    of the screen. `width` and `distance` share any unit."""
    height = width / max(1e-6, aspect)
    v = np.array([(nx - 0.5) * width, (ny - 0.5) * height, distance], dtype=float)
    return v / np.linalg.norm(v)

def angle_deg(a, b):
    """Angle between two unit vectors, in degrees."""
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))

def yaw_pitch_deg(v):
    """Unit direction -> (horizontal, vertical) angle in degrees, right/down positive."""
    return math.degrees(math.atan2(v[0], v[2])), math.degrees(math.atan2(v[1], v[2]))

def target_region(nx, ny):
    """The region a target is in, or None if it sits on a grid line (like the
    centre target), where either neighbouring region is a fair answer."""
    for v, n in ((nx, len(gm.REGION_COLS)), (ny, len(gm.REGION_ROWS))):
        k = round(v * n)
        if 0 < k < n and abs(v * n - k) < 1e-6:
            return None
    return gm.region_name(gm.region_cell(nx, len(gm.REGION_COLS)),
                          gm.region_cell(ny, len(gm.REGION_ROWS)))

# ---- Results ---------------------------------------------------------------
@dataclass(frozen=True)
class TargetStats:
    """What the test measured at one target. The angle fields are None when
    too few frames were tracked to score it."""
    target: Tuple[float, float]         # target position on screen, 0..1
    region: Optional[str]               # the target's region (None on a grid line)
    samples: int                        # tracked frames recorded
    lost: int                           # frames with no face while recording
    mean: Optional[Tuple[float, float]] = None   # mean gaze position, 0..1
    error_deg: Optional[float] = None   # mean angular error of the samples
    bias_x_deg: Optional[float] = None  # mean gaze minus target, + = right
    bias_y_deg: Optional[float] = None  # mean gaze minus target, + = down
    rms_s2s_deg: Optional[float] = None # RMS sample-to-sample distance
    sd_deg: Optional[float] = None      # RMS distance from the mean gaze
    region_hit: Optional[float] = None  # fraction of samples in `region`
                                        # (None when `region` is None)

@dataclass(frozen=True)
class AccuracyReport:
    """Summary over all targets; per-target detail is in `targets`."""
    t: float                            # GazeResult.t when the test finished
    calibrated: bool
    geometry: dict                      # screen width, distance, aspect, units
    targets: List[TargetStats]
    scored: int                         # targets with enough samples
    accuracy_deg: Optional[float] = None
    median_deg: Optional[float] = None
    worst_deg: Optional[float] = None
    worst_region: Optional[str] = None  # e.g. "top left", or "center"
    rms_s2s_deg: Optional[float] = None
    sd_deg: Optional[float] = None
    bias_x_deg: Optional[float] = None
    bias_y_deg: Optional[float] = None
    region_hit_rate: Optional[float] = None
    data_loss: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    def lines(self):
        """The report as short lines of text, for the preview and the terminal."""
        out = ["Accuracy test results"
               + ("" if self.calibrated else " (uncalibrated)")]
        if self.scored == 0:
            out.append("No target had enough tracked frames to score.")
            return out
        h = "right" if self.bias_x_deg >= 0 else "left"
        v = "down" if self.bias_y_deg >= 0 else "up"
        out += [
            f"Accuracy: {self.accuracy_deg:.1f} deg mean error   "
            f"median {self.median_deg:.1f}   worst {self.worst_deg:.1f} ({self.worst_region})",
            f"Precision: {self.rms_s2s_deg:.2f} deg RMS sample-to-sample   "
            f"{self.sd_deg:.2f} deg SD",
            f"Bias: {abs(self.bias_x_deg):.1f} deg {h}, {abs(self.bias_y_deg):.1f} deg {v}",
            ("" if self.region_hit_rate is None
             else f"Region hits: {100 * self.region_hit_rate:.0f}%   ") +
            f"Data loss: {100 * self.data_loss:.0f}%   "
            f"Targets scored: {self.scored}/{len(self.targets)}",
        ]
        g = self.geometry
        out.append(f"Screen {g['screen_width']:g} wide at {g['distance']:g} {g['units']}")
        return out

# ---- Test runner -----------------------------------------------------------
class AccuracyTest:
    """Steps through TARGETS, one GazeResult per frame.

    aspect        screen width / height, for turning positions into angles
    screen_width  physical screen width, in the same unit as `distance`
    distance      eye-to-screen distance
    units         label for those two, shown in the report
    """

    def __init__(self, aspect, screen_width=gm.SCREEN_WIDTH_EYE_UNITS,
                 distance=gm.SCREEN_Z_EYE_UNITS, units="eye widths (model default)",
                 targets=TARGETS):
        self.aspect = aspect
        self.geometry = {"screen_width": screen_width, "distance": distance,
                         "aspect": round(aspect, 4), "units": units}
        self.targets = list(targets)
        self.index = 0
        self._t0 = None             # when the current target appeared
        self._samples = []          # (nx, ny, region) while recording
        self._lost = 0
        self._stats = []
        self._calibrated = False
        self._t_end = None

    @property
    def done(self):
        return self.index >= len(self.targets)

    @property
    def target(self):
        """The target to show now, (nx, ny), or None when the test is over."""
        return None if self.done else self.targets[self.index]

    def capturing(self, t):
        return self._t0 is not None and t - self._t0 >= SETTLE_S

    def progress(self, t):
        """Recording progress for the current target, 0..1."""
        if self._t0 is None:
            return 0.0
        return min(1.0, max(0.0, (t - self._t0 - SETTLE_S) / CAPTURE_S))

    def update(self, result):
        """Feed one GazeResult. Advances to the next target when recorded."""
        if self.done:
            return
        t = result.t
        if self._t0 is None:
            self._t0 = t
        if self.capturing(t):
            if result.region is not None:
                self._samples.append((result.nx, result.ny, result.region))
            elif not result.face:
                self._lost += 1
        self._calibrated = result.calibrated
        if t - self._t0 >= SETTLE_S + CAPTURE_S:
            self._stats.append(self._score(self.targets[self.index]))
            self.index += 1
            self._t0, self._samples, self._lost = None, [], 0
            if self.done:
                self._t_end = t

    def _direction(self, nx, ny):
        return screen_direction(nx, ny, self.aspect, self.geometry["screen_width"],
                                self.geometry["distance"])

    def _score(self, target):
        region = target_region(*target)
        n = len(self._samples)
        if n < MIN_SAMPLES:
            return TargetStats(target, region, n, self._lost)
        pts = np.array([(s[0], s[1]) for s in self._samples])
        dirs = [self._direction(x, y) for x, y in pts]
        tdir = self._direction(*target)
        mean = pts.mean(axis=0)
        mdir = self._direction(*mean)
        my, mp = yaw_pitch_deg(mdir)
        ty, tp = yaw_pitch_deg(tdir)
        s2s = [angle_deg(a, b) for a, b in zip(dirs, dirs[1:])]
        return TargetStats(
            target=target, region=region, samples=n, lost=self._lost,
            mean=(float(mean[0]), float(mean[1])),
            error_deg=float(np.mean([angle_deg(d, tdir) for d in dirs])),
            bias_x_deg=my - ty, bias_y_deg=mp - tp,
            rms_s2s_deg=float(np.sqrt(np.mean(np.square(s2s)))),
            sd_deg=float(np.sqrt(np.mean([angle_deg(d, mdir) ** 2 for d in dirs]))),
            region_hit=(None if region is None
                        else sum(s[2] == region for s in self._samples) / n),
        )

    def report(self):
        """Summary of the finished test (AccuracyReport)."""
        stats = self._stats
        scored = [s for s in stats if s.error_deg is not None]
        frames = sum(s.samples + s.lost for s in stats)
        base = dict(t=self._t_end, calibrated=self._calibrated,
                    geometry=self.geometry, targets=stats, scored=len(scored))
        if not scored:
            return AccuracyReport(**base)
        errors = [s.error_deg for s in scored]
        worst = max(scored, key=lambda s: s.error_deg)
        in_region = [s for s in scored if s.region_hit is not None]
        hit_samples = sum(s.samples for s in in_region)
        return AccuracyReport(
            **base,
            accuracy_deg=float(np.mean(errors)),
            median_deg=float(np.median(errors)),
            worst_deg=worst.error_deg, worst_region=worst.region or "center",
            rms_s2s_deg=float(np.mean([s.rms_s2s_deg for s in scored])),
            sd_deg=float(np.mean([s.sd_deg for s in scored])),
            bias_x_deg=float(np.mean([s.bias_x_deg for s in scored])),
            bias_y_deg=float(np.mean([s.bias_y_deg for s in scored])),
            region_hit_rate=(sum(s.region_hit * s.samples for s in in_region) / hit_samples
                             if hit_samples else None),
            data_loss=sum(s.lost for s in stats) / max(1, frames),
        )
