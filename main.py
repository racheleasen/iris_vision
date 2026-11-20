"""
project iris: webcam gaze tracking demo (angular binocular S² model).

Opens the camera, feeds every frame to the gaze engine and draws the result.
The tracking is in src/gaze_engine.py and the geometry in src/gaze_math.py;
this file is only the camera loop, the preview window, the keys and the export.

    python3 main.py                                  # live preview
    python3 main.py --dwell 1.0                      # select a region by resting on it
    python3 main.py --export gaze.json --screen-size 1920x1080

On start, look at the centre target and hold still: the eye and head pose captured
there becomes "looking at the centre" for this person, head position and camera.
Screen coordinates use a top-left origin, x right and y down, from the user's
point of view. README.md documents the options and the export format.

Keys: ESC quit, C recalibrate, T accuracy test, [ ] vertical gain looking down,
- = looking up. Clicking the "Test accuracy" button also starts the test.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import cv2

from src.accuracy_test import AccuracyTest
from src.gaze_engine import GazeEngine
from src.gaze_math import (HEAD_MIRROR_X, REGION_COLS, REGION_NAMES, REGION_ROWS,
                           norm_to_pixels)

EXPORT_FORMAT = "project-iris-gaze"
EXPORT_VERSION = 1

WINDOW = "project iris"
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREEN = (255, 255, 255), (0, 255, 0)

# ---- JSON export -----------------------------------------------------------
class GazeExporter:
    """Collects per-frame gaze samples and writes them to one JSON file."""

    def __init__(self, path):
        self.path = path
        self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.samples = []
        self.selections = []
        self.accuracy_tests = []

    def add(self, result):
        if result.region is None:       # no face, or the calibration target is up
            return
        self.samples.append({
            "frame": result.frame, "t": round(result.t, 4),
            "x": result.x, "y": result.y,
            "nx": round(result.nx, 4), "ny": round(result.ny, 4),
            "region": result.region,
        })
        if result.dwell is not None and result.dwell.selected:
            self.selections.append({"frame": result.frame, "t": round(result.t, 4),
                                    "region": result.region})

    def save(self, engine):
        width, height = engine.screen_size or (None, None)
        calibration = None
        if engine.calibrated_at is not None:
            calibration = {"type": "single-point", "target": "center",
                           "t": round(engine.calibrated_at, 4)}
        doc = {
            "format": EXPORT_FORMAT,
            "version": EXPORT_VERSION,
            "created": self.created,
            "screen": {"width": width, "height": height},
            "coordinates": "origin top-left, x right, y down, user's point of view",
            "regions": REGION_NAMES,
            "calibration": calibration,
            "dwell_seconds": engine.dwell_s,
            "selections": self.selections,
            "accuracy_tests": self.accuracy_tests,
            "samples": self.samples,
        }
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        os.replace(tmp, self.path)

# ---- UI state --------------------------------------------------------------
class Ui:
    """What the preview is showing besides live tracking: a running accuracy
    test, or the report of the last one."""

    def __init__(self):
        self.test = None        # AccuracyTest while one runs
        self.report = None      # AccuracyReport until dismissed
        self.button = None      # (x0, y0, x1, y1) of the test button, if drawn
        self.clicked = False    # the button was clicked since the last frame

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.report is not None:
            self.report = None              # a click dismisses the report
        elif self.button is not None:
            x0, y0, x1, y1 = self.button
            self.clicked = x0 <= x <= x1 and y0 <= y <= y1

# ---- Preview ---------------------------------------------------------------
def draw_target(frame, c, capturing, progress, text):
    """Dim the preview and draw a target at pixel `c` with capture progress."""
    h, w, _ = frame.shape
    frame = cv2.convertScaleAbs(frame, alpha=0.3)
    cv2.circle(frame, c, 22, WHITE, 2, cv2.LINE_AA)
    cv2.circle(frame, c, 6, (0, 0, 255), -1, cv2.LINE_AA)
    if capturing:
        cv2.ellipse(frame, c, (34, 34), -90, 0, 360 * progress, GREEN, 4, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.9, 2)
    ty = c[1] - 60 if c[1] > h // 3 else c[1] + 80
    tx = min(w - tw - 10, max(10, c[0] - tw // 2))
    cv2.putText(frame, text, (tx, ty), FONT, 0.9, WHITE, 2, cv2.LINE_AA)
    return frame

def draw_calibration(frame, calibration):
    h, w, _ = frame.shape
    text = "Hold still..." if calibration.capturing else "Look at the dot"
    return draw_target(frame, (w // 2, h // 2), calibration.capturing,
                       calibration.progress, text)

def draw_test(frame, test, t):
    h, w, _ = frame.shape
    capturing = test.capturing(t)
    text = f"{'Hold still' if capturing else 'Look at the dot'}  " \
           f"{test.index + 1}/{len(test.targets)}"
    return draw_target(frame, norm_to_pixels(*test.target, w, h), capturing,
                       test.progress(t), text)

def draw_report(frame, report):
    """Dim the preview; draw each target, the mean gaze for it and the error."""
    h, w, _ = frame.shape
    frame = cv2.convertScaleAbs(frame, alpha=0.25)
    for s in report.targets:
        tc = norm_to_pixels(*s.target, w, h)
        cv2.circle(frame, tc, 14, WHITE, 2, cv2.LINE_AA)
        if s.mean is None:
            cv2.putText(frame, "lost", (tc[0] + 18, tc[1] + 5), FONT, 0.5, WHITE, 1,
                        cv2.LINE_AA)
            continue
        mc = norm_to_pixels(*s.mean, w, h)
        cv2.line(frame, tc, mc, GREEN, 1, cv2.LINE_AA)
        cv2.circle(frame, mc, 7, GREEN, -1, cv2.LINE_AA)
        cv2.putText(frame, f"{s.error_deg:.1f} deg", (tc[0] + 18, tc[1] + 5), FONT,
                    0.5, WHITE, 1, cv2.LINE_AA)
    lines = report.lines() + ["Click or press any key to continue, T to test again"]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20, 30 + 26 * i), FONT, 0.6 if i else 0.75,
                    GREEN if i == 0 else WHITE, 2 if i == 0 else 1, cv2.LINE_AA)
    return frame

def draw_button(frame, ui):
    """The 'Test accuracy' button in the top-right corner."""
    h, w, _ = frame.shape
    text = "Test accuracy (T)"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    x1, y0 = w - 15, 15
    x0, y1 = x1 - tw - 24, y0 + th + 20
    cv2.rectangle(frame, (x0, y0), (x1, y1), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), GREEN, 2)
    cv2.putText(frame, text, (x0 + 12, y1 - 10), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
    ui.button = (x0, y0, x1, y1)

def draw(frame, result, engine, ui):
    """Camera frame + GazeResult -> the frame to show in the preview window."""
    # Flip to a selfie view so overlays in the user's point of view line up
    # with the face. Tracking used the unflipped frame.
    if HEAD_MIRROR_X:
        frame = cv2.flip(frame, 1)
    ui.button = None
    if result.calibration is not None:
        return draw_calibration(frame, result.calibration)
    if ui.test is not None:
        # No gaze dot during the test, so the user looks at the target rather
        # than chasing the dot.
        return draw_test(frame, ui.test, result.t)
    if ui.report is not None:
        return draw_report(frame, ui.report)
    draw_button(frame, ui)
    if result.region is None:
        return frame

    # 3x2 region grid, active cell highlighted
    h, w, _ = frame.shape
    cols, rows = len(REGION_COLS), len(REGION_ROWS)
    for k in range(1, cols):
        cv2.line(frame, (k * w // cols, 0), (k * w // cols, h), (90, 90, 90), 1)
    for k in range(1, rows):
        cv2.line(frame, (0, k * h // rows), (w, k * h // rows), (90, 90, 90), 1)
    x0, y0 = result.col * w // cols, result.row * h // rows
    x1, y1 = (result.col + 1) * w // cols - 1, (result.row + 1) * h // rows - 1
    dwell = result.dwell
    selected = dwell is not None and dwell.progress >= 1.0
    cv2.rectangle(frame, (x0, y0), (x1, y1), GREEN, 8 if selected else 2)
    if dwell is not None and not selected:
        # Dwell progress: a bar growing along the bottom of the active cell
        cv2.rectangle(frame, (x0, y1 - 10),
                      (x0 + int(dwell.progress * (x1 - x0)), y1), GREEN, -1)

    # Head marker (midpoint between the eyes, so it sits on the face in the
    # preview) and gaze point
    cv2.drawMarker(frame, norm_to_pixels(*result.head, w, h),
                   (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(frame, norm_to_pixels(result.nx, result.ny, w, h), 18, GREEN, -1,
               cv2.LINE_AA)

    cv2.putText(frame, "project iris - webcam gaze tracking demo",
                (20, h - 20), FONT, 0.6, WHITE, 2)
    status = (f"gain up {engine.gain_y_up:.1f} (-/=)   down {engine.gain_y_down:.1f} ([/])   "
              f"{'calibrated' if result.calibrated else 'uncalibrated'} (C)")
    if dwell is not None:
        status += f"   dwell {engine.dwell_s:g}s"
    cv2.putText(frame, status, (20, 30), FONT, 0.6, WHITE, 2)
    cv2.putText(frame, f"{'selected' if selected else 'looking'} {result.region}",
                (20, 60), FONT, 0.8, GREEN, 2)
    return frame

def new_test(engine, args):
    width, height = engine.screen_size
    if args.screen_cm is not None:
        return AccuracyTest(width / height, args.screen_cm, args.distance_cm, "cm")
    return AccuracyTest(width / height)

def handle_key(key, engine, ui, args):
    """Handle one key press. Returns False when the user quits."""
    if key == 27:
        return False
    if ui.report is not None and key != 255:
        ui.report = None                    # any key dismisses the report...
        if key not in (ord('t'), ord('T')):
            return True                     # ...and T also starts a new test
    if key in (ord('t'), ord('T')) or ui.clicked:
        ui.clicked = False
        if not engine.calibrating and ui.test is None:
            ui.test = new_test(engine, args)
    elif key in (ord('c'), ord('C')):
        ui.test = None
        engine.recalibrate()
    elif key == ord(']'):
        engine.gain_y_down += 0.1
    elif key == ord('['):
        engine.gain_y_down = max(0.1, engine.gain_y_down - 0.1)
    elif key == ord('='):
        engine.gain_y_up += 0.1
    elif key == ord('-'):
        engine.gain_y_up = max(0.1, engine.gain_y_up - 0.1)
    return True

# ---- CLI -------------------------------------------------------------------
def parse_screen_size(text):
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("screen size must be positive")
    return width, height

def parse_dwell(text):
    try:
        seconds = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of seconds, got {text!r}")
    if not 0 < seconds < float("inf"):
        raise argparse.ArgumentTypeError("dwell time must be positive")
    return seconds

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="project iris: webcam gaze tracking demo.")
    p.add_argument("--export", metavar="PATH",
                   help="write per-frame screen coordinates to this JSON file on exit")
    p.add_argument("--screen-size", metavar="WxH", type=parse_screen_size,
                   help="screen size in pixels for exported x/y, e.g. 1920x1080 "
                        "(default: camera frame size)")
    p.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip the centre calibration on start (press C to run it later)")
    p.add_argument("--dwell", metavar="SECONDS", type=parse_dwell,
                   help="select a region by resting the gaze on it for this long, "
                        "e.g. 1.0 (default: off)")
    p.add_argument("--screen-cm", metavar="CM", type=float,
                   help="physical screen width, for accuracy test angles in your setup "
                        "(use with --distance-cm)")
    p.add_argument("--distance-cm", metavar="CM", type=float,
                   help="distance from your eyes to the screen (use with --screen-cm)")
    args = p.parse_args(argv)
    if (args.screen_cm is None) != (args.distance_cm is None):
        p.error("--screen-cm and --distance-cm must be given together")
    for name in ("screen_cm", "distance_cm"):
        value = getattr(args, name)
        if value is not None and not 0 < value < float("inf"):
            p.error(f"--{name.replace('_', '-')} must be positive")
    return args

# ---- Main ------------------------------------------------------------------
def run(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    engine = GazeEngine(calibrate=not args.no_calibrate, dwell_s=args.dwell,
                        screen_size=args.screen_size)
    exporter = GazeExporter(args.export) if args.export else None
    ui = Ui()
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, ui.on_mouse)
    print("ESC to quit, C to recalibrate, T to test accuracy\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = engine.process(frame)
            if exporter is not None:
                exporter.add(result)
            if ui.test is not None:
                ui.test.update(result)
                if ui.test.done:
                    ui.report, ui.test = ui.test.report(), None
                    print("\n".join(ui.report.lines()) + "\n")
                    if exporter is not None:
                        exporter.accuracy_tests.append(ui.report.to_dict())
            cv2.imshow(WINDOW, draw(frame, result, engine, ui))
            if not handle_key(cv2.waitKey(1) & 0xFF, engine, ui, args):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()
        if exporter is not None:
            exporter.save(engine)
            print(f"\nSaved {len(exporter.samples)} gaze samples to {exporter.path}")


if __name__ == "__main__":
    run(parse_args())
