"""
project iris: webcam gaze tracking demo (Gaussian binocular S² model).

Opens the camera, feeds every frame to the gaze engine and draws the result.
The tracking is in src/gaze_engine.py and the geometry in src/gaze_math.py;
this file is only the camera loop, the preview window, the keys and the export.

    python3 main.py                                  # live preview
    python3 main.py --dwell 1.0                      # select a region by resting on it
    python3 main.py --export gaze.json --screen-size 1920x1080

On start, look at the centre target and hold still: the pupil position captured
there becomes "looking at the centre" for this person, head position and camera.
Screen coordinates use a top-left origin, x right and y down, from the user's
point of view. README.md documents the options and the export format.

Keys: ESC quit, C recalibrate, [ ] vertical gain looking down, - = looking up.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import cv2

from src.gaze_engine import GazeEngine
from src.gaze_math import HEAD_MIRROR_X, REGION_NAMES, norm_to_pixels

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
            "samples": self.samples,
        }
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        os.replace(tmp, self.path)

# ---- Preview ---------------------------------------------------------------
def draw_calibration(frame, calibration):
    """Dim the preview and draw the centre target with capture progress."""
    h, w, _ = frame.shape
    frame = cv2.convertScaleAbs(frame, alpha=0.3)
    c = (w // 2, h // 2)
    cv2.circle(frame, c, 22, WHITE, 2, cv2.LINE_AA)
    cv2.circle(frame, c, 6, (0, 0, 255), -1, cv2.LINE_AA)
    text = "Look at the dot"
    if calibration.capturing:
        cv2.ellipse(frame, c, (34, 34), -90, 0, 360 * calibration.progress,
                    GREEN, 4, cv2.LINE_AA)
        text = "Hold still..."
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.9, 2)
    cv2.putText(frame, text, ((w - tw) // 2, h // 2 - 60), FONT, 0.9, WHITE,
                2, cv2.LINE_AA)
    return frame

def draw(frame, result, engine):
    """Camera frame + GazeResult -> the frame to show in the preview window."""
    # Flip to a selfie view so overlays in the user's point of view line up
    # with the face. Tracking used the unflipped frame.
    if HEAD_MIRROR_X:
        frame = cv2.flip(frame, 1)
    if result.calibration is not None:
        return draw_calibration(frame, result.calibration)
    if result.region is None:
        return frame

    # 3x3 region grid, active cell highlighted
    h, w, _ = frame.shape
    for k in (1, 2):
        cv2.line(frame, (k * w // 3, 0), (k * w // 3, h), (90, 90, 90), 1)
        cv2.line(frame, (0, k * h // 3), (w, k * h // 3), (90, 90, 90), 1)
    x0, y0 = result.col * w // 3, result.row * h // 3
    x1, y1 = (result.col + 1) * w // 3 - 1, (result.row + 1) * h // 3 - 1
    dwell = result.dwell
    selected = dwell is not None and dwell.progress >= 1.0
    cv2.rectangle(frame, (x0, y0), (x1, y1), GREEN, 8 if selected else 2)
    if dwell is not None and not selected:
        # Dwell progress: a bar growing along the bottom of the active cell
        cv2.rectangle(frame, (x0, y1 - 10),
                      (x0 + int(dwell.progress * (x1 - x0)), y1), GREEN, -1)

    # Head anchor (where a straight-ahead gaze lands) and gaze point
    cv2.drawMarker(frame, norm_to_pixels(*result.anchor, w, h),
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

def handle_key(key, engine):
    """Handle one key press. Returns False when the user quits."""
    if key in (ord('c'), ord('C')):
        engine.recalibrate()
    elif key == ord(']'):
        engine.gain_y_down += 0.1
    elif key == ord('['):
        engine.gain_y_down = max(0.1, engine.gain_y_down - 0.1)
    elif key == ord('='):
        engine.gain_y_up += 0.1
    elif key == ord('-'):
        engine.gain_y_up = max(0.1, engine.gain_y_up - 0.1)
    return key != 27

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
    return p.parse_args(argv)

# ---- Main ------------------------------------------------------------------
def run(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    engine = GazeEngine(calibrate=not args.no_calibrate, dwell_s=args.dwell,
                        screen_size=args.screen_size)
    exporter = GazeExporter(args.export) if args.export else None
    print("ESC to quit, C to recalibrate\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = engine.process(frame)
            if exporter is not None:
                exporter.add(result)
            if result.dwell is not None and result.dwell.selected:
                print(f"selected {result.dwell.region}")
            cv2.imshow(WINDOW, draw(frame, result, engine))
            if not handle_key(cv2.waitKey(1) & 0xFF, engine):
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
