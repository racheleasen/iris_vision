# iris vision

**Gaze tracking with nothing more than a standard webcam.**

iris vision is an experimental, open-source Python demo. It watches your eyes through your webcam, estimates where on the screen you are looking, and sorts that into one of nine regions. There is no special hardware, and calibration takes about two seconds.

Dedicated eye trackers can be very accurate, but their cameras, infrared lighting, and cost put them out of reach for many people. iris vision asks a different question: **what useful communication capabilities can we build with the webcam people already own?**

> This is a research prototype, not a replacement for dedicated assistive technology. See [Limitations](#limitations).

## Quick start

You need Python **3.9 – 3.12** and a webcam. It works on macOS, Linux, and Windows.

```shell
git clone https://github.com/racheleasen/iris_vision.git
cd iris_vision
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 main.py
```

A window opens showing your camera feed. Press **ESC** to quit.

<details>
<summary>New to Python, or something didn't work?</summary>

- **Check your Python version** with `python3 --version` (on Windows, `python --version`). If it is missing or too old, get it from [python.org](https://www.python.org/downloads/).
- **What the `venv` lines do:** they create a virtual environment, a private folder for this project's libraries (OpenCV, MediaPipe, and NumPy) so they don't mix with the rest of your system.
- **On macOS**, the first run asks for camera permission for your terminal app. Allow it under **System Settings → Privacy & Security → Camera**, then run `python3 main.py` again.
- For anything else, see [Troubleshooting](#troubleshooting).

</details>

## Your first run

1. **Calibrate.** The preview dims and shows a dot in the centre. Look at the dot and hold still for about two seconds while the ring around it fills. This teaches the demo what "looking at the centre" means for you, your seat, and your camera.
2. **Look around.** The preview now shows:
   - **Green dot**: where the demo thinks you are looking
   - **Green rectangle**: which of the nine regions that is
   - **Yellow cross**: where a straight-ahead gaze currently lands
3. **Press C** to calibrate again whenever the dot feels off, for example after shifting in your seat.

Two tips for good results:

- Keep the preview window in the middle of your screen (or maximised), so the calibration dot really is at the screen centre.
- Your head does not need to be centred in the camera, and you can move it after calibrating. The centre point follows your head.

If your face is lost during calibration, it starts over on its own.

## Going further

Everything below is optional. Open the parts you need.

### Keys

Press these while the preview window is selected.

| Key | Action |
| --- | --- |
| `ESC` | Quit |
| `C` | Recalibrate (look at the centre dot) |
| `[` / `]` | Decrease / increase how far the dot moves when you look down |
| `-` / `=` | Decrease / increase how far the dot moves when you look up |

### Command-line options

Add these after `python3 main.py`. You can combine them.

| Option | What it does |
| --- | --- |
| `--export PATH` | Save every gaze sample to this JSON file when you quit. |
| `--screen-size WxH` | Your screen size in pixels, used for the saved `x` and `y`, e.g. `1920x1080`. Defaults to the camera frame size. |
| `--camera N` | Which camera to use, if you have more than one. Defaults to `0`. |
| `--no-calibrate` | Skip the calibration on start. Press `C` to run it later. |
| `--dwell SECONDS` | Select a region by resting your gaze on it for this long, e.g. `1.0`. Off by default. |

For example:

```shell
python3 main.py --export gaze.json --screen-size 1920x1080
```

Without calibration, the demo uses a slowly adapting baseline that drifts toward wherever you keep looking, so calibrating gives better results.

### Selecting with your eyes (dwell)

With `--dwell SECONDS`, keeping your gaze in one region for that long selects it:

```shell
python3 main.py --dwell 1.0
```

A bar fills along the bottom of the active region while you hold your gaze there. When it is full, the region's outline turns thick, the label changes from `looking` to `selected`, and `selected <region>` is printed in the terminal.

A region is selected once per look: to select it again, look away and come back. A blink does not restart the timer, but losing your face for more than about a third of a second does.

### The nine regions

```
top left      top center      top right
middle left   middle center   middle right
bottom left   bottom center   bottom right
```

### Reference

<details>
<summary><strong>The export file</strong>: what <code>--export</code> saves</summary>

`--export` writes one JSON file that looks like this:

```json
{
  "format": "project-iris-gaze",
  "version": 1,
  "created": "2026-01-01T12:00:00+00:00",
  "screen": {"width": 1920, "height": 1080},
  "coordinates": "origin top-left, x right, y down, user's point of view",
  "regions": ["top left", "top center", "..."],
  "calibration": {"type": "single-point", "target": "center", "t": 2.113},
  "dwell_seconds": 1.0,
  "selections": [
    {"frame": 97, "t": 3.2333, "region": "middle center"}
  ],
  "samples": [
    {"frame": 12, "t": 0.412, "x": 963, "y": 547,
     "nx": 0.5016, "ny": 0.5065, "region": "middle center"}
  ]
}
```

Each entry in `samples` is one camera frame:

| Field | Meaning |
| --- | --- |
| `frame` | Camera frame number. Frames with no face are skipped, so a gap means tracking was lost. Frames during calibration are skipped too. |
| `t` | Seconds since the first camera frame. |
| `x`, `y` | Gaze point in pixels, on the screen size given by `--screen-size`. |
| `nx`, `ny` | The same point as a number from 0 to 1, independent of screen size. |
| `region` | Which of the nine regions the gaze is in. |

`x` and `y` start at the top-left corner of the screen: `x` grows to the right and `y` grows downward, from your point of view. Looking at the left of the screen gives a small `x`.

`calibration` describes the most recent calibration (`t` is when it finished), or is `null` if you never calibrated.

`dwell_seconds` is the `--dwell` time, or `null` if dwell was off. `selections` lists every dwell selection: the frame and time the dwell completed, and the region that was selected.

</details>

<details>
<summary><strong>Tuning</strong>: settings you can change in the code</summary>

The settings live as constants near the top of the two files in `src/`. The ones you are most likely to change:

| Constant | File | What it does |
| --- | --- | --- |
| `GAIN_Y_DOWN`, `GAIN_Y_UP` | `gaze_math.py` | How far the dot moves when you look down or up. Find good values live with `[` `]` and `-` `=`, then set them here to keep them. |
| `HEAD_MIRROR_X` | `gaze_math.py` | Set to `False` if your camera delivers a mirrored feed. |
| `SMOOTH_ALPHA` | `gaze_engine.py` | How much the gaze point is smoothed. Lower is steadier but slower to react. |

</details>

<details>
<summary><strong>Using the engine in your own interface</strong></summary>

`main.py` is one interface on top of the engine; you can build another. Feed the engine camera frames and it returns a `GazeResult` for each:

```python
import cv2
from src.gaze_engine import GazeEngine

cap = cv2.VideoCapture(0)
with GazeEngine(dwell_s=1.0, screen_size=(1920, 1080)) as engine:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = engine.process(frame)
        if result.status == "tracking":
            print(result.x, result.y, result.region)
        if result.dwell and result.dwell.selected:
            print("selected", result.dwell.region)
```

| Field | Meaning |
| --- | --- |
| `frame`, `t` | Frame number and seconds since the first frame. |
| `status` | `"calibrating"` while the centre target should be shown, `"no_face"` when no face was found, otherwise `"tracking"`. |
| `face`, `calibrated` | Whether a face was found in this frame, and whether a calibration has completed. |
| `calibration` | While calibrating: `capturing` (`False` while the eyes settle) and `progress` (0 to 1). Otherwise `None`. |
| `nx`, `ny`, `x`, `y` | The gaze point, as in the export file. `x` and `y` use the engine's `screen_size`. |
| `region`, `col`, `row` | The region name and its grid cell, each 0 to 2 from the top-left. |
| `anchor` | Where a straight-ahead gaze currently lands, as `(nx, ny)`. |
| `dwell` | With dwell on: `region`, `elapsed` seconds, `progress` (0 to 1), and `selected`, which is `True` only on the frame the dwell completes. Otherwise `None`. |

The gaze fields from `nx` to `dwell` are `None` unless `status` is `"tracking"`. `result.to_dict()` returns the same data as a plain dictionary, ready for `json.dumps`. Call `engine.recalibrate()` to run the calibration again, and set `engine.gain_y_up` / `engine.gain_y_down` to change the vertical gain.

</details>

## How it works

1. [MediaPipe Face Mesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) finds your pupils and irises in each camera frame.
2. The demo measures how far each pupil has moved from its calibrated centre position.
3. That movement is lifted onto a unit sphere (a model of the eyeball rotating) and smoothed there.
4. The result is projected onto the screen as a gaze point and sorted into one of the nine regions.

The code is three files:

| File | What it holds |
| --- | --- |
| `main.py` | The program you run: camera loop, preview window, keys, and JSON export. |
| `src/gaze_engine.py` | The gaze engine. Takes camera frames, keeps the tracking state (calibration, smoothing, dwell timer), and returns a result for each frame. No window or camera code. |
| `src/gaze_math.py` | The math: sphere lift and filter, screen projection, head anchor, and the nine regions. Needs only NumPy. |

## Troubleshooting

- **`Could not open camera 0`**: another app is using the camera, camera permission was not granted, or the camera number is wrong. Try `--camera 1`.
- **The dot is off even when you look at the centre**: press `C` and recalibrate with the preview window centred on your screen.
- **The dot moves the wrong way left and right**: your camera delivers a mirrored feed. Set `HEAD_MIRROR_X = False` near the top of `src/gaze_math.py`.
- **The dot moves too little or too much up and down**: adjust it live with `[` `]` and `-` `=`.

## Limitations

iris vision is an experimental gaze-tracking prototype. It explores what consumer webcams can do; it does not provide clinically validated eye tracking.

- Single-point calibration gives an approximate screen-relative gaze estimate.
- Accuracy varies with webcam quality, lighting, viewing distance, and camera positioning.
- Glasses, reflections, eyelid occlusion, and individual differences can affect tracking.
- Extended gaze interaction may cause eye fatigue.
- It estimates gaze position but is not yet a complete communication interface.
- No clinical accuracy or usability validation has been performed.

## Want to help?

Bug reports, pull requests, experiments, and ideas are all welcome. If the tracking works poorly for you, please open an issue and mention your camera, lighting, and whether you wear glasses; that is the most useful information for improving it.

### Support project iris

iris vision is affiliated with [project iris](https://www.projectiris.app/), and the best way to support this work is to support project iris.

project iris is building a free gaze communication tool for people who rely on non-verbal communication, using only a webcam. For many people with motor or speech impairments, eye-tracking communication systems can be life-changing, but commercial ones often need specialized hardware and can cost $10,000–$15,000 or more. project iris asks how much of that capability can be reached with everyday consumer technology, and its vision is to make gaze-based communication accessible to anyone with a standard webcam.

You can help in two ways:

- **Contribute** on the [project iris support page](https://www.projectiris.app/support). Contributions go toward research, development, and keeping gaze-assisted communication free.
- **Try it and share it.** The gaze tools run in your browser at [projectiris.app](https://www.projectiris.app/), with no install and no account.

project iris is an experimental, non-clinical research initiative. It is not a medical device and does not replace clinically approved communication (AAC) systems.

## License

Released under the [MIT license](LICENSE).
