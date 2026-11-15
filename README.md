# iris vision

![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

iris vision is a Python (3.9 – 3.12) demo that turns an ordinary webcam into a **real-time eye tracker**. It shows where on the screen you are looking, tells you which of nine screen regions that is (`top left` … `bottom right`), and can save every gaze point to a JSON file.

No special hardware is needed, and everything runs on your own computer.

## Installation

You need Python **3.9 – 3.12** and a webcam. It works on macOS, Linux, and Windows.

Not sure which Python you have? Run `python3 --version` (on Windows, `python --version`). If it is missing or too old, get it from [python.org](https://www.python.org/downloads/).

Download the project:

```shell
git clone https://github.com/racheleasen/iris_vision.git
cd iris_vision
```

Then install the three libraries it uses. The first two lines create a virtual environment, a private folder for this project's libraries so they don't mix with the rest of your system:

```shell
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install opencv-python mediapipe numpy
```

### Run the demo

```shell
python3 src/demo.py
```

A window opens showing your camera feed. Press **ESC** to quit.

> **On macOS**, the first run asks for camera permission for your terminal app. Allow it under **System Settings → Privacy & Security → Camera**, then run the command again.

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

## Documentation

### Command-line options

Add these after `python3 src/demo.py`. You can combine them.

| Option | What it does |
| --- | --- |
| `--export PATH` | Save every gaze sample to this JSON file when you quit. |
| `--screen-size WxH` | Your screen size in pixels, used for the saved `x` and `y`, e.g. `1920x1080`. Defaults to the camera frame size. |
| `--camera N` | Which camera to use, if you have more than one. Defaults to `0`. |
| `--no-calibrate` | Skip the calibration on start. Press `C` to run it later. |

For example:

```shell
python3 src/demo.py --export gaze.json --screen-size 1920x1080
```

Without calibration, the demo uses a slowly adapting baseline that drifts toward wherever you keep looking, so calibrating gives better results.

### Keys

Press these while the preview window is selected.

| Key | Action |
| --- | --- |
| `ESC` | Quit |
| `C` | Recalibrate (look at the centre dot) |
| `[` / `]` | Decrease / increase how far the dot moves when you look down |
| `-` / `=` | Decrease / increase how far the dot moves when you look up |

### The export file

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
| `t` | Seconds since the program started. |
| `x`, `y` | Gaze point in pixels, on the screen size given by `--screen-size`. |
| `nx`, `ny` | The same point as a number from 0 to 1, independent of screen size. |
| `region` | Which of the nine regions the gaze is in. |

`x` and `y` start at the top-left corner of the screen: `x` grows to the right and `y` grows downward, from your point of view. Looking at the left of the screen gives a small `x`.

`calibration` describes the most recent calibration (`t` is when it finished), or is `null` if you never calibrated.

### The nine regions

```
top left      top center      top right
middle left   middle center   middle right
bottom left   bottom center   bottom right
```

### Tuning

The settings live as constants near the top of `src/demo.py`. The ones you are most likely to change:

| Constant | What it does |
| --- | --- |
| `GAIN_Y_DOWN`, `GAIN_Y_UP` | How far the dot moves when you look down or up. Find good values live with `[` `]` and `-` `=`, then set them here to keep them. |
| `HEAD_MIRROR_X` | Set to `False` if your camera delivers a mirrored feed. |
| `SMOOTH_ALPHA` | How much the gaze point is smoothed. Lower is steadier but slower to react. |

## How it works

1. [MediaPipe Face Mesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) finds your pupils and irises in each camera frame.
2. The demo measures how far each pupil has moved from its calibrated centre position.
3. That movement is lifted onto a unit sphere (a model of the eyeball rotating) and smoothed there.
4. The result is projected onto the screen as a gaze point and sorted into one of the nine regions.

All of the code is in one file, `src/demo.py`.

## Troubleshooting

- **`Could not open camera 0`**: another app is using the camera, camera permission was not granted, or the camera number is wrong. Try `--camera 1`.
- **The dot is off even when you look at the centre**: press `C` and recalibrate with the preview window centred on your screen.
- **The dot moves the wrong way left and right**: your camera delivers a mirrored feed. Set `HEAD_MIRROR_X = False` near the top of `src/demo.py`.
- **The dot moves too little or too much up and down**: adjust it live with `[` `]` and `-` `=`.

## You want to help?

Bug reports and pull requests are welcome. If the tracking works poorly for you, please open an issue and mention your camera, lighting, and whether you wear glasses; that is the most useful information for improving it.

## License

Released under the [MIT license](LICENSE).
