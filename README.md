# Project Iris

Webcam gaze tracking demo. It tracks your pupils with MediaPipe Face Mesh, lifts the pupil displacement onto the unit sphere, filters it there, and projects the result onto the screen. Each tracked frame is also classified into one of nine regions on a 3x3 grid (`top left` … `bottom right`).

## Requirements

- Python 3.9 – 3.12 (developed on 3.11)
- A webcam
- macOS, Linux, or Windows

## Setup

```bash
git clone https://github.com/racheleasen/iris_vision.git
cd iris_vision

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install opencv-python mediapipe numpy
```

The demo uses MediaPipe's `solutions.face_mesh` API. It was developed with `mediapipe 0.10.21`, `opencv-python 4.10`, and `numpy 1.24`; if a newer MediaPipe release fails with an error about `mediapipe.solutions`, pin that version:

```bash
pip install mediapipe==0.10.21
```

On macOS, the first run asks for camera permission for your terminal app. Grant it under **System Settings → Privacy & Security → Camera**, then run the program again.

## Running

```bash
python3 src/demo.py
```

A preview window opens showing the camera feed with the gaze point, the 3x3 region grid, and the region you are currently looking at. Press **ESC** to quit.

### Calibration

On start, the preview dims and shows a target in its centre. Look at the dot and hold still: after about a second the ring around it fills while your pupil position is captured (about two seconds in total). That pose becomes "looking at the centre" for you, your head position, and your camera. If your face is lost or too few frames are captured, it starts over on its own.

- Keep the preview window in the middle of your screen (or maximised) so the target really is at the screen centre.
- Press **C** to recalibrate at any time, for example after shifting in your seat.
- Pass `--no-calibrate` to skip it. The demo then uses a slowly adapting baseline, which drifts toward wherever you keep looking.

After calibration you can still move your head: the centre point shifts with your eyes relative to where they were during calibration.

### Saving gaze data to JSON

Gaze samples are not printed to the terminal. To record them, pass `--export`:

```bash
python3 src/demo.py --export gaze.json
python3 src/demo.py --export gaze.json --screen-size 1920x1080
```

The file is written when you quit.

### Options

| Option | Description |
| --- | --- |
| `--export PATH` | Write per-frame gaze samples to this JSON file on exit. |
| `--screen-size WxH` | Screen size in pixels for the exported `x`/`y`, e.g. `1920x1080`. Defaults to the camera frame size. |
| `--camera N` | Camera index. Defaults to `0`. |
| `--no-calibrate` | Skip the centre calibration on start. Press `C` to run it later. |

### Keys

| Key | Action |
| --- | --- |
| `ESC` | Quit |
| `C` | Recalibrate (look at the centre target) |
| `[` / `]` | Decrease / increase vertical gain when looking down |
| `-` / `=` | Decrease / increase vertical gain when looking up |

## Preview window

- **Green dot**: estimated gaze point
- **Green rectangle**: active region of the 3x3 grid
- **Yellow cross**: where a straight-ahead (calibrated centre) gaze currently lands

Your head does not need to be centred in the camera frame. Once calibrated, the neutral gaze is the screen centre for the head position you calibrated in and follows your head from there. Without calibration it is anchored to where your eyes sit in the frame. Either way, the gaze range on each side of that anchor is stretched to reach its screen edge.

## Export format

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

| Field | Meaning |
| --- | --- |
| `calibration` | The most recent centre calibration (`t` = seconds since start when it completed), or `null` if none was run. |
| `frame` | Camera frame index. Frames with no face are skipped, so gaps mean tracking was lost. Frames shown while the calibration target is up are skipped too. |
| `t` | Seconds since the program started. |
| `x`, `y` | Gaze point in pixels on the screen given by `--screen-size`. |
| `nx`, `ny` | The same point normalised to 0..1. |
| `region` | Which of the nine grid regions the gaze is in. |

Coordinates use a top-left origin with x to the right and y down, from your point of view: looking at the left of the screen gives a small `x`.

## Troubleshooting

- **`Could not open camera 0`**: another app is using the camera, camera permission was not granted, or the index is wrong. Try `--camera 1`.
- **Gaze is offset even when looking at the centre**: press `C` and recalibrate with the preview window centred on your screen.
- **Gaze moves the wrong way horizontally**: your camera delivers a mirrored feed. Set `HEAD_MIRROR_X = False` near the top of `src/demo.py`.
- **Vertical gaze under- or overshoots**: adjust the gains live with `[` `]` and `-` `=`, then set `GAIN_Y_DOWN` / `GAIN_Y_UP` in `src/demo.py` to keep them.

## License

[MIT](LICENSE)
