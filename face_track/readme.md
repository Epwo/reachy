# face_track — Reachy follows and recognizes faces

The robot's head follows the most relevant face (a known person first,
otherwise the largest face) and recognizes enrolled people (InsightFace
embeddings). It only moves once the face leaves a dead zone in the middle of
the frame, so it doesn't fidget constantly.

## Setup

```bash
cd face_track
uv venv .venv_reachy --python 3.12
uv pip install --python .venv_reachy/bin/python -r requirements.txt
source .venv_reachy/bin/activate
```

The InsightFace model pack (`buffalo_l` by default) downloads itself to
`~/.insightface/` on first launch.

## Usage

```bash
# 1. Enroll a face (robot camera) — SPACE = capture, Q = done
python enroll.py Ewann
python enroll.py Ewann --forget              # remove them

# 2. Follow + recognize on the robot — Q = quit, R = recenter the head
python main.py

# 3. Tune tracking live with sliders — S = save, R = recenter, Q = quit
python experiment.py --webcam 0              # Mac webcam, robot connected
python experiment.py --webcam 0 --no-robot   # no robot, detection only
python experiment.py --video clip.mp4
```

`experiment.py` saves its settings (**S**) to `tracking_params.json`, which
`main.py` loads automatically.

## Settings (`tracking_params.json`)

| Key | What it does |
|---|---|
| `dead_zone` | offset (fraction of the frame) at which the head starts moving |
| `recenter_zone` | offset under which it stops (must be < `dead_zone`) |
| `tracking_speed` | smoothing of the head pose (higher = snappier) |
| `max_step_deg` | max rotation per frame in degrees (caps the speed) |
| `detection_smoothing` | smoothing of the detected position (anti-jitter) |
| `flip_u` | mirror the horizontal axis if the head turns the wrong way |

## Files

```
face_track/
├── main.py                follow + recognize on the robot
├── enroll.py              enroll / forget a person
├── experiment.py          live tuning bench (image, video or webcam)
├── tracking_params.json   settings loaded by main.py
└── src/
    ├── face_tracker.py    detection, recognition, target selection, head control
    └── known_faces/       .npy embeddings of known people (gitignored)
```
